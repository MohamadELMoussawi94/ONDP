import streamlit as st
import os
import io
import zipfile
import glob
import shutil
import pandas as pd
import pdfplumber
import re
import math
import gc
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import tempfile
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Oman National Development Plan - Similarity Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f2f6, #ffffff);
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    
    .step-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 1.5rem 0 1rem 0;
        padding: 0.5rem;
        background-color: #ecf0f1;
        border-radius: 5px;
        border-left: 4px solid #3498db;
    }
    
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'similarity_results' not in st.session_state:
    st.session_state.similarity_results = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

# Configuration
CANDIDATE_LABELS = [
    "People and society",
    "Economy and development", 
    "Environment",
    "Governance and institutional performance",
]

MULTI_LABEL = True
MULTI_LABEL_THRESHOLD = 0.45
SCORE_THRESHOLD = 0.45
HYPOTHESIS = "This text is about {}."

class PDFProcessor:
    @staticmethod
    def split_into_sentences(text: str):
        """Robust sentence splitter"""
        if not text:
            return []

        # Normalize whitespace & bullets
        text = text.replace("\r", " ")
        text = re.sub(r'(?m)^\s*[•\-\u2022\u25CF]\s*', '', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n+', '\n', text)

        # Split on sentence punctuation or newlines
        parts = re.split(r'(?<=[.!?])\s+|\n+', text)

        # Split long clauses on semicolons
        sentences = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if ';' in p and len(p.split()) > 25:
                sub = [s.strip() for s in p.split(';') if s.strip()]
                sentences.extend(sub)
            else:
                sentences.append(p)

        # Filter out short/empty sentences
        clean = []
        for s in sentences:
            if s.isdigit() or len(s) < 3:
                continue
            s = re.sub(r'\s+', ' ', s).strip()
            if s:
                clean.append(s)
        return clean

    @staticmethod
    def extract_pdf_sentences(pdf_path: str):
        """Extract sentences per page for a single PDF"""
        rows = []
        dedup = set()
        with pdfplumber.open(pdf_path) as pdf:
            for p_idx, page in enumerate(pdf.pages, start=1):
                txt = page.extract_text() or ""
                sents = PDFProcessor.split_into_sentences(txt)
                for i, s in enumerate(sents, start=1):
                    key = s.strip().lower()
                    if key in dedup:
                        continue
                    dedup.add(key)
                    rows.append({"page": p_idx, "sentence_id": i, "sentence": s})
        return rows

class SimilarityAnalyzer:
    def __init__(self):
        self.model = None
        self.zero_shot = None
    
    @st.cache_resource
    def load_models(_self):
        """Load models with caching"""
        device = 0 if torch.cuda.is_available() else -1
        zero_shot = pipeline(
            "zero-shot-classification",
            model="joeddav/xlm-roberta-large-xnli",
            device=device
        )
        sentence_model = SentenceTransformer("all-mpnet-base-v2")
        return zero_shot, sentence_model
    
    def clean_sentence(self, s: str) -> str:
        if not isinstance(s, str):
            return ""
        s = s.strip()
        s = re.sub(r"\s+", " ", s)
        s = "".join(ch for ch in s if ch.isprintable())
        return s
    
    def chunk_iterable(self, lst, size):
        for i in range(0, len(lst), size):
            yield lst[i:i+size]
    
    def classify_sentences(self, df):
        """Classify sentences using zero-shot classification"""
        if self.zero_shot is None:
            self.zero_shot, _ = self.load_models()
        
        df["sentence_clean"] = df["sentence"].apply(self.clean_sentence)
        mask_valid = df["sentence_clean"].str.len() > 0
        idx_valid = df.index[mask_valid].tolist()
        texts = df.loc[idx_valid, "sentence_clean"].tolist()
        
        all_results = [None] * len(df)
        BATCH = 16
        
        progress_bar = st.progress(0)
        total_batches = math.ceil(len(texts) / BATCH)
        
        for batch_idx, chunk in enumerate(self.chunk_iterable(texts, BATCH)):
            progress_bar.progress((batch_idx + 1) / total_batches)
            
            res = self.zero_shot(
                sequences=chunk,
                candidate_labels=CANDIDATE_LABELS,
                hypothesis_template=HYPOTHESIS,
                multi_label=MULTI_LABEL
            )
            
            if isinstance(res, dict):
                res = [res]
            
            start = batch_idx * BATCH
            for k, r in enumerate(res):
                tgt_idx = idx_valid[start + k]
                all_results[tgt_idx] = r
        
        # Initialize columns
        for lab in CANDIDATE_LABELS:
            df[f"zs_score::{lab}"] = None
            if MULTI_LABEL:
                df[f"zs_flag::{lab}"] = 0
        
        df["zs_label"] = None
        df["zs_label_score"] = None
        if MULTI_LABEL:
            df["zs_multi_labels"] = ""
        
        # Fill columns from results
        for i, r in enumerate(all_results):
            if r is None:
                continue
            labels = r["labels"]
            scores = r["scores"]
            
            score_map = {lab: sc for lab, sc in zip(labels, scores)}
            for lab in CANDIDATE_LABELS:
                df.at[i, f"zs_score::{lab}"] = float(score_map.get(lab, 0.0))
            
            # Top-1
            top_idx = int(max(range(len(labels)), key=lambda k: scores[k]))
            df.at[i, "zs_label"] = labels[top_idx]
            df.at[i, "zs_label_score"] = float(scores[top_idx])
            
            # Multi-label assignment
            if MULTI_LABEL:
                assigned = [lab for lab in CANDIDATE_LABELS if score_map.get(lab, 0.0) >= MULTI_LABEL_THRESHOLD]
                for lab in assigned:
                    df.at[i, f"zs_flag::{lab}"] = 1
                df.at[i, "zs_multi_labels"] = ", ".join(assigned)
        
        progress_bar.empty()
        return df
    
    def sentences_for_label(self, df: pd.DataFrame, label: str) -> list[str]:
        """Return sentences assigned to a specific label"""
        df = df.copy()
        df["sentence_clean"] = df["sentence"].apply(self.clean_sentence)
        
        flag_col = f"zs_flag::{label}"
        score_col = f"zs_score::{label}"
        
        if flag_col in df.columns:
            sub = df[df[flag_col] == 1].copy()
            if score_col in df.columns:
                sub = sub[pd.to_numeric(sub[score_col], errors="coerce").fillna(0.0) >= SCORE_THRESHOLD]
            sents = sub["sentence_clean"].dropna().tolist()
        else:
            sents = []
        
        # Deduplicate
        seen = set()
        uniq = []
        for s in sents:
            k = s.lower()
            if len(k) == 0 or k in seen:
                continue
            seen.add(k)
            uniq.append(s)
        return uniq
    
    def symmetric_alignment(self, embA: np.ndarray, embB: np.ndarray) -> dict:
        """Compute symmetric similarity between two sets of embeddings"""
        if embA.size == 0 or embB.size == 0:
            return {"a_to_b": np.nan, "b_to_a": np.nan, "symmetric": np.nan}
        
        S = cosine_similarity(embA, embB)
        a_to_b = S.max(axis=1).mean()
        b_to_a = S.max(axis=0).mean()
        return {"a_to_b": float(a_to_b), "b_to_a": float(b_to_a), "symmetric": float((a_to_b + b_to_a)/2)}
    
    def top_pairs(self, embA, embB, sentsA, sentsB, k=10):
        """Return top-k most similar sentence pairs"""
        if embA.size == 0 or embB.size == 0:
            return []
        
        S = cosine_similarity(embA, embB)
        flat = []
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                flat.append((i, j, S[i, j]))
        
        flat.sort(key=lambda x: x[2], reverse=True)
        out = []
        usedA, usedB = set(), set()
        
        for i, j, sc in flat:
            if i in usedA or j in usedB:
                continue
            out.append({
                "Plan_A_sentence": sentsA[i], 
                "Plan_B_sentence": sentsB[j], 
                "similarity": float(sc)
            })
            usedA.add(i)
            usedB.add(j)
            if len(out) >= k:
                break
        return out

def main():
    # Main header
    st.markdown('<div class="main-header">📊 Oman National Development Plan (Similarity Analysis)</div>', 
                unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.info("Upload a ZIP file containing PDF development plans to start the analysis.")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a ZIP file", 
            type=['zip'],
            help="Upload a ZIP file containing PDF development plans"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        st.markdown("---")
        st.markdown("**Analysis Categories:**")
        for i, label in enumerate(CANDIDATE_LABELS, 1):
            st.markdown(f"{i}. {label}")
    
    if uploaded_file is None:
        st.markdown('<div class="info-box">👆 Please upload a ZIP file containing PDF development plans to begin the analysis.</div>', 
                    unsafe_allow_html=True)
        return
    
    # Process uploaded file
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract ZIP file
        zip_path = os.path.join(temp_dir, "plans.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        plans_dir = os.path.join(temp_dir, "plans")
        os.makedirs(plans_dir, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(plans_dir)
        
        # Find PDF files
        pdf_files = sorted(glob.glob(os.path.join(plans_dir, "**", "*.pdf"), recursive=True))
        
        if len(pdf_files) == 0:
            st.error("❌ No PDF files found in the uploaded ZIP file.")
            return
        
        if len(pdf_files) < 2:
            st.warning("⚠️ At least 2 PDF files are required for similarity analysis.")
            return
        
        # Step 1: PDF Discovery and Validation
        st.markdown('<div class="step-header">📄 Step 1: PDF Discovery and Validation</div>', 
                    unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"**Found {len(pdf_files)} PDF file(s):**")
            for i, pdf_path in enumerate(pdf_files, 1):
                filename = os.path.basename(pdf_path)
                st.markdown(f"{i}. `{filename}`")
        
        with col2:
            # Validate PDFs
            st.markdown("**Validation Status:**")
            valid_pdfs = []
            for pdf_path in pdf_files:
                try:
                    with pdfplumber.open(pdf_path) as pdf:
                        num_pages = len(pdf.pages)
                    valid_pdfs.append(pdf_path)
                    st.success(f"✅ {os.path.basename(pdf_path)}")
                except Exception as e:
                    st.error(f"❌ {os.path.basename(pdf_path)}: {str(e)}")
            
            pdf_files = valid_pdfs
        
        if len(pdf_files) < 2:
            st.error("❌ Not enough valid PDF files for comparison.")
            return
        
        # Step 2: Text Extraction
        st.markdown('<div class="step-header">📝 Step 2: Text Extraction and Sentence Splitting</div>', 
                    unsafe_allow_html=True)
        
        processor = PDFProcessor()
        plan_data = {}
        
        extraction_progress = st.progress(0)
        extraction_status = st.empty()
        
        for i, pdf_path in enumerate(pdf_files):
            plan_name = os.path.splitext(os.path.basename(pdf_path))[0]
            extraction_status.text(f"Processing: {plan_name}")
            
            rows = processor.extract_pdf_sentences(pdf_path)
            df = pd.DataFrame(rows, columns=["page", "sentence_id", "sentence"])
            plan_data[plan_name] = df
            
            extraction_progress.progress((i + 1) / len(pdf_files))
        
        extraction_status.empty()
        extraction_progress.empty()
        
        # Display extraction results
        col1, col2 = st.columns(2)
        plan_names = list(plan_data.keys())
        
        for i, (plan_name, df) in enumerate(plan_data.items()):
            with col1 if i % 2 == 0 else col2:
                st.markdown(f"**{plan_name}**")
                st.markdown(f"- **Total sentences:** {len(df)}")
                st.markdown(f"- **Total pages:** {df['page'].max()}")
                
                # Show sample sentences
                with st.expander(f"Sample sentences from {plan_name}"):
                    sample_df = df.head(5)[['page', 'sentence']]
                    st.dataframe(sample_df, use_container_width=True)
        
        # Step 3: Sentence Classification
        st.markdown('<div class="step-header">🏷️ Step 3: Sentence Classification</div>', 
                    unsafe_allow_html=True)
        
        analyzer = SimilarityAnalyzer()
        classified_data = {}
        
        classification_status = st.empty()
        
        for plan_name, df in plan_data.items():
            classification_status.markdown(f"**Classifying sentences for:** `{plan_name}`")
            classified_df = analyzer.classify_sentences(df.copy())
            classified_data[plan_name] = classified_df
        
        classification_status.empty()
        
        # Display classification results
        st.markdown("**Classification Results:**")
        
        # Create a summary table
        summary_data = []
        for plan_name, df in classified_data.items():
            for label in CANDIDATE_LABELS:
                count = df[f"zs_flag::{label}"].sum() if f"zs_flag::{label}" in df.columns else 0
                summary_data.append({
                    "Plan": plan_name,
                    "Category": label,
                    "Sentences": int(count)
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Create visualization for classification results
        fig = px.bar(
            summary_df, 
            x="Category", 
            y="Sentences", 
            color="Plan",
            title="Sentence Distribution by Category",
            labels={"Sentences": "Number of Sentences"},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            height=500,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed breakdown
        pivot_df = summary_df.pivot(index="Category", columns="Plan", values="Sentences").fillna(0)
        st.markdown("**Detailed Breakdown:**")
        st.dataframe(pivot_df, use_container_width=True)
        
        # Step 4: Similarity Analysis
        if len(classified_data) >= 2:
            st.markdown('<div class="step-header">🔍 Step 4: Similarity Analysis</div>', 
                        unsafe_allow_html=True)
            
            # For now, compare first two plans
            plan_names_list = list(classified_data.keys())
            planA_name, planB_name = plan_names_list[0], plan_names_list[1]
            dfA, dfB = classified_data[planA_name], classified_data[planB_name]
            
            st.markdown(f"**Comparing:** `{planA_name}` vs `{planB_name}`")
            
            # Get sentences by label
            planA_by_label = {lab: analyzer.sentences_for_label(dfA, lab) for lab in CANDIDATE_LABELS}
            planB_by_label = {lab: analyzer.sentences_for_label(dfB, lab) for lab in CANDIDATE_LABELS}
            
            # Load sentence transformer model
            if analyzer.model is None:
                _, analyzer.model = analyzer.load_models()
            
            # Embed sentences
            similarity_status = st.empty()
            similarity_status.markdown("**Computing sentence embeddings...**")
            
            # Collect all sentences for batch embedding
            all_texts = []
            owners = []
            for lab in CANDIDATE_LABELS:
                for s in planA_by_label[lab]:
                    all_texts.append(s)
                    owners.append(("A", lab))
                for s in planB_by_label[lab]:
                    all_texts.append(s)
                    owners.append(("B", lab))
            
            if len(all_texts) > 0:
                emb_all = analyzer.model.encode(all_texts, batch_size=64, convert_to_numpy=True)
                
                # Split embeddings back to plans and labels
                offset = 0
                embA_by_label = {}
                embB_by_label = {}
                
                for lab in CANDIDATE_LABELS:
                    nA = len(planA_by_label[lab])
                    nB = len(planB_by_label[lab])
                    
                    embA_by_label[lab] = emb_all[offset:offset+nA] if nA > 0 else np.empty((0, 768))
                    offset += nA
                    embB_by_label[lab] = emb_all[offset:offset+nB] if nB > 0 else np.empty((0, 768))
                    offset += nB
                
                similarity_status.empty()
                
                # Compute similarities
                similarity_results = []
                top_pairs_data = {}
                
                for lab in CANDIDATE_LABELS:
                    embA = embA_by_label[lab]
                    embB = embB_by_label[lab]
                    result = analyzer.symmetric_alignment(embA, embB)
                    
                    similarity_results.append({
                        "Category": lab,
                        "Plan A Sentences": embA.shape[0],
                        "Plan B Sentences": embB.shape[0],
                        "A→B Score": result["a_to_b"],
                        "B→A Score": result["b_to_a"],
                        "Symmetric Score": result["symmetric"],
                        "Similarity %": round(result["symmetric"] * 100, 2) if not np.isnan(result["symmetric"]) else 0
                    })
                    
                    # Get top pairs for each category
                    pairs = analyzer.top_pairs(embA, embB, planA_by_label[lab], planB_by_label[lab], k=5)
                    top_pairs_data[lab] = pairs
                
                # Display similarity results
                similarity_df = pd.DataFrame(similarity_results)
                
                # Create main similarity visualization
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=similarity_df["Category"],
                    y=similarity_df["Similarity %"],
                    text=similarity_df["Similarity %"].apply(lambda x: f"{x:.1f}%"),
                    textposition='auto',
                    marker_color=px.colors.qualitative.Set1[:len(CANDIDATE_LABELS)],
                    name="Similarity %"
                ))
                
                fig.update_layout(
                    title=f"Semantic Similarity by Category<br>{planA_name} vs {planB_name}",
                    xaxis_title="Category",
                    yaxis_title="Similarity Percentage (%)",
                    yaxis=dict(range=[0, 100]),
                    height=500,
                    xaxis_tickangle=-45
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display detailed results table
                st.markdown("**Detailed Similarity Scores:**")
                display_cols = ["Category", "Plan A Sentences", "Plan B Sentences", "Symmetric Score", "Similarity %"]
                st.dataframe(similarity_df[display_cols], use_container_width=True)
                
                # Top similar sentence pairs by category
                st.markdown('<div class="step-header">🎯 Top Similar Sentence Pairs by Category</div>', 
                            unsafe_allow_html=True)
                
                for label in CANDIDATE_LABELS:
                    if top_pairs_data[label]:
                        with st.expander(f"🔍 {label} - Top Similar Pairs"):
                            pairs_df = pd.DataFrame(top_pairs_data[label])
                            pairs_df["Similarity %"] = (pairs_df["similarity"] * 100).round(1)
                            
                            for idx, row in pairs_df.head(3).iterrows():
                                col1, col2, col3 = st.columns([5, 5, 2])
                                with col1:
                                    st.markdown(f"**Plan A:** {row['Plan_A_sentence'][:200]}...")
                                with col2:
                                    st.markdown(f"**Plan B:** {row['Plan_B_sentence'][:200]}...")
                                with col3:
                                    st.metric("Similarity", f"{row['Similarity %']:.1f}%")
                                st.markdown("---")
                
                # Summary insights
                st.markdown('<div class="step-header">📊 Summary Insights</div>', 
                            unsafe_allow_html=True)
                
                # Calculate overall statistics
                avg_similarity = similarity_df["Similarity %"].mean()
                max_similarity = similarity_df["Similarity %"].max()
                min_similarity = similarity_df["Similarity %"].min()
                max_category = similarity_df.loc[similarity_df["Similarity %"].idxmax(), "Category"]
                min_category = similarity_df.loc[similarity_df["Similarity %"].idxmin(), "Category"]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Average Similarity", f"{avg_similarity:.1f}%")
                with col2:
                    st.metric("Highest Similarity", f"{max_similarity:.1f}%", f"{max_category}")
                with col3:
                    st.metric("Lowest Similarity", f"{min_similarity:.1f}%", f"{min_category}")
                with col4:
                    total_sentences = similarity_df["Plan A Sentences"].sum() + similarity_df["Plan B Sentences"].sum()
                    st.metric("Total Sentences", f"{total_sentences:,}")
                
                # Create a heatmap for better visualization
                st.markdown("**Similarity Heatmap:**")
                
                # Prepare data for heatmap
                heatmap_data = similarity_df.set_index("Category")["Similarity %"].values.reshape(1, -1)
                
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=heatmap_data,
                    x=similarity_df["Category"],
                    y=["Similarity"],
                    colorscale="RdYlBu_r",
                    text=[[f"{val:.1f}%" for val in heatmap_data[0]]],
                    texttemplate="%{text}",
                    textfont={"size": 14},
                    colorbar=dict(title="Similarity %")
                ))
                
                fig_heatmap.update_layout(
                    title="Similarity Scores Across Categories",
                    height=200,
                    xaxis_tickangle=-45
                )
                
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                st.success("✅ Analysis completed successfully!")
            
            else:
                st.warning("⚠️ No sentences found for similarity analysis.")

if __name__ == "__main__":
    main()
