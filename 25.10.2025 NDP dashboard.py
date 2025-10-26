import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import io, zipfile, re

# ==============================
# 1) CONFIG
# ==============================
st.set_page_config(
    page_title="ONDP Program Performance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
alt.data_transformers.disable_max_rows()

# Keep table text white (works nicely with dark theme)
st.markdown("""
<style>
[data-testid="stDataFrame"] thead th { color: #FFFFFF !important; }
[data-testid="stDataFrame"] tbody td { color: #FFFFFF !important; }
</style>
""", unsafe_allow_html=True)

# ==============================
# 2) HELPERS
# ==============================
def read_csv_any(uploaded_obj):
    """
    Read a CSV from:
      - streamlit UploadedFile (has .getvalue())
      - bytes / bytearray
      - io.BytesIO
    with tolerant encodings.
    """
    if hasattr(uploaded_obj, "getvalue"):
        content = uploaded_obj.getvalue()
    elif isinstance(uploaded_obj, (bytes, bytearray)):
        content = uploaded_obj
    elif isinstance(uploaded_obj, io.BytesIO):
        content = uploaded_obj.getvalue()
    else:
        raise ValueError("Unsupported uploaded object type")

    for enc in [None, "utf-8", "utf-8-sig", "cp1252", "ISO-8859-1"]:
        try:
            return pd.read_csv(io.BytesIO(content), encoding=enc, low_memory=False)
        except Exception:
            continue
    return pd.read_csv(io.BytesIO(content), low_memory=False)

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    return df

def norm_code(x):
    """Normalize 'Oman Code' to a robust string key."""
    if pd.isna(x): return ""
    s = str(x).strip()
    try:
        v = float(s.replace(",", ""))
        if np.isfinite(v) and float(v).is_integer():
            return str(int(v))
        return s
    except Exception:
        return s

def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def to_unit_interval(series: pd.Series) -> pd.Series:
    """Coerce [0..100] -> [0..1] if needed."""
    s = to_numeric(series)
    if s.notna().any():
        mx = s.max()
        if mx is not None and mx > 1.5:
            s = s / 100.0
    return s

def parse_percentish(series: pd.Series) -> pd.Series:
    """
    Turn values like '85%', '85.0 %', '85,3٪', '٠٫٨٥', '0.85' into floats in [0,1].
    - Strips percent signs (including Arabic ٪)
    - Keeps digits/.-, converts comma to dot
    - If max > 1.5, assumes 0..100 and divides by 100
    """
    s = series.astype(str)
    s = (s.str.replace('٪', '%', regex=False)
           .str.replace('%', '', regex=False)
           .str.strip()
           .str.replace(r'[^0-9\-,\.]', '', regex=True)
           .str.replace(',', '.', regex=False))
    nums = pd.to_numeric(s, errors='coerce')
    if nums.notna().any():
        if nums.max() is not None and nums.max() > 1.5:
            nums = nums / 100.0
    return nums

def classify(v):
    if pd.isna(v): return 'N/A'
    try: v = float(v)
    except Exception: return 'N/A'
    if 0.0 <= v < 0.3:   return 'Low'
    if 0.3 <= v < 0.6:   return 'Medium'
    if 0.6 <= v <= 1.0:  return 'High'
    return 'Other'

def normalize_class_series(s: pd.Series) -> pd.Series:
    mapping = {
        'low':'Low','lo':'Low','l':'Low',
        'medium':'Medium','med':'Medium','m':'Medium',
        'high':'High','hi':'High','h':'High'
    }
    s2 = s.astype(str).str.strip().str.lower()
    s2 = s2.replace({'nan': np.nan, 'none': np.nan, '': np.nan})
    s2 = s2.map(mapping).astype('object')
    return s2

def qualitative_background_pretty(x_title="Budget", y_title="Impact", width=1100, height=480):
    """3×3 qualitative background with screenshot-like palette and visible axes."""
    tiles = pd.DataFrame({
        "x":[1,2,3,1,2,3,1,2,3],
        "y":[1,1,1,2,2,2,3,3,3],
        "color":[
            "#DFF3D7","#BEE7B6","#81D67F",
            "#FBE3D7","#F1C5A8","#E7A884",
            "#FAD1CB","#F5A79C","#F07267"
        ]
    })
    bg = alt.Chart(tiles, width=width, height=height).mark_rect(opacity=0.65).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[0.5,3.5]), axis=None),
        y=alt.Y("y:Q", scale=alt.Scale(domain=[0.5,3.5]), axis=None),
        color=alt.Color("color:N", scale=None, legend=None)
    )
    v = alt.Chart(pd.DataFrame({"x":[1.5,2.5]}), width=width, height=height)\
        .mark_rule(stroke="#FFFFFF", strokeWidth=2).encode(x="x:Q")
    h = alt.Chart(pd.DataFrame({"y":[1.5,2.5]}), width=width, height=height)\
        .mark_rule(stroke="#FFFFFF", strokeWidth=2).encode(y="y:Q")
    center_row = pd.DataFrame({"x":[1,2,3], "y":[2,2,2], "label":["Low","Medium","High"]})
    mid_col    = pd.DataFrame({"x":[1,1,1], "y":[1,2,3], "label":["High","Medium","Low"]})
    center_labels = alt.Chart(center_row, width=width, height=height)\
        .mark_text(fontWeight="bold").encode(x="x:Q", y="y:Q", text="label:N")
    side_labels   = alt.Chart(mid_col, width=width, height=height)\
        .mark_text(align="right", dx=-10, fontWeight="bold").encode(
            x=alt.value(0.5), y="y:Q", text="label:N"
        )
    x_title_text = alt.Chart(pd.DataFrame({"x":[3.4], "y":[0.55], "t":[x_title]}), width=width, height=height)\
        .mark_text(fontSize=16, fontWeight="bold")\
        .encode(x="x:Q", y="y:Q", text="t:N")
    y_title_text = alt.Chart(pd.DataFrame({"x":[0.55], "y":[3.4], "t":[y_title]}), width=width, height=height)\
        .mark_text(fontSize=16, fontWeight="bold", angle=270)\
        .encode(x="x:Q", y="y:Q", text="t:N")
    return bg + v + h + center_labels + side_labels + x_title_text + y_title_text

def safe_int_text(x):
    if pd.isna(x): return ""
    try:
        v = float(x)
        if float(v).is_integer():
            return str(int(v))
    except Exception:
        pass
    return str(x)

# --- ADD: robust numeric & year parsing helpers for Macro Economic ---
def parse_numberish(series: pd.Series) -> pd.Series:
    """
    Convert strings like '1,234', '1٬234', '12.3', '12٫3', '  12  ', etc. to floats.
    Handles Arabic thousands/decimal separators and stray text.
    """
    s = series.astype(str)
    s = (s.str.replace('\u066C', '', regex=False)   # Arabic thousands '٬'
           .str.replace(',', '', regex=False)       # Western thousands ','
           .str.replace('\u066B', '.', regex=False) # Arabic decimal '٫'
           .str.replace('٪', '', regex=False)       # Arabic percent
           .str.replace('%', '', regex=False)
           .str.replace(r'[^\d\.\-\+eE]', '', regex=True) # keep digits, ., -, +, exponent
           .str.strip())
    return pd.to_numeric(s, errors='coerce')

def detect_year_cols(columns) -> list:
    """
    Detect year columns even if they include spaces or non-digit chars.
    Any column whose name contains exactly 4 digits (e.g., ' 2023 ', 'Year 2021') is kept.
    """
    yrs = []
    for c in columns:
        name = str(c).strip()
        digits = re.sub(r'\D', '', name)
        if len(digits) == 4:
            yrs.append(c)  # keep original label to melt
    return yrs

# ---------- ZIP helpers ----------
REQUIRED_FILES = {
    'ONDP Dataset.csv': 'ONDP Dataset.csv (Program Details - long)',
    'PI Mapping V1.csv': 'PI Mapping V1.csv (PI-Code link)',
    'Proxies list ONDP.csv': 'Proxies list ONDP.csv (PI time series)',
    'Macro Economic.csv': 'Macro Economic.csv (macro indicators)',
    'BPI V1.csv': 'BPI V1.csv (Budget/Impact/Progress)',
    'BPI_v1_Classification.csv': 'BPI_v1_Classification.csv (Qualitative classes: Low/Medium/High)'
}
def _norm_name(s:str)->str:
    s = s.split("/")[-1].split("\\")[-1]
    s = s.lower()
    s = re.sub(r'[\s_\-]+', '', s)
    return s

def load_from_zip(zip_uploaded_file):
    out = {}
    buf = zip_uploaded_file.getvalue()
    with zipfile.ZipFile(io.BytesIO(buf)) as zf:
        members = { _norm_name(n): n for n in zf.namelist() if n.lower().endswith('.csv') }
        for req in REQUIRED_FILES.keys():
            wanted = _norm_name(req)
            wanted2 = wanted.replace('.', '')
            match = None
            if wanted in members:
                match = members[wanted]
            else:
                chunks = re.findall(r'[a-z0-9]+', wanted2)
                for nn, full in members.items():
                    if all(ch in nn for ch in chunks):
                        match = full
                        break
            if match:
                out[req] = zf.read(match)
    return out

# ==============================
# 3) SIDEBAR UPLOADS
# ==============================
st.sidebar.title("📁 Data Upload & Setup")

uploaded_files = {}
all_ok = True

with st.sidebar.expander("✅ Preferred: Upload a single ZIP (e.g., **NDP csv data.zip**)", expanded=True):
    zip_file = st.file_uploader("Upload ZIP with all CSVs", type=["zip"], key="zip_all")
    if zip_file is not None:
        zipped_map = load_from_zip(zip_file)
        for k in REQUIRED_FILES.keys():
            if k in zipped_map:
                uploaded_files[k] = zipped_map[k]
        missing = [k for k in REQUIRED_FILES.keys() if k not in uploaded_files]
        if missing:
            st.warning("These files were not found in the ZIP: " + ", ".join(missing))

st.sidebar.markdown("---")
st.sidebar.markdown("Or upload CSVs individually (fallback)")

for key, label in REQUIRED_FILES.items():
    if key not in uploaded_files:
        f = st.sidebar.file_uploader(f"Upload: {label}", type=["csv"], key=key)
        if f is None:
            all_ok = False
        else:
            uploaded_files[key] = f

# ==============================
# 4) DATA PROCESS
# ==============================
@st.cache_data(show_spinner="Processing and linking all datasets...")
def process_data(files_map):
    # Load & clean
    df_long     = clean_columns(read_csv_any(files_map['ONDP Dataset.csv']))
    df_pimap    = clean_columns(read_csv_any(files_map['PI Mapping V1.csv']))
    df_proxies  = clean_columns(read_csv_any(files_map['Proxies list ONDP.csv']))
    df_macro    = clean_columns(read_csv_any(files_map['Macro Economic.csv']))
    df_bpi      = clean_columns(read_csv_any(files_map['BPI V1.csv']))
    df_bpi_qual = clean_columns(read_csv_any(files_map['BPI_v1_Classification.csv']))

    # Normalize OmanCodeKey where available
    for df in [df_long, df_pimap, df_bpi, df_bpi_qual]:
        if 'Oman Code' in df.columns:
            df['OmanCodeKey'] = df['Oman Code'].apply(norm_code)
        elif 'Oman Plan Code' in df.columns:
            df['OmanCodeKey'] = df['Oman Plan Code'].apply(norm_code)
        else:
            df['OmanCodeKey'] = ""

    # ---- Page 1: Elements table ----
    item_col = None
    if 'Item Name' in df_long.columns: item_col = 'Item Name'
    elif 'Element Name' in df_long.columns: item_col = 'Element Name'
    else:
        for c in df_long.columns:
            if "item" in c.lower() or "element" in c.lower():
                item_col = c; break

    comp_col = 'Element Completion %'
    if comp_col not in df_long.columns:
        for alt_name in ['Element Completion%', 'Element Completion Rate', 'Completion %']:
            if alt_name in df_long.columns:
                comp_col = alt_name; break
    if comp_col in df_long.columns:
        # Parse percent-like strings into 0..1
        df_long[comp_col] = parse_percentish(df_long[comp_col])

    # ---- PI Time Series ----
    year_cols_pi = [c for c in df_proxies.columns if c.isdigit() and len(c) == 4]
    if {'Indicator ID','Indicator'}.issubset(df_proxies.columns) and year_cols_pi:
        df_proxies_long = df_proxies[['Indicator ID','Indicator'] + year_cols_pi].melt(
            id_vars=['Indicator ID','Indicator'], value_vars=year_cols_pi,
            var_name='Year', value_name='Value'
        )
        df_proxies_long['Year']  = to_numeric(df_proxies_long['Year']).astype('Int64')
        df_proxies_long['Value'] = to_numeric(df_proxies_long['Value'])
        if 'Indicator Name' in df_pimap.columns:
            name_map = df_pimap[['Indicator ID','Indicator Name']].drop_duplicates('Indicator ID')
            df_proxies_long = df_proxies_long.merge(name_map, on='Indicator ID', how='left')
        else:
            df_proxies_long['Indicator Name'] = ""
    else:
        df_proxies_long = pd.DataFrame(columns=['Indicator ID','Indicator','Year','Value','Indicator Name'])

    # ---- Macro Economic (robust) ----
    if 'Indicator' in df_macro.columns:
        # Clean indicator text (strip, remove invisible LRM/RLM, etc.)
        df_macro['Indicator'] = (df_macro['Indicator'].astype(str)
                                                   .str.replace('\u200f', '', regex=False)  # RTL mark
                                                   .str.replace('\u200e', '', regex=False)  # LTR mark
                                                   .str.strip())

        year_cols_macro = detect_year_cols(df_macro.columns)

        if year_cols_macro:
            df_macro_long = df_macro.melt(
                id_vars=['Indicator'],
                value_vars=year_cols_macro,
                var_name='Year',
                value_name='Value'
            ).dropna(subset=['Indicator'])

            # Normalize Year: keep the 4 digits even if the column header had extra chars
            df_macro_long['Year'] = (df_macro_long['Year'].astype(str)
                                                     .str.replace(r'\D', '', regex=True))
            # Convert to Int64 (nullable int)
            df_macro_long['Year'] = pd.to_numeric(df_macro_long['Year'], errors='coerce').astype('Int64')

            # Robust numeric parsing (commas, Arabic separators, stray text)
            df_macro_long['Value'] = parse_numberish(df_macro_long['Value'])

            # Drop rows where year or value are missing after cleaning
            df_macro_long = df_macro_long.dropna(subset=['Year', 'Value'])

        else:
            df_macro_long = pd.DataFrame(columns=['Indicator','Year','Value'])
    else:
        df_macro_long = pd.DataFrame(columns=['Indicator','Year','Value'])

    # ---- BPI quantitative (BPI V1.csv) ----
    if 'Program name' in df_bpi.columns and 'Program Name' not in df_bpi.columns:
        df_bpi.rename(columns={'Program name':'Program Name'}, inplace=True)
    for c in ['Budget','Impact','Progress']:
        if c in df_bpi.columns:
            df_bpi[c] = to_unit_interval(df_bpi[c])
            df_bpi[f'{c}_Class'] = df_bpi[c].apply(classify)
    rank_map = {'Low':1, 'Medium':2, 'High':3, 'N/A':0, 'Other':0}
    for c in ['Budget','Impact','Progress']:
        cls = f'{c}_Class'; rnk = f'{c}_Rank'
        if cls in df_bpi.columns:
            df_bpi[rnk] = df_bpi[cls].map(rank_map).fillna(0)
    if 'Program Name' not in df_bpi.columns:
        for alt in ['Program Title','Title','Program','Name']:
            if alt in df_bpi.columns:
                df_bpi.rename(columns={alt:'Program Name'}, inplace=True); break
    if 'Program Name' not in df_bpi.columns:
        df_bpi['Program Name'] = ""
    df_bpi['Program_ID'] = df_bpi['OmanCodeKey'].astype(str) + ' - ' + df_bpi['Program Name'].astype(str).str.strip()

    # ---- BPI qualitative (Classification CSV ONLY) ----
    if 'Program name' in df_bpi_qual.columns and 'Program Name' not in df_bpi_qual.columns:
        df_bpi_qual.rename(columns={'Program name':'Program Name'}, inplace=True)
    if 'Program Name' not in df_bpi_qual.columns:
        for alt in ['Program Title','Title','Program','Name']:
            if alt in df_bpi_qual.columns:
                df_bpi_qual.rename(columns={alt:'Program Name'}, inplace=True); break
    if 'Program Name' not in df_bpi_qual.columns:
        df_bpi_qual['Program Name'] = ""

    for col in ['Budget_Class', 'Impact_Class', 'Progress_Class']:
        if col not in df_bpi_qual.columns:
            for cand in [col.replace('_',' '), col.replace('_',''), col.split('_')[0]]:
                if cand in df_bpi_qual.columns:
                    df_bpi_qual.rename(columns={cand: col}, inplace=True); break
        if col in df_bpi_qual.columns:
            df_bpi_qual[col] = normalize_class_series(df_bpi_qual[col])

    keep = ['OmanCodeKey','Program Name','Budget_Class','Impact_Class','Progress_Class']
    df_bpi_qual = df_bpi_qual[[c for c in keep if c in df_bpi_qual.columns]].copy()

    for base in ['Budget','Impact','Progress']:
        cls = f'{base}_Class'; rnk = f'{base}_Rank'
        if cls in df_bpi_qual.columns:
            df_bpi_qual[rnk] = df_bpi_qual[cls].map(rank_map).fillna(0)

    df_bpi_qual['Program_ID'] = df_bpi_qual['OmanCodeKey'].astype(str) + ' - ' + df_bpi_qual['Program Name'].astype(str).str.strip()

    return {
        'long': df_long,
        'long_item_col': item_col,
        'long_comp_col': comp_col,
        'pi_map': df_pimap,
        'proxies_pi_long': df_proxies_long,
        'macro_long': df_macro_long,
        'bpi': df_bpi,
        'bpi_qual': df_bpi_qual
    }

# ==============================
# 5) PAGES
# ==============================
def page_program_details(data):
    st.title("Program Deep Dive: Elements, Mapping, and PI Trends")
    st.markdown("---")

    df_long = data['long']
    item_col = data['long_item_col']
    comp_col = data['long_comp_col']

    if 'OmanCodeKey' not in df_long.columns:
        st.error("`Oman Code` not found in ONDP Dataset.csv")
        return

    codes = sorted([c for c in df_long['OmanCodeKey'].unique() if c])
    if not codes:
        st.warning("No Oman Codes found.")
        return

    selected_code = st.sidebar.selectbox("Select Oman Code", codes, index=0)
    subset = df_long[df_long['OmanCodeKey'] == selected_code].copy()
    if subset.empty:
        st.warning(f"No detailed data found for Oman Code: {selected_code}.")
        return

    row0 = subset.iloc[0]
    st.subheader(f"Program: {str(row0.get('Program Title','')).strip()}")

    # Wider space for Responsible Party; move years to a new line
    c1,c2,c3 = st.columns([1,3,1])
    c1.metric("Tag", row0.get('Tag','N/A'))
    c2.metric("Responsible Party", row0.get('Responsible Party','N/A'))
    c3.metric("Status", row0.get('Status','N/A'))

    c4,c5 = st.columns(2)
    c4.metric("Start Year", safe_int_text(row0.get('Start Year')))
    c5.metric("End Year", safe_int_text(row0.get('End Year')))

    st.markdown("---")
    st.subheader("Program Elements — Data Type & Element Completion %")

    dtype_col = 'Data Type' if 'Data Type' in subset.columns else None
    if item_col and comp_col and {item_col, comp_col}.issubset(subset.columns):
        cols = [item_col, comp_col]
        if dtype_col: cols.insert(1, dtype_col)
        elems = subset[cols].copy()
        rename_map = {item_col: 'Element', comp_col: 'Element Completion %'}
        if dtype_col: rename_map[dtype_col] = 'Data Type'
        elems.rename(columns=rename_map, inplace=True)
        elems['Element'] = elems['Element'].astype(str).str.strip()

        # Ensure completion is formatted as % string but keep blanks for NaN
        elems['Element Completion %'] = elems['Element Completion %'].apply(
            lambda x: "" if pd.isna(x) else f"{x*100:.1f}%"
        )

        col_config = {
            "Element": st.column_config.TextColumn("Element"),
            "Element Completion %": st.column_config.TextColumn("Element Completion %"),
        }
        order = ["Element"]
        if 'Data Type' in elems.columns:
            col_config["Data Type"] = st.column_config.TextColumn("Data Type")
            order.append("Data Type")
        order.append("Element Completion %")

        st.dataframe(elems, use_container_width=True, hide_index=True,
                     column_config=col_config, column_order=order)
    else:
        st.warning("Could not find the element/completion columns in ONDP Dataset.csv.")
        st.info(f"Detected element column: {item_col}, completion column: {comp_col}")

    st.markdown("---")
    st.subheader("Performance Indicator (PI) Mapping")

    df_map = data['pi_map']
    mask = (df_map['OmanCodeKey'] == selected_code) if 'OmanCodeKey' in df_map.columns else pd.Series(False)
    df_pi_filtered = df_map[mask]
    if not df_pi_filtered.empty:
        pi_ids = df_pi_filtered['Indicator ID'].dropna().unique().tolist()
        if pi_ids:
            sel_pi = st.selectbox("Select Indicator ID for Details", pi_ids, key="page1_pi_select")
            pi_row = df_pi_filtered[df_pi_filtered['Indicator ID'] == sel_pi].iloc[0]

            # Full-width indicator name
            st.write("**Indicator Name (EN):**")
            st.write(str(pi_row.get('Indicator Name','')))

            cL, cR = st.columns([1,2])
            with cL:
                conf = str(pi_row.get('Confidence Score (0-100)','')).split('.')[0]
                st.metric("Confidence Score", f"{conf}%" if conf else "")
            with cR:
                st.markdown("**Mapping Justification**")
                st.info(pi_row.get('Mapping Justification',''))

            st.markdown("---")
            st.subheader(f"Time-Series Trend for Indicator: {pi_row.get('Indicator Name','')}")
            ts = data['proxies_pi_long'][data['proxies_pi_long']['Indicator ID'] == sel_pi].copy()
            if not ts.empty:
                line = alt.Chart(ts).mark_line(point=True, interpolate='monotone', strokeWidth=3, color='#1E90FF').encode(
                    x=alt.X('Year:Q', axis=alt.Axis(title='Year', format='d')),
                    y=alt.Y('Value:Q', title='Indicator Value'),
                    tooltip=['Year', alt.Tooltip('Value:Q', format='.2f'), 'Indicator Name']
                )
                st.altair_chart(line, use_container_width=True)
            else:
                st.info("No time-series values available for this indicator.")
        else:
            st.info("No Indicator IDs mapped to this program.")
    else:
        st.warning("No PI mapping found for the selected Oman Code.")

def page_bpi_only(data):
    st.title("BPI Matrix — Single View")
    st.caption("Quantitative uses BPI V1.csv; Qualitative uses BPI_v1_Classification.csv.")
    st.markdown("---")

    dq  = data['bpi']
    dql = data['bpi_qual']
    if dq.empty:
        st.error("BPI V1.csv is empty or missing data."); return
    if dql.empty:
        st.error("BPI_v1_Classification.csv is required for qualitative view."); return

    # Controls
    available_codes = sorted([c for c in dq['OmanCodeKey'].unique() if c])
    pick_codes = st.multiselect("Select Oman Code(s)", available_codes, default=available_codes[:1])

    matrix_type = st.radio(
        "Matrix",
        ["Budget vs Impact (Bubble = Progress)", "Progress vs Impact (Bubble = Budget)"],
        horizontal=True
    )
    method = st.radio("Style", ["Quantitative", "Qualitative"], horizontal=True)

    df_q  = dq[dq['OmanCodeKey'].isin(pick_codes)].copy()
    df_ql = dql[dql['OmanCodeKey'].isin(pick_codes)].copy()

    if method == "Quantitative" and df_q.empty:
        st.info("No quantitative rows for selected code(s)."); return
    if method == "Qualitative" and df_ql.empty:
        st.info("No qualitative rows for selected code(s)."); return

    names = (df_ql if method=="Qualitative" else df_q)['Program Name'].dropna().unique().tolist()
    if names:
        st.subheader("Programs: " + ", ".join(names))

    bubble_fill  = "#FFD84D"
    bubble_edge  = "#333333"

    if "Budget vs Impact" in matrix_type:
        x_field, y_field = "Budget", "Impact"
        size_class_for_qual = "Progress_Class"
        title = "Budget vs Impact (Bubble = Progress)"
    else:
        x_field, y_field = "Progress", "Impact"
        size_class_for_qual = "Budget_Class"
        title = "Progress vs Impact (Bubble = Budget)"

    # Quantitative
    def quantitative_scatter(df, x_field, y_field, size_field, title):
        jj = df.copy()
        jj[x_field] = to_numeric(jj[x_field]); jj[y_field] = to_numeric(jj[y_field])
        jj['x_jit'] = jj[x_field] + np.random.normal(0, 0.003, size=len(jj))
        jj['y_jit'] = jj[y_field] + np.random.normal(0, 0.003, size=len(jj))
        return alt.Chart(jj, width=1100, height=480).mark_circle(
            opacity=0.9, fill=bubble_fill, stroke=bubble_edge, strokeWidth=1.5
        ).encode(
            x=alt.X('x_jit:Q', scale=alt.Scale(domain=[0,1]), axis=alt.Axis(format='%', title=x_field)),
            y=alt.Y('y_jit:Q', scale=alt.Scale(domain=[0,1]), axis=alt.Axis(format='%', title=y_field)),
            size=alt.Size(f'{size_field}:Q', scale=alt.Scale(domain=[0,1], range=[80,1600]),
                          legend=alt.Legend(title=size_field)),
            tooltip=['OmanCodeKey:N','Program Name',
                     alt.Tooltip(f'{x_field}:Q', format='.1%'),
                     alt.Tooltip(f'{y_field}:Q', format='.1%'),
                     alt.Tooltip(f'{size_field}:Q', title=size_field, format='.1%')]
        ).properties(title=title).configure_view(stroke=None)

    # Qualitative
    def qualitative_matrix(df_ql, x_field, y_field, title):
        cats_x = ["Low", "Medium", "High"]
        cats_y = ["High", "Medium", "Low"]  # top to bottom

        # normalize (safety)
        for c in [f'{x_field}_Class', f'{y_field}_Class', 'Budget_Class', 'Progress_Class', 'Impact_Class']:
            if c in df_ql.columns:
                df_ql[c] = normalize_class_series(df_ql[c])

        need = [f'{x_field}_Class', f'{y_field}_Class', size_class_for_qual]
        missing_rows = df_ql[need].isna().any(axis=1).sum()
        df_plot = df_ql.dropna(subset=need).copy()
        if missing_rows:
            st.caption(f"ℹ️ Skipped {missing_rows} row(s) with missing class values.")

        bg_df = pd.DataFrame({
            "x": ["Low","Medium","High"] * 3,
            "y": sum([[y]*3 for y in cats_y], []),
            "color": [
                "#DFF3D7","#BEE7B6","#81D67F",
                "#FBE3D7","#F1C5A8","#E7A884",
                "#FAD1CB","#F5A79C","#F07267",
            ]
        })

        bg = alt.Chart(bg_df, width=1100, height=480).mark_rect(stroke="#FFFFFF", strokeWidth=1).encode(
            x=alt.X("x:N",
                    scale=alt.Scale(domain=cats_x, paddingInner=0, paddingOuter=0),
                    axis=alt.Axis(title=x_field, labelAngle=0)),
            y=alt.Y("y:N",
                    scale=alt.Scale(domain=cats_y, paddingInner=0, paddingOuter=0),
                    axis=alt.Axis(title=y_field, labelAngle=0)),
            color=alt.Color("color:N", scale=None, legend=None),
        )

        size_rank_map = {"Low": 1, "Medium": 2, "High": 3}
        size_range    = [350, 950, 1800]

        df_plot['x_cat']   = pd.Categorical(df_plot[f'{x_field}_Class'], categories=cats_x, ordered=True)
        df_plot['y_cat']   = pd.Categorical(df_plot[f'{y_field}_Class'], categories=cats_y, ordered=True)
        df_plot['SizeKey'] = df_plot[size_class_for_qual].map(size_rank_map).fillna(2)

        points = alt.Chart(df_plot, width=1100, height=480).mark_circle(
            opacity=0.95, fill="#FFD84D", stroke="#333333", strokeWidth=1.8
        ).encode(
            x=alt.X('x_cat:N', scale=alt.Scale(domain=cats_x, paddingInner=0, paddingOuter=0), axis=alt.Axis(title=None)),
            y=alt.Y('y_cat:N', scale=alt.Scale(domain=cats_y, paddingInner=0, paddingOuter=0), axis=alt.Axis(title=None)),
            size=alt.Size('SizeKey:Q',
                          scale=alt.Scale(domain=[1,3], range=size_range),
                          legend=alt.Legend(title=f"{size_class_for_qual.replace('_Class','')} Class (size)",
                                            values=[1,2,3],
                                            labelExpr="[ 'Low','Medium','High' ][datum.value-1]")),
            tooltip=[
                'OmanCodeKey:N', 'Program Name:N',
                alt.Tooltip(f'{x_field}_Class:N', title=x_field),
                alt.Tooltip(f'{y_field}_Class:N', title=y_field),
                alt.Tooltip(f'{size_class_for_qual}:N', title="Size Class"),
            ],
            color=alt.value('#00000000')
        )

        return (bg + points).properties(title=title).configure_axis(
            labelFontSize=13, titleFontSize=14
        ).configure_view(stroke=None)

    st.markdown("### Matrix")
    if method == "Quantitative":
        if "Budget vs Impact" in matrix_type:
            chart = quantitative_scatter(df_q, "Budget", "Impact", "Progress", "Budget vs Impact (Bubble = Progress)")
        else:
            chart = quantitative_scatter(df_q, "Progress", "Impact", "Budget", "Progress vs Impact (Bubble = Budget)")
    else:
        chart = qualitative_matrix(df_ql, x_field=("Budget" if "Budget vs" in matrix_type else "Progress"),
                                   y_field="Impact",
                                   title=("Budget vs Impact (Bubble = Progress)" if "Budget vs" in matrix_type
                                          else "Progress vs Impact (Bubble = Budget)"))

    st.altair_chart(chart, use_container_width=False)

def page_macro(data):
    st.title("Macro Economic Indicators Trend")
    st.markdown("---")
    macro = data['macro_long']
    if macro.empty:
        st.info("No macro data available."); return
    options = sorted(macro['Indicator'].dropna().unique().tolist())
    if not options:
        st.info("No indicators found."); return
    sel = st.selectbox("Select Macro Economic Indicator (Arabic Name)", options)
    if sel:
        d = macro[(macro['Indicator'] == sel) & (macro['Value'].notna())].copy()
        if d.empty:
            st.info("No values for the selected indicator after cleaning.")
            return
        line = alt.Chart(d).mark_line(point=True, strokeWidth=3, color='#FF4B4B').encode(
            x=alt.X('Year:Q', axis=alt.Axis(title='Year', format='d')),   # numeric year axis
            y=alt.Y('Value:Q', title='Value'),
            tooltip=['Year', alt.Tooltip('Value:Q', format=',.1f'), 'Indicator']
        )
        st.altair_chart(line, use_container_width=True)

# ==============================
# 6) MAIN
# ==============================
if not uploaded_files or any(k not in uploaded_files for k in REQUIRED_FILES):
    st.title("Welcome to the ONDP Performance Dashboard")
    st.warning("**Please upload all CSVs (preferably in one ZIP named like `NDP csv data.zip`).**")
else:
    try:
        data_bundle = process_data(uploaded_files)
        st.sidebar.header("Dashboard Navigation")
        pages = {
            "Page 1: Program Details & PI Trends": page_program_details,
            "Page 2: Macro Economic Trends": page_macro,
            "Page 3: BPI (Single Matrix, Codes Multi-Select)": page_bpi_only,
        }
        page = st.sidebar.selectbox("Go to Page", list(pages.keys()))
        pages[page](data_bundle)
    except Exception as e:
        st.error("A critical error occurred during data processing.")
        st.markdown("**Error Details:**")
        st.code(str(e))
        st.warning("Ensure the ZIP contains all required CSVs and column names are consistent.")
