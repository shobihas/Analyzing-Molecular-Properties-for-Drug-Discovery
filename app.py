# app.py
import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, Draw
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io
import plotly.express as px
import plotly.graph_objects as go

# -------- CONFIG --------
DATA_PATH = "data/molecules.csv"  # update if your file name differs
st.set_page_config(page_title="SMILES Molecules Dashboard", layout="wide")

# -------- UTILITIES --------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

def mol_from_smiles(smiles):
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None

def compute_descriptors(df, smiles_col="SMILES"):
    # Ensure columns exist
    cols_to_add = [
        "MolWt", "LogP", "NumHDonors", "NumHAcceptors",
        "TPSA", "NumRotatableBonds", "NumAromaticRings",
        "LipinskiViolations", "ValidSMILES"
    ]
    for c in cols_to_add:
        if c not in df.columns:
            df[c] = np.nan

    mols = []
    for i, smi in enumerate(df[smiles_col].astype(str)):
        mol = mol_from_smiles(smi)
        mols.append(mol)
        if mol is None:
            df.at[i, "ValidSMILES"] = False
            continue
        df.at[i, "ValidSMILES"] = True
        df.at[i, "MolWt"] = Descriptors.MolWt(mol)
        df.at[i, "LogP"] = Crippen.MolLogP(mol)
        df.at[i, "NumHDonors"] = Lipinski.NumHDonors(mol)
        df.at[i, "NumHAcceptors"] = Lipinski.NumHAcceptors(mol)
        df.at[i, "TPSA"] = rdMolDescriptors.CalcTPSA(mol)
        df.at[i, "NumRotatableBonds"] = Lipinski.NumRotatableBonds(mol)
        df.at[i, "NumAromaticRings"] = rdMolDescriptors.CalcNumAromaticRings(mol)

        # Lipinski rule of five check (classical)
        violations = 0
        if df.at[i, "MolWt"] > 500: violations += 1
        if df.at[i, "LogP"] > 5: violations += 1
        if df.at[i, "NumHDonors"] > 5: violations += 1
        if df.at[i, "NumHAcceptors"] > 10: violations += 1
        df.at[i, "LipinskiViolations"] = violations

    df["LipinskiDruglike"] = df["LipinskiViolations"] == 0
    df["mol_obj"] = mols
    return df

def draw_molecule_image(mol, size=(300, 200)):
    if mol is None:
        return None
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()
    return Image.open(io.BytesIO(png))

# -------- APPLICATION --------
st.title("SMILES Molecules — Interactive Dashboard")
st.markdown("Compute RDKit molecular descriptors, Lipinski filter, and visualize descriptors (box/line/heatmap).")

# Load data
with st.spinner("Loading dataset..."):
    df_raw = load_data(DATA_PATH)

# Try to auto-detect SMILES column name
possible_smiles_cols = [c for c in df_raw.columns if c.lower() in ("smiles", "smiles_string", "smile", "canonical_smiles", "smiles_")]
if len(possible_smiles_cols) > 0:
    SMILES_COL = possible_smiles_cols[0]
else:
    # fallback to first column
    SMILES_COL = df_raw.columns[0]

st.sidebar.header("Data & Filters")
st.sidebar.write(f"Detected SMILES column: **{SMILES_COL}**")
st.sidebar.markdown("If this is wrong, update `SMILES_COL` in app.py.")

# Compute descriptors (cached)
if "processed" not in st.session_state:
    with st.spinner("Computing molecular descriptors (RDKit)..."):
        df = compute_descriptors(df_raw.copy(), smiles_col=SMILES_COL)
        st.session_state["processed"] = True
        st.session_state["df"] = df
else:
    df = st.session_state["df"]

# Clean: only valid SMILES for plotting
df_valid = df[df["ValidSMILES"] == True].copy()

# Sidebar filters
min_mw, max_mw = float(df_valid["MolWt"].min()), float(df_valid["MolWt"].max())
mw_range = st.sidebar.slider("Molecular Weight (MolWt)", min_value=float(min_mw), max_value=float(max_mw),
                             value=(min_mw, max_mw))
logp_min, logp_max = float(df_valid["LogP"].min()), float(df_valid["LogP"].max())
logp_range = st.sidebar.slider("LogP", min_value=float(logp_min), max_value=float(logp_max),
                               value=(logp_min, logp_max))

lipinski_filter = st.sidebar.selectbox("Lipinski drug-like filter", options=["All", "Drug-like (0 violations)", "Non drug-like (>0 violations)"])

# apply filters
df_filtered = df_valid[(df_valid["MolWt"] >= mw_range[0]) & (df_valid["MolWt"] <= mw_range[1]) &
                       (df_valid["LogP"] >= logp_range[0]) & (df_valid["LogP"] <= logp_range[1])]
if lipinski_filter == "Drug-like (0 violations)":
    df_filtered = df_filtered[df_filtered["LipinskiViolations"] == 0]
elif lipinski_filter == "Non drug-like (>0 violations)":
    df_filtered = df_filtered[df_filtered["LipinskiViolations"] > 0]

# --- Dashboard top metrics (Cards) ---
st.subheader("Summary metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total molecules (valid)", len(df_valid))
col2.metric("Drug-like (Lipinski)", int(df_valid["LipinskiDruglike"].sum()))
col3.metric("Avg MolWt", f"{df_valid['MolWt'].mean():.2f}")
col4.metric("Avg LogP", f"{df_valid['LogP'].mean():.2f}")

# --- Main layout ---
left_col, right_col = st.columns((3,2))

with left_col:
    st.markdown("### Descriptor Distributions (Box Plots)")
    # Box plot for key descriptors
    descriptors = ["MolWt", "LogP", "NumHDonors", "NumHAcceptors", "TPSA", "NumRotatableBonds"]
    melted = df_filtered[descriptors].melt(var_name="Descriptor", value_name="Value")
    fig_box = px.box(melted, x="Descriptor", y="Value", points="outliers", title="Descriptor Boxplots")
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("### Descriptor Correlation Heatmap")
    corr = df_filtered[descriptors].corr()
    fig_heat = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation heatmap (descriptors)")
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("### Trend / Line Plot")
    # create bins of MolWt and compute mean LogP per bin
    df_filtered["MolWt_bin"] = pd.cut(df_filtered["MolWt"], bins=20)
    trend = df_filtered.groupby("MolWt_bin").agg(mean_LogP=("LogP","mean"), count=("LogP","count")).reset_index().dropna()
    trend["bin_center"] = trend["MolWt_bin"].apply(lambda r: r.mid)
    fig_line = px.line(trend.sort_values("bin_center"), x="bin_center", y="mean_LogP",
                       labels={"bin_center":"MolWt (bin center)", "mean_LogP":"Mean LogP"},
                       title="Mean LogP vs MolWt (binned)")
    fig_line.add_trace(go.Bar(x=trend["bin_center"], y=trend["count"], name="count", yaxis="y2", opacity=0.15))
    # Add secondary axis
    fig_line.update_layout(
        yaxis=dict(title="Mean LogP"),
        yaxis2=dict(title="Count", overlaying="y", side="right", showgrid=False)
    )
    st.plotly_chart(fig_line, use_container_width=True)

with right_col:
    st.markdown("### Filters summary")
    st.write(f"MolWt in {mw_range}")
    st.write(f"LogP in {logp_range}")
    st.write("Lipinski filter:", lipinski_filter)
    st.markdown("---")
    st.markdown("### Molecule table (sample)")
    # show table with key descriptors
    display_cols = [SMILES_COL, "MolWt", "LogP", "NumHDonors", "NumHAcceptors", "TPSA", "LipinskiViolations"]
    st.dataframe(df_filtered[display_cols].sample(min(200, len(df_filtered))).reset_index(drop=True))

    st.markdown("### Molecule viewer")
    idx = st.number_input("Row index (in filtered dataset) to view molecule:", min_value=0, max_value=max(0, len(df_filtered)-1), value=0, step=1)
    if len(df_filtered) > 0:
        mol = df_filtered.reset_index(drop=True).loc[idx, "mol_obj"]
        st.write("SMILES:", df_filtered.reset_index(drop=True).loc[idx, SMILES_COL])
        img = draw_molecule_image(mol, size=(400,300))
        if img is not None:
            st.image(img, use_column_width=True)
        else:
            st.write("Could not draw molecule.")

# --- Footer: download processed data ---
st.markdown("---")
st.markdown("### Download processed dataset")
@st.cache_data
def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

csv_bytes = df_to_csv_bytes(df)
st.download_button("Download full processed CSV", data=csv_bytes, file_name="molecules_processed.csv", mime="text/csv")
