import os
import io

import streamlit as st
import pandas as pd
import numpy as np

from PIL import Image

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SMILES Molecules Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

# IMPORTANT:
# Streamlit Community Cloud runs the app from the repository root.
#
# Repository:
# Analyzing-Molecular-Properties-for-Drug-Discovery/
#
# App:
# DRUG_DASHBOARD/app.py
#
# Dataset:
# DRUG_DASHBOARD/data/molecules.csv

DATA_PATH = os.path.join(
    "DRUG_DASHBOARD",
    "data",
    "molucules.csv"
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data(path):
    """Load molecular dataset from CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at: {path}"
        )

    return pd.read_csv(path)


# ============================================================
# MOLECULAR UTILITIES
# ============================================================

def mol_from_smiles(smiles):
    """Convert SMILES string to RDKit molecule."""
    try:
        if pd.isna(smiles):
            return None

        smiles = str(smiles).strip()

        if not smiles:
            return None

        return Chem.MolFromSmiles(smiles)

    except Exception:
        return None


def compute_descriptors(df, smiles_col):
    """
    Calculate RDKit molecular descriptors for every molecule.
    """

    df = df.copy()

    descriptor_columns = [
        "MolWt",
        "LogP",
        "NumHDonors",
        "NumHAcceptors",
        "TPSA",
        "NumRotatableBonds",
        "NumAromaticRings",
        "LipinskiViolations",
        "ValidSMILES",
    ]

    for column in descriptor_columns:
        if column not in df.columns:
            df[column] = np.nan

    mols = []

    for idx, smiles in df[smiles_col].items():

        mol = mol_from_smiles(smiles)

        mols.append(mol)

        if mol is None:
            df.at[idx, "ValidSMILES"] = False
            continue

        df.at[idx, "ValidSMILES"] = True

        # Molecular weight
        df.at[idx, "MolWt"] = Descriptors.MolWt(mol)

        # LogP
        df.at[idx, "LogP"] = Crippen.MolLogP(mol)

        # Hydrogen bond donors
        df.at[idx, "NumHDonors"] = Lipinski.NumHDonors(mol)

        # Hydrogen bond acceptors
        df.at[idx, "NumHAcceptors"] = Lipinski.NumHAcceptors(mol)

        # Topological polar surface area
        df.at[idx, "TPSA"] = rdMolDescriptors.CalcTPSA(mol)

        # Rotatable bonds
        df.at[idx, "NumRotatableBonds"] = (
            Lipinski.NumRotatableBonds(mol)
        )

        # Aromatic rings
        df.at[idx, "NumAromaticRings"] = (
            rdMolDescriptors.CalcNumAromaticRings(mol)
        )

        # Lipinski Rule of Five
        violations = 0

        if df.at[idx, "MolWt"] > 500:
            violations += 1

        if df.at[idx, "LogP"] > 5:
            violations += 1

        if df.at[idx, "NumHDonors"] > 5:
            violations += 1

        if df.at[idx, "NumHAcceptors"] > 10:
            violations += 1

        df.at[idx, "LipinskiViolations"] = violations

    df["LipinskiDruglike"] = (
        df["LipinskiViolations"] == 0
    )

    df["mol_obj"] = mols

    return df


# ============================================================
# MOLECULE DRAWING
# ============================================================

def draw_molecule_image(mol, size=(500, 350)):
    """Draw RDKit molecule and return PIL image."""

    if mol is None:
        return None

    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(
            size[0],
            size[1]
        )

        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer,
            mol
        )

        drawer.FinishDrawing()

        png = drawer.GetDrawingText()

        return Image.open(
            io.BytesIO(png)
        )

    except Exception:
        return None


# ============================================================
# HEADER
# ============================================================

st.title("🧬 SMILES Molecules — Interactive Dashboard")

st.markdown(
    """
    Analyze molecular properties using **RDKit** and interactively
    explore molecular descriptors, Lipinski drug-likeness,
    correlations, trends, and molecular structures.
    """
)


# ============================================================
# LOAD DATA
# ============================================================

with st.spinner("Loading molecular dataset..."):

    try:
        df_raw = load_data(DATA_PATH)

    except FileNotFoundError:

        st.error(
            f"""
            ❌ Dataset not found.

            Expected file:

            `{DATA_PATH}`

            Please make sure your GitHub repository contains:

            `DRUG_DASHBOARD/data/molecules.csv`
            """
        )

        st.stop()

    except Exception as e:

        st.error(
            f"Unable to load the dataset: {e}"
        )

        st.stop()


# ============================================================
# VALIDATE DATASET
# ============================================================

if df_raw.empty:

    st.error(
        "The CSV file is empty."
    )

    st.stop()


# ============================================================
# AUTO-DETECT SMILES COLUMN
# ============================================================

possible_smiles_cols = [
    column
    for column in df_raw.columns
    if column.lower().strip()
    in (
        "smiles",
        "smiles_string",
        "smile",
        "canonical_smiles",
        "smiles_",
    )
]


if possible_smiles_cols:

    SMILES_COL = possible_smiles_cols[0]

else:

    SMILES_COL = df_raw.columns[0]


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Data & Filters")

st.sidebar.write(
    f"Detected SMILES column: **{SMILES_COL}**"
)

st.sidebar.caption(
    "The SMILES column is automatically detected."
)


# ============================================================
# COMPUTE DESCRIPTORS
# ============================================================

with st.spinner(
    "Computing molecular descriptors using RDKit..."
):

    df = compute_descriptors(
        df_raw,
        smiles_col=SMILES_COL
    )


# ============================================================
# VALID MOLECULES
# ============================================================

df_valid = df[
    df["ValidSMILES"] == True
].copy()


if df_valid.empty:

    st.error(
        "No valid SMILES were found in the dataset."
    )

    st.stop()


# ============================================================
# SIDEBAR FILTERS
# ============================================================

min_mw = float(
    df_valid["MolWt"].min()
)

max_mw = float(
    df_valid["MolWt"].max()
)

if min_mw == max_mw:

    mw_range = (
        min_mw,
        max_mw
    )

    st.sidebar.info(
        f"Molecular Weight: {min_mw:.2f}"
    )

else:

    mw_range = st.sidebar.slider(
        "Molecular Weight (MolWt)",
        min_value=min_mw,
        max_value=max_mw,
        value=(min_mw, max_mw),
    )


# ------------------------------------------------------------

logp_min = float(
    df_valid["LogP"].min()
)

logp_max = float(
    df_valid["LogP"].max()
)

if logp_min == logp_max:

    logp_range = (
        logp_min,
        logp_max
    )

    st.sidebar.info(
        f"LogP: {logp_min:.2f}"
    )

else:

    logp_range = st.sidebar.slider(
        "LogP",
        min_value=logp_min,
        max_value=logp_max,
        value=(logp_min, logp_max),
    )


# ------------------------------------------------------------

lipinski_filter = st.sidebar.selectbox(
    "Lipinski drug-like filter",
    options=[
        "All",
        "Drug-like (0 violations)",
        "Non drug-like (>0 violations)",
    ],
)


# ============================================================
# APPLY FILTERS
# ============================================================

df_filtered = df_valid[
    (df_valid["MolWt"] >= mw_range[0])
    &
    (df_valid["MolWt"] <= mw_range[1])
    &
    (df_valid["LogP"] >= logp_range[0])
    &
    (df_valid["LogP"] <= logp_range[1])
].copy()


if lipinski_filter == "Drug-like (0 violations)":

    df_filtered = df_filtered[
        df_filtered["LipinskiViolations"] == 0
    ].copy()


elif lipinski_filter == "Non drug-like (>0 violations)":

    df_filtered = df_filtered[
        df_filtered["LipinskiViolations"] > 0
    ].copy()


# ============================================================
# EMPTY FILTER PROTECTION
# ============================================================

if df_filtered.empty:

    st.warning(
        """
        ⚠️ No molecules match the selected filters.

        Please widen the Molecular Weight / LogP ranges
        or change the Lipinski filter.
        """
    )

    st.info(
        f"Total valid molecules in dataset: {len(df_valid)}"
    )

    st.stop()


# ============================================================
# SUMMARY METRICS
# ============================================================

st.subheader("📊 Summary Metrics")

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Molecules",
        f"{len(df_valid):,}"
    )


with col2:

    st.metric(
        "Drug-like Molecules",
        f"{int(df_valid['LipinskiDruglike'].sum()):,}"
    )


with col3:

    st.metric(
        "Average MolWt",
        f"{df_valid['MolWt'].mean():.2f}"
    )


with col4:

    st.metric(
        "Average LogP",
        f"{df_valid['LogP'].mean():.2f}"
    )


st.markdown("---")


# ============================================================
# FILTER STATUS
# ============================================================

st.subheader("🔎 Current Selection")

f1, f2, f3 = st.columns(3)

with f1:

    st.write(
        f"**Molecules:** {len(df_filtered):,}"
    )

with f2:

    st.write(
        f"**MolWt:** {mw_range[0]:.2f} – {mw_range[1]:.2f}"
    )

with f3:

    st.write(
        f"**LogP:** {logp_range[0]:.2f} – {logp_range[1]:.2f}"
    )


st.markdown("---")


# ============================================================
# MAIN LAYOUT
# ============================================================

left_col, right_col = st.columns(
    (3, 2)
)


# ============================================================
# LEFT COLUMN
# ============================================================

with left_col:

    # --------------------------------------------------------
    # BOX PLOTS
    # --------------------------------------------------------

    st.markdown(
        "### 📦 Descriptor Distributions"
    )

    descriptors = [
        "MolWt",
        "LogP",
        "NumHDonors",
        "NumHAcceptors",
        "TPSA",
        "NumRotatableBonds",
    ]

    melted = (
        df_filtered[descriptors]
        .melt(
            var_name="Descriptor",
            value_name="Value"
        )
    )

    fig_box = px.box(
        melted,
        x="Descriptor",
        y="Value",
        points="outliers",
        title="Molecular Descriptor Distributions",
    )

    fig_box.update_layout(
        xaxis_title="Descriptor",
        yaxis_title="Value",
        height=500,
    )

    st.plotly_chart(
        fig_box,
        use_container_width=True
    )


    # --------------------------------------------------------
    # CORRELATION HEATMAP
    # --------------------------------------------------------

    st.markdown(
        "### 🔥 Descriptor Correlation"
    )

    if len(df_filtered) >= 2:

        corr = (
            df_filtered[descriptors]
            .corr()
        )

        fig_heat = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            title="Correlation Heatmap",
        )

        fig_heat.update_layout(
            height=500
        )

        st.plotly_chart(
            fig_heat,
            use_container_width=True
        )

    else:

        st.info(
            "At least two molecules are required "
            "to calculate correlations."
        )


    # --------------------------------------------------------
    # TREND / LINE PLOT
    # --------------------------------------------------------

    st.markdown(
        "### 📈 MolWt vs LogP Trend"
    )

    if len(df_filtered) >= 2:

        # Use fewer bins for small datasets
        number_of_bins = min(
            20,
            max(2, len(df_filtered) // 2)
        )

        temp_df = df_filtered.copy()

        temp_df["MolWt_bin"] = pd.cut(
            temp_df["MolWt"],
            bins=number_of_bins,
            duplicates="drop"
        )

        trend = (
            temp_df
            .groupby(
                "MolWt_bin",
                observed=True
            )
            .agg(
                mean_LogP=("LogP", "mean"),
                count=("LogP", "count"),
            )
            .reset_index()
            .dropna()
        )

        if not trend.empty:

            trend["bin_center"] = (
                trend["MolWt_bin"]
                .apply(lambda x: x.mid)
            )

            fig_line = px.line(
                trend.sort_values("bin_center"),
                x="bin_center",
                y="mean_LogP",
                markers=True,
                labels={
                    "bin_center": "MolWt",
                    "mean_LogP": "Mean LogP",
                },
                title="Mean LogP vs Molecular Weight",
            )

            fig_line.add_trace(
                go.Bar(
                    x=trend["bin_center"],
                    y=trend["count"],
                    name="Molecule Count",
                    opacity=0.25,
                    yaxis="y2",
                )
            )

            fig_line.update_layout(
                yaxis=dict(
                    title="Mean LogP"
                ),
                yaxis2=dict(
                    title="Molecule Count",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                ),
                height=500,
            )

            st.plotly_chart(
                fig_line,
                use_container_width=True
            )

        else:

            st.info(
                "Not enough variation in molecular weight "
                "to generate the trend."
            )

    else:

        st.info(
            "At least two molecules are required "
            "to generate the trend."
        )


# ============================================================
# RIGHT COLUMN
# ============================================================

with right_col:

    # --------------------------------------------------------
    # FILTER SUMMARY
    # --------------------------------------------------------

    st.markdown(
        "### 🎛️ Filters Summary"
    )

    st.write(
        f"**Molecular Weight:** "
        f"{mw_range[0]:.2f} – {mw_range[1]:.2f}"
    )

    st.write(
        f"**LogP:** "
        f"{logp_range[0]:.2f} – {logp_range[1]:.2f}"
    )

    st.write(
        f"**Lipinski:** "
        f"{lipinski_filter}"
    )

    st.write(
        f"**Filtered molecules:** "
        f"{len(df_filtered):,}"
    )


    st.markdown("---")


    # --------------------------------------------------------
    # MOLECULE TABLE
    # --------------------------------------------------------

    st.markdown(
        "### 🧪 Molecule Table"
    )

    display_cols = [
        SMILES_COL,
        "MolWt",
        "LogP",
        "NumHDonors",
        "NumHAcceptors",
        "TPSA",
        "NumRotatableBonds",
        "NumAromaticRings",
        "LipinskiViolations",
    ]

    table_df = (
        df_filtered[display_cols]
        .copy()
    )

    # Round numerical values for readability
    numeric_columns = [
        "MolWt",
        "LogP",
        "TPSA",
    ]

    for column in numeric_columns:

        table_df[column] = (
            table_df[column]
            .round(2)
        )


    sample_size = min(
        200,
        len(table_df)
    )

    if len(table_df) > sample_size:

        table_display = (
            table_df
            .sample(
                sample_size,
                random_state=42
            )
            .reset_index(drop=True)
        )

        st.caption(
            f"Showing a random sample of "
            f"{sample_size} molecules."
        )

    else:

        table_display = (
            table_df
            .reset_index(drop=True)
        )

        st.caption(
            f"Showing all {len(table_df)} molecules."
        )


    st.dataframe(
        table_display,
        use_container_width=True,
        hide_index=True,
    )


    # --------------------------------------------------------
    # MOLECULE VIEWER
    # --------------------------------------------------------

    st.markdown(
        "### 🔬 Molecule Viewer"
    )

    viewer_df = (
        df_filtered
        .reset_index(drop=True)
    )

    if len(viewer_df) > 0:

        idx = st.number_input(
            "Select molecule index:",
            min_value=0,
            max_value=len(viewer_df) - 1,
            value=0,
            step=1,
        )

        selected_row = viewer_df.iloc[
            int(idx)
        ]

        selected_smiles = selected_row[
            SMILES_COL
        ]

        selected_mol = selected_row[
            "mol_obj"
        ]

        st.write(
            f"**SMILES:** `{selected_smiles}`"
        )

        image = draw_molecule_image(
            selected_mol,
            size=(500, 350)
        )

        if image is not None:

            st.image(
                image,
                use_container_width=True
            )

        else:

            st.warning(
                "Could not draw this molecule."
            )

        # Display selected molecule properties
        st.markdown(
            "#### Molecular Properties"
        )

        p1, p2 = st.columns(2)

        with p1:

            st.write(
                f"**MolWt:** "
                f"{selected_row['MolWt']:.2f}"
            )

            st.write(
                f"**LogP:** "
                f"{selected_row['LogP']:.2f}"
            )

            st.write(
                f"**TPSA:** "
                f"{selected_row['TPSA']:.2f}"
            )

            st.write(
                f"**HBD:** "
                f"{int(selected_row['NumHDonors'])}"
            )

        with p2:

            st.write(
                f"**HBA:** "
                f"{int(selected_row['NumHAcceptors'])}"
            )

            st.write(
                f"**Rotatable Bonds:** "
                f"{int(selected_row['NumRotatableBonds'])}"
            )

            st.write(
                f"**Aromatic Rings:** "
                f"{int(selected_row['NumAromaticRings'])}"
            )

            violations = int(
                selected_row["LipinskiViolations"]
            )

            st.write(
                f"**Lipinski Violations:** "
                f"{violations}"
            )


# ============================================================
# DOWNLOAD PROCESSED DATA
# ============================================================

st.markdown("---")

st.subheader(
    "⬇️ Download Processed Dataset"
)


@st.cache_data
def df_to_csv_bytes(dataframe):
    """Convert dataframe to downloadable CSV."""
    return dataframe.drop(
        columns=["mol_obj"],
        errors="ignore"
    ).to_csv(
        index=False
    ).encode("utf-8")


csv_bytes = df_to_csv_bytes(df)


st.download_button(
    label="Download Full Processed CSV",
    data=csv_bytes,
    file_name="molecules_processed.csv",
    mime="text/csv",
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Built with Python • Streamlit • RDKit • Pandas • Plotly"
)
