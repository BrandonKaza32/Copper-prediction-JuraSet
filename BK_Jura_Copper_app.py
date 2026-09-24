"""
Copper (Cu) class prediction for Swiss Jura topsoils.

Run locally:   streamlit run app.py
The app trains the models when it starts (about 5 seconds) and caches them,
so every number shown is calculated by the same code as the notebook.
"""

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------
st.set_page_config(page_title="Copper Prediction – Jura Soils", page_icon="🟠", layout="wide")

# Look for the CSVs in a "data" folder next to app.py; if they are not there, look next to app.py itself
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
if not os.path.exists(os.path.join(DATA_DIR, "jura_pred.csv")):
    DATA_DIR = BASE_DIR
METALS = ["Cd", "Co", "Cr", "Ni", "Pb", "Zn"]
CLASSES = ["low", "medium", "high"]
COLOURS = {"low": "#3A9E96", "medium": "#E0B04A", "high": "#B5562C"}
LANDUSE = {1: "Forest", 2: "Pasture", 3: "Meadow", 4: "Tillage"}
ROCK = {1: "Argovian", 2: "Kimmeridgian", 3: "Sequanian", 4: "Portlandian", 5: "Quaternary"}


# ------------------------------------------------------------------
# Helper functions (same steps as the notebook)
# ------------------------------------------------------------------
def to_codes(d):
    """If Landuse / Rock are written as words, change them to the numbers used in jura_pred.csv."""
    d = d.copy()
    for col, names in [("Landuse", LANDUSE), ("Rock", ROCK)]:
        if not pd.api.types.is_numeric_dtype(d[col]):
            lookup = {name.lower(): code for code, name in names.items()}
            d[col] = d[col].astype(str).str.strip().str.lower().map(lookup)
    return d


def make_features(d, columns=None):
    """Log-transform the metals and one-hot encode Landuse and Rock."""
    X = d[["Xloc", "Yloc", "Landuse", "Rock"] + METALS].copy()
    for m in METALS:
        X[m] = np.log1p(X[m])
    X = pd.get_dummies(X, columns=["Landuse", "Rock"], dtype=int)
    if columns is not None:                      # make sure new data has the same columns as training
        X = X.reindex(columns=columns, fill_value=0)
    return X


def make_models():
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "Linear SVM": make_pipeline(StandardScaler(), SVC(kernel="linear")),
        "RBF SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=3)),
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=2, random_state=42),
    }


def report_table(y_true, y_pred):
    rep = classification_report(y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0)
    table = pd.DataFrame(rep).T.loc[CLASSES, ["precision", "recall", "f1-score", "support"]]
    table["support"] = table["support"].astype(int)
    return table.round(2)


def confusion_figure(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)   # labels= keeps low, medium, high order
    fig = px.imshow(cm, x=CLASSES, y=CLASSES, text_auto=True, color_continuous_scale="Oranges",
                    labels=dict(x="Predicted", y="Actual", color="Samples"), title=title)
    fig.update_layout(coloraxis_showscale=False, height=380, margin=dict(t=50, b=10))
    fig.update_traces(textfont_size=18)
    return fig


# ------------------------------------------------------------------
# Load data (cached)
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "jura_pred.csv"))

    val = None
    val_path = os.path.join(DATA_DIR, "jura_val.csv")
    if os.path.exists(val_path):
        v = pd.read_csv(val_path)
        if "Cu" in v.columns:                    # only a real validation file has Cu
            val = to_codes(v)

    grid = None
    grid_path = os.path.join(DATA_DIR, "jura_grid.csv")
    if os.path.exists(grid_path):
        grid = to_codes(pd.read_csv(grid_path)).dropna(subset=["Landuse", "Rock"])

    return df, val, grid


# ------------------------------------------------------------------
# Train and evaluate everything (cached, runs once)
# ------------------------------------------------------------------
@st.cache_resource
def run_analysis(df, val, grid):
    out = {}

    # classes
    y, bins = pd.qcut(df["Cu"], q=3, labels=CLASSES, retbins=True)
    y = y.astype(str)
    out["bins"] = bins
    X = make_features(df)
    out["columns"] = X.columns

    # cross-validation
    blocks = df["Xloc"].astype(int).astype(str) + "_" + df["Yloc"].astype(int).astype(str)
    random_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    spatial_cv = GroupKFold(n_splits=5)
    rows = []
    for name, model in make_models().items():
        r = cross_val_score(model, X, y, cv=random_cv)
        s = cross_val_score(model, X, y, cv=spatial_cv, groups=blocks)
        rows.append([name, r.mean(), r.std(), s.mean(), s.std()])
    cv = pd.DataFrame(rows, columns=["Model", "Random CV", "Random sd", "Spatial CV", "Spatial sd"])
    out["cv"] = cv
    out["n_blocks"] = blocks.nunique()
    best = cv.sort_values("Spatial CV", ascending=False)["Model"].iloc[0]
    out["best"] = best

    # hold-out 80/20 split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    hold = make_models()[best].fit(X_tr, y_tr)
    out["holdout"] = (y_te, hold.predict(X_te))

    # feature importance (Random Forest), one-hot columns added back together
    rf = make_models()["Random Forest"].fit(X_tr, y_tr)
    imp = pd.Series(rf.feature_importances_, index=X.columns)
    grouped = imp.drop(imp.filter(regex="^(Landuse|Rock)_").index)
    grouped["Landuse"] = imp.filter(like="Landuse_").sum()
    grouped["Rock"] = imp.filter(like="Rock_").sum()
    out["importance"] = grouped.sort_values()

    # final models trained on all 259 samples
    final = {name: m.fit(X, y) for name, m in make_models().items()}
    out["final"] = final

    # independent validation
    if val is not None:
        y_val = pd.cut(val["Cu"], bins=[-np.inf, bins[1], bins[2], np.inf], labels=CLASSES).astype(str)
        v_pred = final[best].predict(make_features(val, X.columns))
        out["val"] = (y_val, v_pred)

    # map: estimate metals on the grid from the 12 nearest samples, then predict
    if grid is not None:
        g = grid.copy()
        for m in METALS:
            knn = KNeighborsRegressor(n_neighbors=12, weights="distance")
            knn.fit(df[["Xloc", "Yloc"]], np.log1p(df[m]))
            g[m] = np.expm1(knn.predict(g[["Xloc", "Yloc"]]))   # back to mg/kg; make_features logs again
        g["Cu_pred"] = final[best].predict(make_features(g, X.columns))
        out["grid"] = g

    return out


df, val, grid = load_data()
with st.spinner("Training models… (only on first load)"):
    res = run_analysis(df, val, grid)
best = res["best"]
t1, t2 = res["bins"][1], res["bins"][2]

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.title("Predicting Copper Concentration in Jura Soils")
st.caption("Brandon Kazangarare · Mining Engineering, Central South University · "
           "Classifying topsoil copper as low, medium or high from co-occurring metals and site data")

tab_overview, tab_data, tab_models, tab_val, tab_map, tab_predict = st.tabs(
    ["Overview", "Data", "Models", "Validation", "Map", "Try a prediction"])

# ------------------------------------------------------------------
# Overview
# ------------------------------------------------------------------
with tab_overview:
    best_row = res["cv"].set_index("Model").loc[best]
    c1, c2, c3, c4 = st.columns(4)
    if "val" in res:
        y_val, v_pred = res["val"]
        high_recall = report_table(y_val, v_pred).loc["high", "recall"]
        c1.metric("Independent validation accuracy", f"{accuracy_score(y_val, v_pred):.0%}",
                  help=f"{len(y_val)} samples never used for training or model choice")
        c3.metric("High-Cu samples detected", f"{high_recall:.0%}", help="Recall of the high class on validation")
    else:
        c1.metric("Hold-out accuracy", f"{accuracy_score(*res['holdout']):.0%}")
        c3.metric("High-Cu recall (hold-out)", f"{report_table(*res['holdout']).loc['high', 'recall']:.0%}")
    c2.metric(f"Spatial CV accuracy ({best})", f"{best_row['Spatial CV']:.1%}",
              help="Whole 1 km blocks held out, so the model is tested on areas it has not seen")
    c4.metric("Training samples", f"{len(df)}")

    st.subheader("What this project does")
    st.markdown(f"""
Copper in topsoil is costly to measure everywhere. This project asks whether a sample's copper level can be
predicted from **other metals** (Cd, Co, Cr, Ni, Pb, Zn), **land use**, **rock type** and **location**.

* Copper is split into three equal-sized classes: **low** (< {t1:.1f} mg/kg), **medium** ({t1:.1f}–{t2:.1f}) and **high** (> {t2:.1f}).
* Metals are log-transformed because they are strongly skewed; land use and rock are one-hot encoded because they are categories.
* Four models are compared with **random** and **spatial** 5-fold cross-validation. The best by spatial CV (**{best}**) is then tested on an independent validation set and used to map the whole study area.
""")
    st.info("Explore the tabs above, or open **Try a prediction** to classify your own sample.")

# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
with tab_data:
    st.subheader("The training data")
    show = df.copy()
    show["Landuse"] = show["Landuse"].map(LANDUSE)
    show["Rock"] = show["Rock"].map(ROCK)
    st.dataframe(show, height=260, width="stretch")
    st.caption("Coordinates in km, metal concentrations in mg/kg. Source: Goovaerts (1997), distributed with the R package gstat.")

    left, right = st.columns(2)
    with left:
        skew = pd.DataFrame({"Raw": df[METALS + ["Cu"]].skew(),
                             "log(1 + x)": np.log1p(df[METALS + ["Cu"]]).skew()}).reset_index(names="Metal")
        fig = px.bar(skew.melt(id_vars="Metal", var_name="Scale", value_name="Skewness"),
                     x="Metal", y="Skewness", color="Scale", barmode="group",
                     color_discrete_sequence=["#B5562C", "#3A9E96"], title="Skewness before and after log transform")
        st.plotly_chart(fig, width="stretch")
    with right:
        scale = st.radio("Cu scale", ["Raw (mg/kg)", "log(1 + Cu)"], horizontal=True)
        values = df["Cu"] if scale.startswith("Raw") else np.log1p(df["Cu"])
        fig = px.histogram(values, nbins=30, title="Distribution of Cu", color_discrete_sequence=["#B5562C"])
        fig.update_layout(showlegend=False, xaxis_title=scale, yaxis_title="Samples")
        st.plotly_chart(fig, width="stretch")
    st.caption("Skewness above about 1 means a long right tail. The log transform brings the metals close to symmetric, "
               "so all 259 samples are kept instead of deleting the high values.")

# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------
with tab_models:
    st.subheader("Model comparison")
    cv = res["cv"]
    long = cv.melt(id_vars="Model", value_vars=["Random CV", "Spatial CV"], var_name="Test", value_name="Accuracy")
    fig = px.bar(long, x="Model", y="Accuracy", color="Test", barmode="group", text_auto=".1%",
                 color_discrete_sequence=["#9AA5A8", "#B5562C"])
    fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0.5, 0.9], height=400)
    st.plotly_chart(fig, width="stretch")

    table = cv.copy()
    for a, b in [("Random CV", "Random sd"), ("Spatial CV", "Spatial sd")]:
        table[a] = table[a].map("{:.3f}".format) + " ± " + table[b].map("{:.3f}".format)
    st.dataframe(table[["Model", "Random CV", "Spatial CV"]], hide_index=True, width="stretch")
    st.markdown(f"""
**Random CV** shuffles samples into 5 groups. **Spatial CV** splits the area into {res['n_blocks']} blocks of 1 km × 1 km
and holds out whole blocks, which is a fairer test because nearby soil samples are similar.
**{best}** has the highest spatial CV score and was chosen, but the top models are within one standard deviation of each other.
""")

    left, right = st.columns(2)
    with left:
        y_te, p_te = res["holdout"]
        st.plotly_chart(confusion_figure(y_te, p_te, f"{best} – 80/20 hold-out ({len(y_te)} samples)"), width="stretch")
        st.dataframe(report_table(y_te, p_te), width="stretch")
    with right:
        imp = res["importance"]
        fig = px.bar(x=imp.values, y=imp.index, orientation="h", title="Feature importance (Random Forest)",
                     color_discrete_sequence=["#B5562C"], text_auto=".2f")
        fig.update_layout(xaxis_title="Importance", yaxis_title="", height=460)
        st.plotly_chart(fig, width="stretch")
        st.caption("Land use and rock type were one-hot encoded, so their columns are added together here.")

# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------
with tab_val:
    if "val" in res:
        y_val, v_pred = res["val"]
        st.subheader("Independent validation set")
        st.markdown(f"""
{best} was trained on all {len(df)} samples and tested **once** on {len(y_val)} samples from different locations.
These samples were never used to choose or tune the model, so this is the most honest score.
""")
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{accuracy_score(y_val, v_pred):.1%}")
        c2.metric("Weighted F1", f"{f1_score(y_val, v_pred, average='weighted'):.3f}")
        left, right = st.columns(2)
        with left:
            st.plotly_chart(confusion_figure(y_val, v_pred, "Validation confusion matrix"), width="stretch")
        with right:
            st.dataframe(report_table(y_val, v_pred), width="stretch")
            st.caption("The class boundaries from the training data are reused, so validation classes are not exactly balanced.")
    else:
        st.info("No validation file with a Cu column was found in the data folder, so this section is skipped.")

# ------------------------------------------------------------------
# Map
# ------------------------------------------------------------------
with tab_map:
    if "grid" in res:
        g = res["grid"].copy()
        st.subheader(f"Predicted copper class across the study area ({best})")
        g["Land use"] = g["Landuse"].map(LANDUSE)
        g["Rock type"] = g["Rock"].map(ROCK)
        fig = px.scatter(g, x="Xloc", y="Yloc", color="Cu_pred", color_discrete_map=COLOURS,
                         category_orders={"Cu_pred": CLASSES},
                         hover_data={"Xloc": ":.2f", "Yloc": ":.2f", "Land use": True, "Rock type": True, "Cu_pred": True},
                         labels={"Cu_pred": "Predicted Cu", "Xloc": "X (km)", "Yloc": "Y (km)"})
        fig.update_traces(marker=dict(symbol="square", size=5))
        if st.checkbox("Show sample locations", value=True):
            fig.add_trace(go.Scattergl(x=df["Xloc"], y=df["Yloc"], mode="markers", name="samples",
                                     marker=dict(color="#9B59D0", size=6, line=dict(color="white", width=0.8)), hovertemplate="Sample Cu: %{text} mg/kg",
                                     text=df["Cu"]))
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_xaxes(constrain="domain")
        fig.update_layout(height=780, legend=dict(
            title=dict(text="Predicted Cu", font=dict(size=18)),
            font=dict(size=16),
            itemsizing="constant",
            bgcolor="rgba(128,128,128,0.15)",
            bordercolor="gray",
            borderwidth=1,
            x=1.0, xanchor="left", y=0.95),
                         )
        st.plotly_chart(fig, width="stretch")

        shares = g["Cu_pred"].value_counts(normalize=True).reindex(CLASSES)
        c1, c2, c3 = st.columns(3)
        for col, cls in zip([c1, c2, c3], CLASSES):
            col.metric(f"{cls.capitalize()} Cu", f"{shares[cls]:.1%} of area")
        st.caption("Land use and rock type come from the grid file. Metal values at each grid point are estimated from the "
                   "12 nearest samples, weighted by distance, so the map is least reliable far from samples.")
    else:
        st.info("No jura_grid.csv found in the data folder, so the map is skipped.")

# ------------------------------------------------------------------
# Try a prediction
# ------------------------------------------------------------------
with tab_predict:
    st.subheader("Classify a new soil sample")
    st.markdown("Set the sample's properties below. Sliders start at the median of the training data.")

    with st.form("predict"):
        c1, c2 = st.columns(2)
        landuse = c1.selectbox("Land use", list(LANDUSE.values()), index=2)
        rock = c2.selectbox("Rock type", list(ROCK.values()), index=2)
        xloc = c1.slider("X coordinate (km)", float(df["Xloc"].min()), float(df["Xloc"].max()), float(df["Xloc"].median()))
        yloc = c2.slider("Y coordinate (km)", float(df["Yloc"].min()), float(df["Yloc"].max()), float(df["Yloc"].median()))

        st.markdown("**Metal concentrations (mg/kg)**")
        cols = st.columns(3)
        metal_values = {}
        for i, m in enumerate(METALS):
            lo, hi, med = float(df[m].min()), float(df[m].max()), float(df[m].median())
            metal_values[m] = cols[i % 3].slider(m, lo, hi, med)
        submitted = st.form_submit_button("Predict copper class", type="primary")

    if submitted:
        sample = pd.DataFrame([{
            "Xloc": xloc, "Yloc": yloc,
            "Landuse": {v: k for k, v in LANDUSE.items()}[landuse],
            "Rock": {v: k for k, v in ROCK.items()}[rock],
            **metal_values,
        }])
        Xs = make_features(sample, res["columns"])
        pred = res["final"][best].predict(Xs)[0]
        ranges = {"low": f"below {t1:.1f} mg/kg", "medium": f"{t1:.1f}–{t2:.1f} mg/kg", "high": f"above {t2:.1f} mg/kg"}

        st.markdown(
            f"<div style='padding:18px;border-radius:10px;background:{COLOURS[pred]};color:white;font-size:22px'>"
            f"Predicted copper class: <b>{pred.upper()}</b> ({ranges[pred]})</div>", unsafe_allow_html=True)

        votes = {name: m.predict(Xs)[0] for name, m in res["final"].items()}
        agree = sum(v == pred for v in votes.values())
        st.markdown(f"**{agree} of 4 models agree.**")
        st.dataframe(pd.DataFrame(votes.items(), columns=["Model", "Prediction"]), hide_index=True, width="stretch")

        # nearest real sample, as a sanity check
        dist = np.hypot(df["Xloc"] - xloc, df["Yloc"] - yloc)
        near = df.loc[dist.idxmin()]
        st.caption(f"Nearest training sample is {dist.min():.2f} km away, with measured Cu = {near['Cu']:.1f} mg/kg.")
        if dist.min() > 0.5:
            st.warning("This location is far from any training sample, so treat the prediction with caution.")

st.divider()
st.caption("Data: Goovaerts, P. (1997). Geostatistics for Natural Resources Evaluation. Oxford University Press; "
           "distributed with the R package gstat. Built with scikit-learn and Streamlit.")
