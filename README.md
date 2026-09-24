# 🟠 Predicting Copper Concentration in Jura Soils

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://BK_Jura_Copper_app.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.64-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-7.1-3F4F75?logo=plotly&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

A machine-learning project that classifies copper (Cu) in Swiss Jura topsoils as **low**, **medium** or **high**, using co-occurring heavy metals, land use, rock type and location. It then maps the predicted copper class across the whole study area.

**🔗 Live app:** https://YOUR-APP-NAME.streamlit.app

---

## 📌 Key results

| Metric | Result |
|---|---|
| Independent validation accuracy (100 unseen samples) | **80.0%** |
| Spatial cross-validation accuracy (Linear SVM) | **74.5% ± 7.0%** |
| High-Cu samples correctly detected (validation recall) | **92%** |
| Most important predictors | Pb, Zn, land use |

### Model comparison (5-fold cross-validation)

| Model | Random CV | Spatial CV |
|---|---|---|
| Logistic Regression | 0.791 ± 0.023 | 0.737 ± 0.051 |
| **Linear SVM** | 0.784 ± 0.042 | **0.745 ± 0.070** |
| RBF SVM | 0.741 ± 0.030 | 0.702 ± 0.092 |
| Random Forest | 0.784 ± 0.037 | 0.741 ± 0.061 |

Linear SVM was selected because it had the highest spatial CV score. The top three models are within one standard deviation of each other, so they perform about equally.

---

## 🖥️ App features

| Tab | What it shows |
|---|---|
| **Overview** | Headline metrics and a summary of the method |
| **Data** | The training data, skewness before and after the log transform, and the Cu distribution |
| **Models** | Random vs. spatial CV comparison, hold-out confusion matrix, feature importance |
| **Validation** | Accuracy, confusion matrix and per-class scores on the independent validation set |
| **Map** | Interactive map of predicted Cu class over 5,957 grid points |
| **Try a prediction** | Enter a sample's metals, land use, rock type and location to get a predicted Cu class, plus how many of the 4 models agree |

<!-- Add a screenshot: save it as images/app_screenshot.png, then remove the comment markers below -->
<!-- ![App screenshot](images/app_screenshot.png) -->

---

## 🧪 Method

1. **Classes.** Cu is split at its tertiles into three equal-sized classes: low (< 13.1 mg/kg), medium (13.1–23.4 mg/kg) and high (> 23.4 mg/kg).
2. **Skewed data.** Metal concentrations are strongly right-skewed (Cu skewness 2.88, Pb 2.91). They are log-transformed with `log(1 + x)` instead of removing high values, because the high values are real contaminated sites.
3. **Categories.** Land use and rock type are category codes, so they are one-hot encoded.
4. **Models.** Logistic Regression, Linear SVM, RBF SVM and Random Forest. Scaling is fitted inside a pipeline on training data only.
5. **Testing.**
   - *Random 5-fold CV* shuffles samples into 5 groups.
   - *Spatial 5-fold CV* divides the area into 21 blocks of 1 km × 1 km and holds out whole blocks. This is fairer, because nearby soil samples are similar.
   - *Independent validation* trains the chosen model on all 259 samples and tests it once on 100 samples from different locations.
6. **Map.** At each grid point, metal values are estimated from the 12 nearest samples (inverse-distance weighted), and the model predicts the Cu class.

### Land use and rock type codes

| Code | Land use | Rock type |
|---|---|---|
| 1 | Forest | Argovian |
| 2 | Pasture | Kimmeridgian |
| 3 | Meadow | Sequanian |
| 4 | Tillage | Portlandian |
| 5 | — | Quaternary |

---

## 📂 Project structure

```
copper-prediction-jura/
├── app.py                          # Streamlit app
├── requirements.txt                # Pinned package versions
├── Copper_Prediction_From_Jura_Data_Set.ipynb   # Analysis notebook
├── data/
│   ├── jura_pred.csv               # 259 training samples
│   ├── jura_val.csv                # 100 independent validation samples
│   └── jura_grid.csv               # 5,957 grid points for the map
├── .streamlit/
│   └── config.toml                 # App colour theme
├── LICENSE
└── README.md
```

---

## 🚀 Run locally

```bash
git clone https://github.com/BrandonKaza32/copper-prediction-jura.git
cd copper-prediction-jura
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. The models train on first load, which takes a few seconds, and are then cached.

---

## ⚠️ Limitations

- **Metals predicting metals.** Pb and Zn drive the model, but a lab that measures them usually measures Cu too. The approach is most useful with cheaper proxies such as portable XRF.
- **Small dataset.** 259 samples in 21 spatial blocks, so fold scores vary by ±5–9 percentage points.
- **Interpolated map.** Metal values on the grid are estimated, not measured, so the map is least reliable far from samples.
- **Statistical classes.** Tertile boundaries have no regulatory meaning. Classes based on soil guideline values would be more actionable.
- **Version sensitivity.** Two samples lie exactly on the class boundaries, and pandas 2 and 3 assign them differently, which shifts results slightly. `requirements.txt` pins versions for reproducibility.

---

## 📚 Data source

The Jura dataset comes from:

> Goovaerts, P. (1997). *Geostatistics for Natural Resources Evaluation*. Oxford University Press.

It is distributed with the R package **gstat**:

> Pebesma, E. J. (2004). Multivariable geostatistics in S: the gstat package. *Computers & Geosciences*, 30, 683–691.

---

## 👤 Author

**Brandon Kazangarare**
Mining Engineering (Class 2304), School of Resources and Safety Engineering, Central South University

[![GitHub](https://img.shields.io/badge/GitHub-BrandonKaza32-181717?logo=github)](https://github.com/BrandonKaza32)

## 📄 License

The code is released under the [MIT License](LICENSE). The Jura data belongs to its original authors. Please cite the sources above if you use it.
