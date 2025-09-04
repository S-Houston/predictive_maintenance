# Predictive Maintenance with NASA CMAPSS Dataset

## Project Overview
This project develops a predictive maintenance solution for jet engines using the NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset.  
The objective is to predict the **Remaining Useful Life (RUL)** of engines from multivariate sensor data and demonstrate a full **end-to-end ML workflow**.

This project is designed to be portfolio-ready and highlight skills across:
- Exploratory Data Analysis (EDA) and feature engineering  
- ML model development (regression + classification)  
- Reproducibility, versioning, and modular code organisation  
- Experiment tracking and reporting  
- Deployment-ready architecture (FastAPI, testing, CI/CD ready)

---

## Goals
- Understand engine degradation through sensor behavior analysis  
- Engineer labels for **RUL prediction** and **failure classification**  
- Benchmark baseline models and compare with advanced ML methods  
- Build a clean, reproducible pipeline suitable for deployment  
- Demonstrate good MLOps practices (experiment tracking, modular repo, testing)

---

Project Structure
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── cleaned        <- Cleaned and preprocessed data ready for feature engineering and modeling.
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- Makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw or cleaned data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------
## 🔄 Workflow Overview

```mermaid
flowchart TD
    A[📥 Raw Data<br>NASA CMAPSS] --> B[🧹 Data Preparation<br>(cleaning, normalisation, RUL labels)]
    B --> C[🔧 Feature Engineering<br>(rolling stats, degradation heuristics, composite score)]
    C --> D[🤖 Modeling<br>Baseline: Linear/Tree Models<br>Advanced: LSTM/Temporal CNN]
    D --> E[📊 Evaluation<br>Regression: RMSE/MAE<br>Classification: Precision/Recall/F1]
    E --> F[📝 Experiment Tracking<br>MLflow logs & comparisons]
    F --> G[🚀 Deployment<br>FastAPI endpoint + Swagger docs]
    G --> H[📈 Monitoring<br>(Gap: metrics dashboards, drift detection)]





<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
