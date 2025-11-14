# 🚀 CPU Usage Prediction Dashboard
A machine-learning dashboard built with **Streamlit** + **MLflow** to predict CPU usage of Kubernetes pods and compare model performance across multiple algorithms.

This project helps you:
- Track multiple ML models using **MLflow**
- Find the best model for predicting CPU usage
- Visualize feature importance, RMSE, R², and model timings
- Upload your own CSV for batch predictions
- Run **single-row prediction** against **all trained models**
- Deploy the Streamlit app easily (Streamlit Cloud / Docker / Local)

## 📂 Project Structure
```
├── app_streamlit.py
├── train_and_track.py
├── artifacts/          (ignored in Git)
├── mlruns/             (ignored in Git)
├── requirements.txt
├── README.md
└── .gitignore
```

## ✨ Features
### 🔍 Dataset-wide predictions
Upload a CSV containing required CPU/memory fields and optional `cpu_usage` for evaluation.

### 🤖 Single-Row Model Comparison
The app outputs a table:
| Model | Prediction |
|-------|------------|
| local_best_model | 0.00412 |
| LinearRegression | 0.00201 |
| Lasso | 0.00094 |
| RandomForest | 0.01422 |

### 📈 MLflow Integration
- Full experiment tracking
- RMSE-sorted tables
- Model timings
- Per-run predictions & comparison

## 🛠️ Installation
```
git clone https://github.com/Inferno5704/Cpu-Predictor
cd <repo-name>
python -m venv myenv
myenv/Scripts/activate   # Windows
pip install -r requirements.txt
```

## 🧪 Training
```
python train_and_track.py
```

## 🖥️ Run Dashboard
```
streamlit run app_streamlit.py
```

## 🚀 Deploy on Streamlit Cloud
Push repo → select app file → deploy.

## 🗂️ .gitignore
```
mlruns/
artifacts/
*.pkl
myenv/
__pycache__/
```

## 📜 License
MIT License
