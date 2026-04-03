# 🧠 LSTM Time Series Forecasting

[![Framework](https://img.shields.io/badge/Framework-PyTorch_2.5-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Dashboard](https://img.shields.io/badge/UI-Streamlit_1.31-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
![Model](https://img.shields.io/badge/Model-LSTM_Forecasting-purple)
![Task](https://img.shields.io/badge/Task-Time_Series-orange)

> **A professional deep learning implementation from the 13-Project Machine Learning Curriculum.**
> Fully deployed on Hugging Face Spaces and GitHub using custom Docker networking.

---

## 🎯 The Problem
Predicting sequential trends and stock patterns using PyTorch LSTMs.

## 🛠️ Architecture
- **Data Engineering**: Custom `DataLoader` utilities and artifact tracking
- **Model Framework**: PyTorch `nn.Module` with optimized training loops
- **Inference Pipeline**: Real-time prediction tracking inside the GUI

## 🚀 How to Run Locally

```bash
# 1. Clone the master repository
git clone [repo_url]
cd [project_folder]

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the AI Dashboard
streamlit run app.py
```

## 📁 Repository Structure
```text
📦 lstm_time_series_forecasting
 ┣ 📂 data/        # Raw and processed serialized data (.pkl)
 ┣ 📂 models/      # Saved neural network weights and metrics
 ┣ 📂 charts/      # Evaluative tracking graphs
 ┣ 📂 notebooks/   # Original experimentation files
 ┣ 📜 app.py       # Core Streamlit execution layer
 ┣ 📜 Dockerfile   # Deployment configurations
 ┗ 📜 path_utils.py # Resilient pathing handlers
```

---
*Maintained by the automated DL Architecture Pipeline.*
