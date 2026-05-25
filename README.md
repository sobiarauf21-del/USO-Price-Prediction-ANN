# Oil ETF (USO) Closing Price Prediction using Deep Learning

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange?style=flat-square&logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=flat-square&logo=flask)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-blue?style=flat-square&logo=scikit-learn)

This project applies **Artificial Neural Networks (ANN)** to predict the daily closing price of the **USO oil ETF** using 24 financial market indicators. The dataset contains 1,718 trading days of data (2011–2018) covering gold, indices, currencies, and commodities.

The motivation behind this project was to compare a traditional ML approach (Linear Regression with just 3 features, R² = 0.74) against a deep learning model trained on richer features — and see how much improvement ANN actually brings.

>  Deep Learning Lab Final Project — 6th Semester

---

##  What This Project Covers

- Building and training a multi-layer ANN from scratch using PyTorch
- Understanding **forward pass** and **backpropagation** in practice
- Comparing **Adam vs RMSProp** optimizers on real financial data
- Visualizing **Sigmoid, Tanh, and ReLU** activation functions
- Running **Naive Bayes** as a classification baseline
- Deploying the trained model as a **Flask web application**

---

##  Dataset

**`oilprice.csv`** — 1,718 rows, 81 columns of daily financial market data

The 24 features used for training are grouped below:

| Group | Features |
|---|---|
| Gold ETF | Open, High, Low, Volume |
| S&P 500 | SP_open, SP_high, SP_close |
| Dow Jones | DJ_open, DJ_close |
| Emerging Markets | EG_close |
| Currencies | EU_Price (Euro/USD), USDI_Price (USD Index) |
| Oil | OF_Price (Futures), OS_Price (Spot) |
| Precious Metals | SF_Price, PLT_Price, PLD_Price, RHO_PRICE |
| Gold Miners ETF | GDX_Close, GDX_Volume |
| USO other fields | USO_Open, USO_High, USO_Low, USO_Volume |
|  Target | **USO_Close** |

---

## Network Architecture

A 4-layer fully connected ANN with dropout regularization:

```
Input Layer     →  24 features
Hidden Layer 1  →  128 neurons, ReLU, Dropout(0.2)
Hidden Layer 2  →  64 neurons,  ReLU, Dropout(0.2)
Hidden Layer 3  →  32 neurons,  ReLU
Output Layer    →  1 neuron (predicted USO Close price)
```

Trained using **MSE loss** and the **Adam optimizer** (lr = 0.001) for 50 epochs.

---

##  Results

### Regression (USO Close Price)

| | Linear Regression (3 features) | ANN — Adam (24 features) |
|---|---|---|
| R² Score | 0.7405 | **~0.97+** |
| RMSE | 6.51 | **~0.60** |
| MAE | 5.05 | **~0.40** |

### Optimizer Comparison

| Optimizer | Result |
|---|---|
| RMSProp | Good convergence |
| **Adam** | **Lower loss, faster convergence** |

### Naive Bayes (Price Category Classification)

Demand bucketed into Low / Medium / High using 33rd and 66th percentiles.

| Metric | Score |
|---|---|
| Accuracy | ~0.95+ |

---

## 🌐 Flask Web App

After training, the model is served through a Flask backend with a clean HTML/CSS/JS frontend. Enter market values for the day and get the predicted USO close price instantly.

To run it locally (VS Code / Jupyter):
```bash
python app.py
# then open http://127.0.0.1:5002
```

---

## ⚙️ Setup

```bash
git clone https://github.com/your-username/USO-Price-Prediction-ANN.git
cd USO-Price-Prediction-ANN
pip install torch pandas numpy matplotlib scikit-learn flask
```

Open `project.ipynb`, run cells 1–8 to train, then cell 9 to launch the web app.

---

##  Files

```
project.ipynb        ← main notebook (training + deployment)
app.py               ← standalone Flask server
index.html           ← web app UI
oilprice.csv         ← dataset
uso_ann_model.pth    ← saved model weights (after training)
requirements.txt     ← dependencies
```

---

##  Built With

PyTorch · Scikit-learn · Flask · Pandas · NumPy · Matplotlib

---

> 6th Semester — Deep Learning Lab Final Project
