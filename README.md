# 🔧 Predictive Maintenance Prototype for Nuclear Equipment  
### Machine Learning Proof of Concept (PoC)

This repository contains a Machine Learning prototype designed to detect anomalies and predict failures in industrial pump systems used in nuclear facilities. The goal is to support proactive maintenance actions, reduce downtime, and enhance operational safety in critical energy infrastructures.

---

## 📘 Project Overview

This project implements a binary classification system using time-series sensor data.  
It predicts whether the machine is in:

- **Normal State (0)**
- **Anomaly / Failure-Risk State (1)**  
  (corresponding to *BROKEN* or *RECOVERING* statuses)

The prototype compares several machine learning models and evaluates them using metrics adapted to severe class imbalance.

---

## 🧠 Key Features

- **Time-Series Cleaning & Preprocessing**
- **Imbalanced Class Handling**
  - Class weighting  
  - Macro F1-score optimization
- **Model Benchmarking**
  - Logistic Regression (baseline)  
  - Random Forest (optimized)  
  - XGBoost (optimized)
- **Hyperparameter Optimization**
  - GridSearchCV
- **Evaluation on Rare but Critical Anomalies**

---

## 📊 Dataset

- **Source:** Kaggle – Pump Sensor Data  
- **Link:** https://www.kaggle.com/datasets/nphantawee/pump-sensor-data  
- ~2.2 million rows of sensor readings  
- 52 sensors + timestamps  
- Labels: *NORMAL*, *BROKEN*, *RECOVERING*

The dataset is **not included** in this repository due to its size.  
Download it from Kaggle and place it inside a folder named:
data/



