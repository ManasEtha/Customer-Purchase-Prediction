# Customer Purchase Prediction
# 🛍️ Black Friday Customer Purchase Behavior Predictor

An end-to-end Machine Learning and Business Intelligence solution developed for **ABC Private Limited** to analyze and predict customer purchase amounts during high-volume retail events. This project implements a hybrid machine learning pipeline combining **Unsupervised Clustering (K-Means)** for customer persona segmentation and **Supervised Regression (Random Forest)** for highly personalized purchase amount predictions.

---

## 🎯 Project Overview & Objective

In high-volume retail settings like Black Friday, understanding how customer demographics map to purchasing capacity is critical for profit maximization. 

**ABC Private Limited** aims to transition from generalized promotional targeting to data-driven personalized marketing. This system processes customer demographic markers and product profiles to:
1. **Segment Consumers:** Automatically group customers into behavioral archetypes.
2. **Predict Expenditure:** Estimate the exact purchase amount a customer is likely to spend on a specific category to optimize dynamic offer delivery.

---

## 🏗️ System Architecture & Data Pipeline

The project utilizes a modular data engineering and machine learning workflow:

1. **Exploratory Data Analysis (EDA) & Feature Engineering:**
   * Cleans messy categorical profiles (Gender, Age brackets, City Categories).
   * Implements robust **Label Encoding** for high-cardinality demographic ranges.
   * Handles missing product categories natively (`Product_Category_2` & `Product_Category_3`) to maximize feature retention.

2. **Unsupervised Customer Segmentation (K-Means):**
   * Standardizes continuous demographic fields using a `StandardScaler`.
   * Maps users into **3 Core Data-Driven Personas**:
     * 🛒 *Budget-Conscious Browser*
     * 📊 *Mainstream Consistent Consumer*
     * 💎 *High-Value Power Shopper*
   * The assigned cluster is dynamically injected back into the dataset as a highly predictive synthetic feature.

3. **Supervised Regression Framework (Random Forest):**
   * Trains a multi-tree ensemble model utilizing the engineered structural features combined with the newly minted cluster personas.
   * Outputs localized target expenditure forecasts matching consumer trends.

4. **Dynamic User Interface (Streamlit Cloud Integration):**
   * Loads serialized pipeline artifacts (`rf_model.pkl`, `kmeans_model.pkl`, `scaler.pkl`, `label_encoders.pkl`) securely via optimized resource caching hooks.
   * Provides an intuitive graphical sandbox for business stakeholders to input simulated client features and render instant insights.

---

## 📂 Project Repository Structure

```text
black_friday_analytics/
│
├── Black_Friday(Project).py    # Data acquisition engine & model training pipeline
├── app.py                      # Interactive Streamlit dashboard deployment code
├── README.md                   # Technical portfolio documentation
│
% Artifacts generated automatically during the training execution phase:
├── rf_model.pkl                # Trained Random Forest Regressor weights
├── kmeans_model.pkl            # Trained K-Means Clusterer weights
├── scaler.pkl                  # Serialized demographic feature scaler
├── label_encoders.pkl          # Serialized demographic category mappings
└── feature_names.pkl           # Final standardized feature alignment structure
