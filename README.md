# SONAR-Rock-vs-Mine-Prediction-with-Python-End-To-End-Python-Machine-Learning-Project


This project applies **Machine Learning** to classify sonar signals as either **Rock (R)** or **Mine (M)** using **Logistic Regression**.  
It demonstrates data preprocessing, model training, evaluation, and prediction — forming a complete ML pipeline.

---

## 📌 Project Overview
Sonar systems emit acoustic signals that bounce back differently from underwater objects.  
Classifying these signals accurately can be critical for applications such as:
- **Marine safety**
- **Defense systems**
- **Seafloor exploration**

This project:
1. Loads and processes sonar dataset signals.
2. Trains a Logistic Regression classifier.
3. Evaluates model accuracy on both training and test data.
4. Predicts the class of a custom sonar reading.

---

## 📂 Dataset
- **Source:** [UCI Machine Learning Repository – Sonar Mines vs Rocks](https://drive.google.com/file/d/1pQxtljlNVh0DHYg-Ye7dtpDTlFceHVfa/view)  
- **Shape:** 208 samples × 60 features  
- **Labels:**  
  - `M` → Mine  
  - `R` → Rock  

The dataset contains energy values of sonar returns at different frequency bands.

---

## ⚙️ Methodology
1. **Data Loading & Exploration**  
   - Pandas for loading CSV data  
   - Descriptive statistics & label distribution analysis  

2. **Preprocessing**  
   - Separate features (X) and labels (Y)  
   - Train-test split (90% training, 10% testing) with stratification  

3. **Model Training**  
   - Algorithm: **Logistic Regression**  
   

4. **Evaluation**  
   - Metrics: Accuracy on train and test sets  
    

5. **Prediction System**  
   - Accepts new sonar readings as input  
   - Outputs prediction (`Mine` or `Rock`)

---

## 📊 Results
| Dataset      | Accuracy |
|--------------|----------|
| Training Set | 85%      |
| Test Set     | 78%      |





## 🚀 How to Run
### **1. Clone the repository**
```bash
git clone https://github.com/riminipa16/SONAR-Rock-vs-Mine-Prediction-with-Python-End-To-End-Python-Machine-Learning-Project
cd sonar-rock-vs-mine
