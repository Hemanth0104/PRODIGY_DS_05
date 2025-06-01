# PRODIGY_DS_05

# 🚧 US Accidents Data Analysis (2016–2022)

This repository presents an in-depth exploratory data analysis (EDA) of the [US Accidents Dataset (March 2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents), a comprehensive dataset that contains information on over 7 million traffic accidents across the United States.

---

## 📁 Dataset Overview

- **Source**: Kaggle – [US Accidents (March 2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
- **Size**: 7+ million records
- **Period Covered**: 2016 to early 2023
- **Features**: Time, location, weather, visibility, severity, etc.

---

## 📌 Objectives

- Perform data cleaning and preprocessing
- Identify and visualize missing values
- Discover relationships between features using correlation
- Visualize accident trends over years, states, cities, weather conditions, and weekdays
- Highlight severity levels and impact
- Standardize features and visualize spatial distribution

---

## 📊 Key Insights

### 🔷 Correlation-Based Analysis

**Strong Positive Correlations (≥ 0.7):**
- `Temperature (°F)` ↔ `Wind Chill (°F)` → **0.99**
- `End Latitude` ↔ `End Longitude` → **0.99**
- `End Latitude` ↔ `Severity` → **0.91**
- `End Longitude` ↔ `Severity` → **0.91**

**Weak Positive Correlations (0.1 to 0.3):**
- `Start Latitude`, `Start Longitude`, `Distance (mi)`, `Wind Speed`, and `Precipitation` vs. `Severity`

**Weak/Negligible Negative Correlations:**
- `Wind Chill`, `Temperature`, and `Visibility` show weak or near-zero correlation with `Severity`

---

### 📍 Location-Based Insights

- **Top 5 States** with highest accident count: **California (CA), Florida (FL), Texas (TX), South Carolina (SC), New York (NY)**
- **Top 5 Cities** with most accidents: **Miami, Houston, Los Angeles, Charlotte, Dallas**

---

### 📆 Time-Based Trends

- Accidents have been **increasing from 2016 to 2021**
- A **slight dip in 2022**, potentially due to improved safety measures or post-pandemic travel patterns

---

### 📅 Weekday vs Weekend

- Accidents on **weekdays are nearly 2x higher** than weekends

---

### 🌦️ Weather Insights

- **Most accidents occurred during "Fair" weather**, suggesting bad weather was not the major factor

---

### 🚨 Severity Insights

- **Severity 2 and 4** had the most impact:
  - **Severity 2** is most common
  - **Severity 4** has the **greatest traffic impact per incident**

---

## 📈 Visuals & Plots

- Correlation Heatmap

 ![crr-1](https://github.com/user-attachments/assets/089813e4-feee-4e85-b184-34ee276c5988)
- Missing Value Bar Plot
  ![bar_plot](https://github.com/user-attachments/assets/d71d636a-94ba-4da7-a89e-5f38a1f8f987)
- Accident Count by:
  - State
  - ![accidents in different sates](https://github.com/user-attachments/assets/9152980c-6802-485e-ba37-509f0cf92d9c)

  - City
  - ![accidents in different cities](https://github.com/user-attachments/assets/23ae49ff-7496-4243-865e-3e72cf6b6475)

  
  - Weekday
    ![accidents on different weekdays](https://github.com/user-attachments/assets/aaa66de3-70cd-4bcf-a428-adabe97bed4f)

  - Weather Conditions
- Temperature Distribution Across Years
  ![Wconditions](https://github.com/user-attachments/assets/79614ad7-55eb-4931-b3fe-bcfd3796a00c)

- Pie Chart for Severity Distribution
- ![piechart](https://github.com/user-attachments/assets/5b1fe60a-936e-4fad-b53c-bd4bf5c38fe9)

- Latitude/Longitude-based Scatter Plot of Accident Locations
- ![map1](https://github.com/user-attachments/assets/011497e8-71ac-42ab-af8b-b736cbc8d828)
![map](https://github.com/user-attachments/assets/c7d1f309-3e3a-4632-aaf7-171402d82062)


---

## 🛠️ Tech Stack & Libraries

- **Languages**: Python
- **Libraries**:
  - `pandas`, `numpy`
  - `matplotlib`, `seaborn`
  - `scikit-learn`

---


