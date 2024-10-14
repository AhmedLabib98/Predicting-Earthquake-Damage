# 🌍 **Predicting Earthquake Damage** 🏢💥

[![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Project-success?style=flat&logo=data:image/svg+xml;base64,<logo>)](#)

This project is part of the **Nepal Earthquake Damage** prediction competition hosted on [DrivenData](https://www.drivendata.org/competitions/57/nepal-earthquake/page/136/). The goal is to predict the severity of damage to buildings caused by earthquakes, using machine learning models trained on various building attributes, geographic information, and historical earthquake data.


## 🌟 **Project Overview**

In this project, we aim to classify the **damage grade** of buildings after an earthquake, based on features like **geographic location**, **building materials**, and other structural/non-structural attributes.


- **Target variable**: `damage_grade`
  - `1`: Minor damage 🟢
  - `2`: Moderate damage 🟡
  - `3`: Severe damage 🔴

### 🎯 **Features**:
- **Geographic identifiers**: `geo_level_1_id`, `geo_level_2_id`, `geo_level_3_id`
- **Building characteristics**: Foundation type, roof type, and age of the building.
- **Usage features**: Whether the building has secondary uses (e.g., agriculture, industry).
- **Target**: `damage_grade`

---

## 📂 **Project Structure**
```bash
├── src/
│   ├── main.py                 # Main entry point for training models
│   ├── target_encoding.py       # Functions for target encoding
│   ├── encoding.py              # Label encoding and basic encoders
│   ├── scratch_book.py          # Code testing and experiments
├── data/                        # Directory for datasets
├── README.md                    # This README file
└── requirements.txt             # Python dependencies
```

### 📂 **Key Scripts**:
- **`main.py`**: Loads the data, processes the features, and trains the machine learning models.
- **`target_encoding.py`**: Contains functions to perform **Target Encoding**, converting categorical variables into meaningful numerical representations based on the target variable.
- **`encoding.py`**: Includes basic label encoding for categorical features.
- **`scratch_book.py`**: A playground for experimenting with code snippets and functions.

---

## ⚙️ **How It Works**

### 1. **Data Loading** 📊
   We load both training and test data using the `data_loading()` function. This extracts building attributes and the target variable, `damage_grade`, for the training set.

### 2. **Label Encoding** 🔢
   Categorical features that are not being target-encoded are label-encoded using the **`basic_encoding()`** function. This helps the model process categorical data as integers.

### 3. **Target Encoding** 🎯
   - For important geographic features (`geo_level_1_id`, `geo_level_2_id`, `geo_level_3_id`), we use **Target Encoding**. 
   - **Training Data** is encoded using the `target_encoding_train()` function, which calculates the mean target value for each category.
   - **Test Data** is encoded using the `target_encoding_test()` function, applying the same encoding learned from the training data, without using the test target.

### 4. **Cross-Validation** 🔍
   We implement **StratifiedKFold Cross-Validation** to ensure that the class distribution of `damage_grade` is preserved across the folds during model training.

---

## 🚀 **Installation and Usage**

### 1. Clone the repository:
```bash
    git clone https://github.com/AhmedLabib98/Predicting-Earthquake-Damage.git
    cd Predicting-Earthquake-Damage
```
  2. Install dependencies:
```bash
    pip install -r requirements.txt
```
3. Run the project:
```bash
python src/main.py
```
This command will preprocess the data, apply encoding techniques, train the model, and generate predictions for the test set.

## 📈 **Results**

After applying **Target Encoding** and other feature engineering techniques, our models achieve a strong performance in predicting the damage grades. Key machine learning algorithms such as **Random Forests**, **XGBoost**, and **Logistic Regression** have been fine-tuned for the best results.

### Example Metrics:
- **Accuracy**: Achieved competitive accuracy in predicting earthquake damage.
- **Cross-Validation**: Robust performance with balanced class representation.

---

## 🤝 **Contributing**

We welcome contributions! If you'd like to improve this project, feel free to fork the repository, create a new branch, and submit a pull request. Any feedback or suggestions are also appreciated via GitHub issues.

---

## 📄 **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

🌟 If you like this project, don't forget to give it a star! 🌟
