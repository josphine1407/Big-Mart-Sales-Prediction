# Big-Mart-Sales-Prediction

# Overview
This project focuses on predicting the sales of products across different Big Mart outlets using machine learning. By analyzing historical sales data and various product/outlet attributes, the goal is to build a model that helps retailers optimize inventory, pricing strategies, and demand forecasting.

# Dataset

Source: [Kaggle – Big Mart Sales Prediction Dataset](https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data)


# Files:

Train.csv – Training dataset with features and sales values.

Test.csv – Test dataset without sales values (to be predicted).

# Key Features:

Item_Identifier – Unique product ID

Item_Weight – Weight of the product

Item_Fat_Content – Categorical (Low Fat, Regular, etc.)

Item_Visibility – % of total display area given to the product

Item_Type – Category of the product

Item_MRP – Maximum Retail Price

Outlet_Identifier – Unique store ID

Outlet_Establishment_Year – Year of establishment

Outlet_Size, Outlet_Type, Outlet_Location_Type – Store attributes

Item_Outlet_Sales – Target variable (only in train set)

# Project Workflow

 **1.Exploratory Data Analysis (EDA)**

Distribution plots, missing value analysis, categorical vs. numerical comparisons.

**2.Data Preprocessing**

Handling missing values (Item_Weight, Outlet_Size)

Encoding categorical features (Label Encoding / One-Hot Encoding)

Feature engineering (e.g., Years_Since_Establishment, cleaning Item_Fat_Content)

**3.Model Building**

Linear Regression

Decision Tree Regressor

Random Forest Regressor

XGBoost Regressor

**4.Model Evaluation**

Metrics: RMSE, MAE, R² Score

Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)

**5.Final Predictions**

Generate BigMart_Submission.csv for test dataset.

# Setup
# Clone the repository
git clone https://github.com/josphine1407/Big-Mart-Sales-Prediction.git
cd Big-Mart-Sales-Prediction

# Create a virtual environment (recommended)
python -m venv env
source env/bin/activate   # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# File Structure
```
Big-Mart-Sales-Prediction/
│── BigMart Sales Prediction Project.ipynb   # Jupyter notebook
│── BigMart Sales Prediction Project.py      # Python script
│── BigMart_Submission.csv                   # Output predictions
│── requirements.txt                         # Dependencies
│── README.md                                # Project documentation
└── Datasets/
    ├── Train.csv
    └── Test.csv
