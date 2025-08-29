import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from IPython.display import FileLink


train = pd.read_csv("train_v9rqX0R.csv")
test = pd.read_csv("test_AbJTz2l.csv")
print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()


train['Item_Fat_Content'] = train['Item_Fat_Content'].replace({
    'LF':'Low Fat', 'low fat':'Low Fat', 'reg':'Regular'
})
test['Item_Fat_Content'] = test['Item_Fat_Content'].replace({
    'LF':'Low Fat', 'low fat':'Low Fat', 'reg':'Regular'
})
item_weight_mean = train.groupby('Item_Identifier')['Item_Weight'].mean()
train['Item_Weight'] = train.apply(lambda row: item_weight_mean[row['Item_Identifier']] 
                                   if pd.isnull(row['Item_Weight']) else row['Item_Weight'], axis=1)
test['Item_Weight'] = test.apply(lambda row: item_weight_mean.get(row['Item_Identifier'], np.nan) 
                                 if pd.isnull(row['Item_Weight']) else row['Item_Weight'], axis=1)
outlet_size_mode = train.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=lambda x: x.mode()[0])
for df in [train, test]:
    df['Outlet_Size'] = df.apply(
        lambda row: outlet_size_mode[row['Outlet_Type']].iloc[0] if pd.isnull(row['Outlet_Size']) else row['Outlet_Size'],
        axis=1
    )



for df in [train, test]:
    df['Item_Category'] = df['Item_Identifier'].apply(lambda x: x[:2])
for df in [train, test]:
    df['Outlet_Age'] = 2013 - df['Outlet_Establishment_Year']
le = LabelEncoder()
cat_cols = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type',
            'Outlet_Type', 'Item_Category', 'Outlet_Identifier']
for col in cat_cols:
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])


missing_train = train.isnull().sum()
missing_train = missing_train[missing_train > 0]
missing_train


plt.figure(figsize=(8,5))
sns.histplot(train['Item_Outlet_Sales'], bins=30, kde=True)
plt.title("Distribution of Sales")
plt.show()


target = 'Item_Outlet_Sales'
features = [col for col in train.columns if col not in [target, 'Item_Identifier']]
X = train[features]
y = train[target]
X_test = test[features]


from sklearn.preprocessing import LabelEncoder
X = train.drop(columns=['Item_Outlet_Sales']).copy()
y = train['Item_Outlet_Sales']
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
if X.isnull().sum().any():
    X = X.fillna(0)
model = RandomForestRegressor(
    n_estimators=600,
    max_depth=20,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = -1 * cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=kf)
print("CV RMSE scores:", cv_scores)
print("Mean CV RMSE:", np.mean(cv_scores))


train = pd.read_csv("train_v9rqX0R.csv")
test = pd.read_csv("test_AbJTz2l.csv")
X = train.drop(columns=['Item_Outlet_Sales']).copy()
y = train['Item_Outlet_Sales']
le_dict = {}  
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    le_dict[col] = le  
if X.isnull().sum().any():
    X = X.fillna(0)
model = RandomForestRegressor(
    n_estimators=600,
    max_depth=20,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
model.fit(X, y)
X_test = test.copy()
for col in X_test.select_dtypes(include=['object']).columns:
    if col in le_dict:
        X_test[col] = le_dict[col].transform(X_test[col].astype(str))
    else:
        X_test[col] = LabelEncoder().fit_transform(X_test[col].astype(str))  # fallback
if X_test.isnull().sum().any():
    X_test = X_test.fillna(0)
test_pred = model.predict(X_test)
submission = pd.DataFrame({
    'Item_Identifier': test['Item_Identifier'],
    'Outlet_Identifier': test['Outlet_Identifier'],
    'Item_Outlet_Sales': test_pred
})
submission.to_csv("BigMart_Submission.csv", index=False)
print("Submission CSV created successfully!")
# Create a download link for the CSV in Jupyter Notebook
FileLink("BigMart_Submission.csv")


importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(data=importances.head(20), x='Importance', y='Feature')
plt.title("Top 20 Feature Importances")
plt.show()
