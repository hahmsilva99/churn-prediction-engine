import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics import accuracy_score

# 1. Data loading
print("Data loading now...")
df = pd.read_excel('churn_data.xlsx')

# 2. Data Cleaning & Preprocessing
# Filling the blanking spaces in Total Charges using  ' 0 '  
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce').fillna(0)

# Remove unneccessary columns leaking data that columns and Target 
columns_to_drop = ['CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code', 'Lat Long', 
                   'Churn Label', 'Churn Score', 'CLTV', 'Churn Reason']
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Transforming the Text data into numbers 
print("Data cleaning now ...")
for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])

# Seperating Features (X) and Target (y - Churn Value)
X = df.drop('Churn Value', axis=1)
y = df['Churn Value']

# Divide the data as Train and Test  data  (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Balancing the imbalanced data thruogh SMOTE 
print("Balancing the imbalanced data thruogh SMOTE...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# 4. Training XGBoost Model
print("XGBoost model is training now...")
model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
model.fit(X_train_smote, y_train_smote)

# Checking Model Accuracy 
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy : {accuracy * 100:.2f}%")

# 5. Saving the Model 
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
print("Model is saving as 'model.pkl' , Task is success!")