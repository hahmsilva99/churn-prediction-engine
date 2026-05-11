import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics import accuracy_score

# 1. Data load kirima
print("Data load wenawa...")
df = pd.read_excel('churn_data.xlsx')

# 2. Data Cleaning & Preprocessing
# Total Charges wala his than 0 walin purawanawa (Nama wenas wela thibbe 'Total Charges' kiyala)
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce').fillna(0)

# Wedak nathi columns saha Target eka leak wena columns ain karanawa
columns_to_drop = ['CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code', 'Lat Long', 
                   'Churn Label', 'Churn Score', 'CLTV', 'Churn Reason']
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Text data numbers walata harawanawa
print("Data clean karanawa...")
for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])

# Features (X) saha Target (y - Churn Value eka) wen kirima
X = df.drop('Churn Value', axis=1)
y = df['Churn Value']

# Train saha Test widiyata data bedima (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. SMOTE walin imbalanced data balance kirima
print("SMOTE walin data balance karanawa...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# 4. XGBoost Model eka Train kirima
print("XGBoost model eka train wenawa...")
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