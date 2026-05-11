import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics import accuracy_score

# 1. Data load kirima
print("Data load wenawa...")
df = pd.read_csv('churn_data.csv')

# 2. Data Cleaning & Preprocessing
# TotalCharges wala thiyena his than 0 walin purawanawa
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

# customerID eka model ekata wadak nathi nisa ain karanawa
df.drop('customerID', axis=1, inplace=True)

# Text data (Yes/No, Gender wage ewa) numbers walata harawanawa (Machine learning walata oni numbers nisa)
print("Data clean karanawa...")
for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])

# Features (X) saha Target (y - Churn eka) wen kirima
X = df.drop('Churn', axis=1)
y = df['Churn']

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

# Accuracy eka balamu
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy eka: {accuracy * 100:.2f}%")

# 5. Model eka Save kirima
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
print("Model eka 'model.pkl' widiyata save una! Wede success!")