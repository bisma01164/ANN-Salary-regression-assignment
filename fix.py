import pandas as pd, pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

df = pd.read_csv("Churn_Modelling.csv")

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

ohe = OneHotEncoder(sparse_output=False)
geo_encoded = ohe.fit_transform(df[['Geography']])
geo_df = pd.DataFrame(geo_encoded, columns=ohe.get_feature_names_out(['Geography']))

X = pd.concat([
    df[['CreditScore','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember']],
    geo_df
], axis=1)

scaler = StandardScaler()
scaler.fit(X)  # <-- Ab 11 par fit hoga

with open("StandardScaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("Scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print(f"Ban gaya! Features: {scaler.n_features_in_}") # 11 aana chahiye