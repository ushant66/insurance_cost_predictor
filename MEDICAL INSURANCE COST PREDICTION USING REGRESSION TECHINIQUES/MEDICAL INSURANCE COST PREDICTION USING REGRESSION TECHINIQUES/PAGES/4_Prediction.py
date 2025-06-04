import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

st.title(" Predict Medical Insurance Cost")

df = pd.read_csv("insurance.csv")

st.markdown("### Enter the details below to predict insurance charges:")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.selectbox("Sex", ["female", "male"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, format="%.1f")
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

input_df = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker],
    "region": [region]
})

cat_features = ["sex", "smoker", "region"]
num_features = ["age", "bmi", "children"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(drop='first'), cat_features)
])

lr_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

dt_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DecisionTreeRegressor(random_state=42, max_depth=5, min_samples_split=2))
])

knn_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", KNeighborsRegressor(n_neighbors=5, weights='distance'))
])

X = df.drop("charges", axis=1)
y = df["charges"]
lr_model.fit(X, y)
dt_model.fit(X, y)
knn_model.fit(X, y)

if st.button("Predict Insurance Cost"):
    pred_lr = lr_model.predict(input_df)[0]
    pred_dt = dt_model.predict(input_df)[0]
    pred_knn = knn_model.predict(input_df)[0]

    st.subheader("Predicted Insurance Charges")
    st.write(f"Linear Regression: ${pred_lr:,.2f}")
    st.write(f"Decision Tree Regressor: ${pred_dt:,.2f}")
    st.write(f"KNN Regressor: ${pred_knn:,.2f}")
