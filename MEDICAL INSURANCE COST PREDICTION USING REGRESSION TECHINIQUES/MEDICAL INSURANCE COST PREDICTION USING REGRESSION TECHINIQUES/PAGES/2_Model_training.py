import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

st.title(" Model Training & Evaluation")

df = pd.read_csv("insurance.csv")
X = df.drop("charges", axis=1)
y = df["charges"]

cat_features = ["sex", "smoker", "region"]
num_features = ["age", "bmi", "children"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(drop='first'), cat_features)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model_results = {}

# Linear Regression
lr_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])
lr_pipe.fit(X_train, y_train)
y_pred_lr = lr_pipe.predict(X_test)
lr_r2 = r2_score(y_test, y_pred_lr)
model_results["Linear Regression"] = lr_r2

st.subheader("Linear Regression Performance")
st.write("R2 Score:", lr_r2)

# Decision Tree
dt_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DecisionTreeRegressor())
])
dt_param_grid = {
    'model__max_depth': [3, 5, 10, None],
    'model__min_samples_split': [2, 5, 10]
}
dt_grid = GridSearchCV(dt_pipe, dt_param_grid, cv=5, scoring='r2')
dt_grid.fit(X_train, y_train)
y_pred_dt = dt_grid.predict(X_test)
dt_r2 = r2_score(y_test, y_pred_dt)
model_results["Decision Tree Regressor"] = dt_r2

st.subheader("Decision Tree Regressor Performance")
st.write("Best Parameters:", dt_grid.best_params_)
st.write("R2 Score:", dt_r2)

# KNN
knn_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", KNeighborsRegressor())
])
knn_param_grid = {
    'model__n_neighbors': [3, 5, 7, 9],
    'model__weights': ['uniform', 'distance']
}
knn_grid = GridSearchCV(knn_pipe, knn_param_grid, cv=5, scoring='r2')
knn_grid.fit(X_train, y_train)
y_pred_knn = knn_grid.predict(X_test)
knn_r2 = r2_score(y_test, y_pred_knn)
model_results["KNN Regressor"] = knn_r2

st.subheader("KNN Regressor Performance")
st.write("Best Parameters:", knn_grid.best_params_)
st.write("R2 Score:", knn_r2)

# Save to session state
if "model_results" not in st.session_state:
    st.session_state["model_results"] = {}
st.session_state["model_results"].update(model_results)
