import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title(" EDA & Data Overview")

df = pd.read_csv("insurance.csv")
st.subheader("Raw Dataset")
st.dataframe(df.head())

st.subheader("Summary Statistics")
st.write(df.describe())

st.subheader("Smoker Distribution by Charges")
fig1 = sns.boxplot(x='smoker', y='charges', data=df)
st.pyplot(fig1.figure)

st.subheader("Pairplot by Smoker")
sns_plot = sns.pairplot(df, hue="smoker")
st.pyplot(sns_plot.figure)

st.subheader("Correlation Heatmap")
fig2, ax = plt.subplots()
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig2)
