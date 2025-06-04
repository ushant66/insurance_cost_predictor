import streamlit as st
import pandas as pd

st.title(" Model Comparison Table")

if "model_results" in st.session_state:
    result_df = pd.DataFrame.from_dict(st.session_state["model_results"], orient='index', columns=["R2 Score"])
    result_df = result_df.sort_values(by="R2 Score", ascending=False).reset_index()
    result_df.columns = ["Model", "R2 Score"]
    st.dataframe(result_df)
else:
    st.warning("Please train the models from the 'Model Training & Evaluation' page first.")
