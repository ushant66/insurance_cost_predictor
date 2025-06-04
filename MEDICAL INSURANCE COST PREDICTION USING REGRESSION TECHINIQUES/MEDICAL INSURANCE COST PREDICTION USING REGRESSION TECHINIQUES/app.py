import streamlit as st

st.title("Medical Insurance Cost Prediction")

st.markdown("""
Welcome to the **Medical Insurance Cost Prediction** app.

This project demonstrates the application of multiple machine learning algorithms to predict medical insurance charges based on:

- Age
- Sex
- BMI
- Number of Children
- Smoking Status
- Region

---

###  Factors Affecting Insurance Premiums

Many factors that affect how much you pay for health insurance are not within your control. Nonetheless, it's good to have an understanding of what they are. Here are some factors that affect how much health insurance premiums cost:

- **age**: Age of primary beneficiary.
- **sex**: Insurance contractor gender, female or male.
- **bmi**: Body Mass Index (kg/m²), ideally 18.5 to 24.9.
- **children**: Number of children covered by health insurance / dependents.
- **smoker**: Smoking status (yes/no).
- **region**: Beneficiary's residential area in the US: northeast, southeast, southwest, northwest.
""")
st.image("https://www.verywellhealth.com/thmb/2b1d3a4f8c6e0f5c7b9c5d8e2f0a4b3c.jpg", caption="Factors Affecting Insurance Premiums")            
