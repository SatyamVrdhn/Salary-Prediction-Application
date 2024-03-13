import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np

# Set page config
st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💰",
    layout="wide"
)

# Load data
data = pd.read_csv("data//Salary_Data.csv")

# Train linear regression model
x = np.array(data['YearsExperience']).reshape(-1, 1)
lr = LinearRegression()
lr.fit(x, np.array(data['salary']))

# Page styling
st.title("Salary Predictor")
st.sidebar.image("data//a.jpg", width=200)

# Sidebar navigation
nav = st.sidebar.radio("Navigation", ["Home", "Prediction", "Contribute"])

# Home page
if nav == "Home":
    st.image("data//a.jpg", width=1000)
    
    # Display table if checkbox is selected
    if st.checkbox("Show Table"):
        st.table(data)

    # Graph options
    graph = st.selectbox("What kind of graph?", ["Non-Interactive", "Interactive"])

    val=st.slider("Filter Data Using Years",0,20)
    data=data.loc[data["YearsExperience"]>=val]

    # Non-Interactive graph
    if graph == "Non-Interactive":
        plt.figure(figsize=(10, 5))
        plt.scatter(data["YearsExperience"], data["salary"])
        plt.ylim(0)
        plt.xlabel("Years Of Experience")
        plt.ylabel("Salary")
        plt.tight_layout()
        st.pyplot()
        st.set_option('deprecation.showPyplotGlobalUse', False)
    
    # Interactive graph
    if graph == "Interactive":
        layout = go.Layout(
            xaxis=dict(range=[0, 16]),
            yaxis=dict(range=[0, 2100000])
        )
        fig = go.Figure(data=go.Scatter(x=data["YearsExperience"], y=data["salary"], mode="markers"), layout=layout)
        st.plotly_chart(fig)

# Prediction page
elif nav == "Prediction":
    st.header("Know your Salary")
    val = st.number_input("Enter your experience", 0.00, 20.00, step=0.25)
    val = np.array(val).reshape(1, -1)
    pred = lr.predict(val)[0]

    # Display prediction
    if st.button("Predict"):
        st.success(f"Your predicted salary is {round(pred)}")
        st.balloons()

# Contribute page
elif nav == "Contribute":
    st.header("Contribute to our dataset")
    ex = st.number_input("Enter your Experience", 0.0, 20.0)
    sal = st.number_input("Enter your Salary", 0.00, 1000000.00, step=1000.0)

    # Submit button
    if st.button("Submit"):
        to_add = {"YearsExperience": [ex], "Salary": [sal]}
        to_add = pd.DataFrame(to_add)
        to_add.to_csv("data//Salary_Data.csv", mode='a', header=False, index=False)
        st.success("Submitted. Thank You!")
        st.balloons()
