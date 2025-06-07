import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load Dataset
df = pd.read_csv("dataset/customers.csv")

# Data Preprocessing
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.iloc[:, 1:])

# Train K-Means Model
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)

# Streamlit UI
st.title("Market  Segmentation using K-Means")
st.subheader("👥 Group Members")
st.markdown("""
- **Yash Bhosale** | 🆔 CAN ID: `CAN_33900360  
- **Dhanashree Redekar** | 🆔 CAN ID: `CAN_33326816`  
- **Saeem M Desai** | 🆔 CAN ID: `CAN_33302689`  
- **Md Furkan** | 🆔 CAN ID: `CAN_34003833`
""")


# Display Data
if st.checkbox("Show Data"):
    st.write(df.head())

# Cluster Visualization
fig, ax = plt.subplots()
sns.scatterplot(x=df.iloc[:, 1], y=df.iloc[:, 2], hue=df['Cluster'], palette='viridis', ax=ax)
st.pyplot(fig)

# User Input for Prediction
st.sidebar.header("Predict Customer Segment")
income = st.sidebar.slider("Annual Income", int(df.iloc[:, 1].min()), int(df.iloc[:, 1].max()))
spending = st.sidebar.slider("Spending Score", int(df.iloc[:, 2].min()), int(df.iloc[:, 2].max()))

if st.sidebar.button("Predict"):
    input_data = scaler.transform([[income, spending]])
    cluster = kmeans.predict(input_data)[0]
    st.sidebar.success(f"Predicted Segment: {cluster}")
