# streamlit_app.py

import streamlit as st
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import load_iris
import pandas as pd
import plotly.express as px

# Load dataset (mock for now)
data = load_iris()
X = data.data
features = data.feature_names
df = pd.DataFrame(X, columns=features)

# Title
st.title("🔬 Molecule Similarity Explorer (Quantum vs Classical)")

# Dropdown to choose method
method = st.selectbox("Choose dimensionality reduction method:", ["Classical PCA", "Quantum Kernel PCA (simulated)"])

# Process data
if method == "Classical PCA":
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X)
else:
    kpca = KernelPCA(kernel='rbf', n_components=2)  # Simulates quantum kernel
    reduced = kpca.fit_transform(X)

# Convert to DataFrame
reduced_df = pd.DataFrame(reduced, columns=['Component 1', 'Component 2'])
reduced_df['Target'] = data.target

# Plot
fig = px.scatter(reduced_df, x='Component 1', y='Component 2', color=reduced_df['Target'].astype(str),
                 title=f"2D Projection using {method}")

st.plotly_chart(fig)