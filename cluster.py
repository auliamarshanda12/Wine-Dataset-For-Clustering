import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, silhouette_samples
import pandas as pd
import numpy as np

# Fungsi untuk menjalankan KMeans clustering dan menampilkan hasil
def run_kmeans(n_clusters, X):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, labels)
    
    # Menampilkan dataset asli dengan label cluster yang ditetapkan
    original_data = pd.DataFrame(X, columns=['Ash', 'Ash_Alcanity'])
    original_data['Cluster'] = labels
    st.subheader("Dataset Asli dengan Label Cluster:")
    st.write(original_data)

    # Menampilkan skor siluet
    st.subheader(f"Skor Siluet: {silhouette_avg:.2f}")

    # Menampilkan elbow plot
    inertias = []
    for k in range(1, n_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, n_clusters + 1), inertias, marker='o')
    ax.set_title('Elbow Plot untuk KMeans Clustering')
    ax.set_xlabel('Jumlah Cluster')
    ax.set_ylabel('Inertia')

    st.pyplot(fig)

# Aplikasi Streamlit
def main():
    st.title("Hasil KMeans Clustering")

    # Menghasilkan dataset contoh
    data = {
        'Ash': [2.43, 2.14, 2.67, 2.5, 2.87, 2.45, 2.45, 2.61, 2.17, 2.27],
        'Ash_Alcanity': [15.6, 11.2, 18.6, 16.8, 21, 15.2, 14.6, 17.6, 14, 16],
    }
    X = pd.DataFrame(data)

    # Sidebar
    st.sidebar.header("Pengaturan")
    n_clusters = st.sidebar.slider("Pilih Jumlah Maksimal Cluster", 2, 10, value=4)

    # Menjalankan KMeans dan menampilkan hasil
    run_kmeans(n_clusters, X)

if __name__ == "__main__":
    main()
