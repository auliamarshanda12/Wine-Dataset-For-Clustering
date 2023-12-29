import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
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
    original_data = pd.DataFrame(X, columns=['Alcohol', 'Magnesium'])
    original_data['Cluster'] = labels
    st.subheader("Dataset Original dengan Label Cluster:")
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
        'Alcohol': [14.23, 13.2, 13.16, 14.37, 13.24, 14.2, 14.39, 14.06, 14.83, 13.86],
        'Magnesium': [127, 100, 101, 113, 118, 112, 96, 121, 97, 98],
    }
    X = pd.DataFrame(data)
    X1 = pd.read_csv('wine-clustering.csv')
    # X2 = X1['Alcohol','Magnesium', 'Labels']
    
    chart_data = X1

    st.scatter_chart(
    chart_data,
    x='Alcohol',
    y='Magnesium',
    color='Labels' )
    
    st.scatter_chart(chart_data)
    

    # Sidebar
    st.sidebar.header("Pengaturan")
    n_clusters = st.sidebar.slider("Pilih Jumlah Maksimal Cluster", 2, 10, value=4)

    # Menjalankan KMeans dan menampilkan hasil
    run_kmeans(n_clusters, X)
    

if __name__ == "__main__":
    main()
