import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, silhouette_samples
import pandas as pd
import numpy as np
import seaborn as sns

# Fungsi untuk menjalankan KMeans clustering dan menampilkan hasil
def run_kmeans(n_clusters, X1):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X1)
    silhouette_avg = silhouette_score(X1, labels)
    
    # Menampilkan dataset asli dengan label cluster yang ditetapkan
    original_data = pd.DataFrame(X1, columns=['Alcohol', 'Magnesium'])
    original_data['Cluster'] = labels
    st.subheader("Dataset Asli dengan Label Cluster:")
    st.write(original_data)

    # Menampilkan skor siluet
    st.subheader(f"Skor Siluet: {silhouette_avg:.2f}")

    # Menampilkan elbow plot
    inertias = []
    for k in range(1, n_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X1)
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
    X1 = pd.DataFrame(data)

    # Sidebar
    st.sidebar.header("Pengaturan")
    n_clusters = st.sidebar.slider("Pilih Jumlah Maksimal Cluster", 2, 10, value=4)

    def KMeans(n_clust = 5):
        kmeans = KMeans(n_clusters=n_clust).fit(X1)
        X1['Labels'] = kmeans.labels_

        plt.figure(figsize=(6, 4))
        
        sns.scatterplot(data=X1, x=X1['Alcohol'], y=X1['Magnesium'], hue=X1['Labels'], palette='bright', markers=True)
        
        plt.legend(title='Clusters', loc='upper right', bbox_to_anchor=(1.2, 1))

        plt.title('Distribusi dari Kluster Alcohol dan Magnesium')
        plt.xlabel('Alcohol')
        plt.ylabel('Magnesium')

        plt.show()
    
        st.header('Cluster Plot')
        st.pyplot()
        st.write(X1)


    # Menjalankan KMeans dan menampilkan hasil
    run_kmeans(n_clusters, X1)

if __name__ == "__main__":
    main()
