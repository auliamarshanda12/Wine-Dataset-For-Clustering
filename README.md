# Laporan Proyek Machine Learning

### Nama : Aulia Marshanda
### Nim : 211351034
### Kelas : Pagi B

## Domain Proyek

Wine Dataset For Clustering ini merupakan hasil analisis kimiawi anggur yang ditanam di wilayah yang sama di Italia tetapi berasal dari tiga kultivar berbeda. Analisis tersebut menentukan jumlah 13 konstituen yang ditemukan di masing-masing dari tiga jenis anggur.

## Business Understanding

Untuk orang-orang yang ingin mengetahui sekumpulan data analisis kimiawi anggur tanpa harus pergi ke italia terlebih dahulu

Bagian Laporan ini mencakup:

### Problem Statements

Ketidakmungkinan bagi seseorang untuk pergi ke italia langsung hanya untuk mencari tahu analisis kimiawi anggur

### Goals

- Membuat penelitian dalam studi pengelompokkan anggur

- Mengembangkan model clustering atau pengelompokkan anggur berdasarkan karakteristik  tertentu

### Solution Statements

- Mengembangkan model clustering atau pengelompokkan yang akurat berdasarkan atribut kandidat, yang mengintegrasikan data dari Kaggle.com untuk memberikan informasi terkait jumlah 13 konsituen yang ditemukan di masing-masing anggur
- Model yang dihasilkan dari dataset itu menggunakan algoritma K-Means (Pengelompokkan)
    

## Data Understanding

Dataset yang saya gunakan berasal dari Kaggle yang berisi Analisis yang  menentukan jumlah 13 konstituen yang ditemukan di masing-masing dari tiga jenis anggur. 

[Wine Dataset For Clustering]
https://www.kaggle.com/datasets/harrywang/wine-dataset-for-clustering

### Variabel-variabel pada Heart Failure Prediction Dataset adalah sebagai berikut:
- Alcohol: Persentase alkohol dalam anggur, yang dapat mempengaruhi rasa dan kekuatan anggur. [Tipe Data : int ]

- Malic acid: Kandungan asam malat dalam anggur, memberikan kontribusi terhadap rasa asam atau segar. [Tipe Data : int ]

- Ash: Jumlah abu dalam anggur setelah pembakaran, dapat memberikan indikasi terhadap kandungan mineral. [Tipe Data : int ]

- Alcalinity of ash: Kekuatan basa dari abu dalam anggur, dapat memengaruhi keseimbangan pH anggur. [Tipe Data : int ]

- Magnesium: Kandungan magnesium dalam anggur, yang dapat berperan dalam berbagai proses kimia dan metabolisme tanaman. [Tipe Data : int ]

- Total phenols: Jumlah total senyawa fenolik dalam anggur, dapat memberikan petunjuk tentang potensi kesehatan dan daya tahan anggur. [Tipe Data : int ]

- Flavanoids: Kelas senyawa fenolik tertentu dalam anggur, dapat memberikan kontribusi terhadap rasa dan aroma. [Tipe Data : int ]

- Nonflavanoid phenols: Jumlah senyawa fenolik selain flavanoid, dapat memberikan wawasan tambahan tentang komposisi fenolik anggur. [Tipe Data : int ]

- Proanthocyanins: Jenis senyawa fenolik dalam anggur, memiliki potensi antioksidan dan dapat memberikan warna kepada anggur. [Tipe Data : int ]

- Color intensity: Intensitas warna anggur, memberikan informasi tentang ketebalan dan konsentrasi warna anggur. [Tipe Data : int ]

- Hue: Warna tertentu dalam spektrum warna, dapat memberikan informasi tentang nuansa atau nuansa warna anggur. [Tipe Data : int ]

- OD280/OD315 of diluted wines: Rasio absorbansi cahaya pada dua panjang gelombang yang berbeda, dapat memberikan informasi tentang kejernihan atau kepekatan anggur. [Tipe Data : int ]

- Proline: Kandungan prolin dalam anggur, yang dapat menjadi indikator kualitas dan maturitas buah anggur. [Tipe Data : int ]


## Data Collection

Untuk data collection ini, saya mendapatkan dataset yang nantinya digunakan dari website kaggle dengan nama Wine Dataset For Clustering jika Anda tertarik dengan datasetnya, Anda bisa click link diatas.

## Data Discovery And Profilling

Untuk bagian ini, kita akan menggunakan teknik EDA.
Pertama kita mengimport semua library yang dibutuhkan,

## Import Library
``` bash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
```

## Data Discovery
``` bash
df = pd.read_csv('wine-clustering.csv')
df1 = pd.read_csv('wine-clustering.csv')
```
- Lalu tipe data dari masing-masing kolom, kita bisa menggunakan properti info,
``` bash
df.info()
```
- Selanjutnya kita akan mengonversi data dari float ke integer
``` bash
df.fillna(0, inplace=True)

cols_to_convert = [
    "Alcohol", "Malic_Acid", "Ash", "Ash_Alcanity",
    "Magnesium", "Total_Phenols", "Flavanoids",
    "Nonflavanoid_Phenols", "Proanthocyanins",
    "Color_Intensity", "Hue", "OD280",
    "Proline"
]

df[cols_to_convert] = df[cols_to_convert].astype(int)
```
- selanjutnya kita menghapus kolom dari dataframe
``` bash
X = df.drop(['OD280'], axis=1)
```
- Menyimpan dataframe kedalam csv
``` bash
df.to_csv('wine-clustering.csv', index=False)
```
- Selanjutnya kita mencetak atau menampilkan nilai dari variabel X
``` bash
print(X)
```
``` bash
X.head()
```
## EDA
``` bash
data = {
    'Alcohol': [14.23, 13.2, 13.16, 14.37, 13.24, 14.2, 14.39, 14.06, 14.83, 13.86],
    'Malic_Acid': [1.71, 1.78, 2.36, 1.95, 2.59, 1.76, 1.87, 2.15, 1.64, 1.35],
    'Ash': [2.43, 2.14, 2.67, 2.5, 2.87, 2.45, 2.45, 2.61, 2.17, 2.27],
    'Ash_Alcanity': [15.6, 11.2, 18.6, 16.8, 21, 15.2, 14.6, 17.6, 14, 16],
    'Magnesium': [127, 100, 101, 113, 118, 112, 96, 121, 97, 98],
    'Total_Phenols': [2.8, 2.65, 2.8, 3.85, 2.8, 3.27, 2.5, 2.6, 2.8, 2.98],
    'Flavanoids': [3.06, 2.76, 3.24, 3.49, 2.69, 3.39, 2.52, 2.51, 2.98, 3.15],
    'Nonflavanoid_Phenols': [0.28, 0.26, 0.3, 0.24, 0.39, 0.34, 0.3, 0.31, 0.29, 0.22],
    'Proanthocyanins': [2.29, 1.28, 2.81, 2.18, 1.82, 1.97, 1.98, 1.25, 1.98, 1.85],
    'Color_Intensity': [5.64, 4.38, 5.68, 7.8, 4.32, 6.75, 5.25, 5.05, 5.2, 7.22],
    'Hue': [1.04, 1.05, 1.03, 0.86, 1.04, 1.05, 1.02, 1.06, 1.08, 1.01],
    'OD280': [3.92, 3.4, 3.17, 3.45, 2.93, 2.85, 3.58, 3.58, 2.85, 3.55],
    'Proline': [1065, 1050, 1185, 1480, 735, 1450, 1290, 1295, 1045, 1045]
}

# Membuat dataframe
df = pd.DataFrame(data)

# Membuat heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

# Menambahkan judul
plt.title('Heatmap of Correlation Matrix')

# Menampilkan plot
plt.show()
```
![image](https://github.com/auliamarshanda12/Wine-Dataset-For-Clustering/assets/148952831/aa213b35-d1d0-4136-a071-e58d2ca84908)

``` bash
sns.barplot(x='Alcohol', y='Magnesium', data=df)
plt.show()
```
![image](https://github.com/auliamarshanda12/Wine-Dataset-For-Clustering/assets/148952831/51bf2ced-cf89-42b4-a662-4f4e7e1fda00)

``` bash
sns.jointplot(x='Alcohol', y='Magnesium', data=df, kind='scatter')
plt.show()
```
sns.jointplot(x='Alcohol', y='Magnesium', data=df, kind='scatter')
plt.show()
``` bash
sns.set(style="whitegrid")

# Data frame yang diambil (_df_6) dan kolom yang digunakan ('Alcohol' dan 'Magnesium')
x = df['Alcohol']
y = df['Magnesium']

# Membuat scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=x, y=y, s=32, alpha=0.8, color='orange')

# Menambahkan judul dan mengatur label sumbu
plt.title('Scatter Plot of Ash & Ash_Alcanity')
plt.xlabel('Alcohol')
plt.ylabel('Magnesium')

# Menghilangkan garis atas dan kanan pada plot
sns.despine(right=True, top=True)

# Menampilkan plot
plt.show()
```
![image](https://github.com/auliamarshanda12/Wine-Dataset-For-Clustering/assets/148952831/b36e30f2-d5c1-45d3-a12c-9320dc55270a)

``` bash
sns.set(style="whitegrid")

# Mengambil kolom 'Alcohol' dari DataFrame (df)
df = ['Alcohol']

# Membuat histogram plot
plt.figure(figsize=(10, 6))
sns.histplot(data, bins=20, kde=False, color='skyblue')

# Menambahkan judul dan mengatur label sumbu
plt.title('Distribusi Kandungan Alcohol')
plt.xlabel('Kandungan Alcohol')
plt.ylabel('Frekuensi')

# Menghilangkan garis atas dan kanan pada plot
sns.despine(right=True, top=True)

# Menampilkan plot
plt.show()
```
![image](https://github.com/auliamarshanda12/Wine-Dataset-For-Clustering/assets/148952831/dfa7f9ce-8283-4c48-bc97-218828980b9f)

## Prepocessing
``` bash
clusters = []
for i in range(1, 11):
    km = KMeans(n_clusters=i).fit(X)
    clusters.append(km.inertia_)

# Menemukan elbow point
diff = [clusters[i] - clusters[i + 1] for i in range(len(clusters)-1)]

# Mencari indeks dengan perubahan terbesar
elbow_index = diff.index(max(diff)) + 1

fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x=list(range(1, 11)), y=clusters, ax=ax)
ax.set_title('Mencari Elbow')
ax.set_xlabel('Clusters')
ax.set_ylabel('Inertia')

# Menandai elbow point pada grafik
ax.annotate('Possible elbow point',
            xy=(elbow_index, clusters[elbow_index]),
            xytext=(elbow_index+1, clusters[elbow_index] + 5000),
            arrowprops=dict(facecolor='red', arrowstyle='->'),
            bbox=dict(facecolor='red', edgecolor='black', boxstyle='round,pad=0.5'))

plt.show()
```

## Modeling

Model cluster dalam konteks algoritma K-Means adalah representasi hasil dari proses clustering. Ini mencakup pusat-pusat kluster dan label kluster untuk setiap titik data dalam dataset. Dengan kata lain, model cluster memberikan gambaran tentang bagaimana data terorganisir menjadi kelompok berdasarkan kesamaan atau jarak antara titik-titik data.
``` bash
# data = {
#     'Alcohol': [14.23, 13.20, 13.16, 14.37, 13.24],
#     'Malic_Acid': [1.71, 1.78, 2.36, 1.95, 2.59],
#     'Ash': [2.43, 2.14, 2.67, 2.50, 2.87],
#     'Ash_Alcanity': [15.6, 11.2, 18.6, 16.8, 21.0],
#     'Magnesium': [127, 100, 101, 113, 118],
#     'Total_Phenols': [2.80, 2.65, 2.80, 3.85, 2.80],
#     'Flavanoids': [3.06, 2.76, 3.24, 3.49, 2.69],
#     'Nonflavanoid_Phenols': [0.28, 0.26, 0.30, 0.24, 0.39],
#     'Proanthocyanins': [2.29, 1.28, 2.81, 2.18, 1.82],
#     'Color_Intensity': [5.64, 4.38, 5.68, 7.80, 4.32],
#     'Hue': [1.04, 1.05, 1.03, 0.86, 1.04],
#     'OD280': [3.92, 3.40, 3.17, 3.45, 2.93],
#     'Proline': [1065, 1050, 1185, 1400, 735],
# }

# Membuat DataFrame dari data
X1 = df1

# Mengisi nilai yang hilang dengan 0
# X.fillna(0, inplace=True)

# Menghilangkan kolom 'Color_Intensity' sebelum melakukan clustering
# X_clustering = X.drop('Alcohol', axis=1)

# Menentukan jumlah kluster
n_clust = 5

# Melakukan KMeans clustering
kmeans = KMeans(n_clusters=n_clust).fit(X1)

# Menambahkan kolom label kluster ke DataFrame
X1['Labels'] = kmeans.labels_

# Tampilkan DataFrame setelah penambahan label kluster
print(X1)
```
``` bash
X1.head()
```
``` bash
# data = {
     'Alcohol': [14.23, 13.20, 13.16, 14.37, 13.24],
     'Malic_Acid': [1.71, 1.78, 2.36, 1.95, 2.59],
     'Ash': [2.43, 2.14, 2.67, 2.50, 2.87],
     'Ash_Alcanity': [15.6, 11.2, 18.6, 16.8, 21.0],
     'Magnesium': [127, 100, 101, 113, 118],
     'Total_Phenols': [2.80, 2.65, 2.80, 3.85, 2.80],
     'Flavanoids': [3.06, 2.76, 3.24, 3.49, 2.69],
     'Nonflavanoid_Phenols': [0.28, 0.26, 0.30, 0.24, 0.39],
     'Proanthocyanins': [2.29, 1.28, 2.81, 2.18, 1.82],
     'Color_Intensity': [5.64, 4.38, 5.68, 7.80, 4.32],
     'Hue': [1.04, 1.05, 1.03, 0.86, 1.04],
     'OD280': [3.92, 3.40, 3.17, 3.45, 2.93],
     'Proline': [1065, 1050, 1185, 1400, 735],
 }

 # Membuat DataFrame dari data
 X = pd.DataFrame(data)

 # Mengisi nilai yang hilang dengan 0
 X.fillna(0, inplace=True)

 # Menghilangkan kolom 'Color_Intensity' sebelum melakukan clustering
 X_clustering = X.drop('Color_Intensity', axis=1)

 # Menentukan jumlah kluster
 n_clust = 5

 # Melakukan KMeans clustering
 kmeans = KMeans(n_clusters=n_clust).fit(X_clustering)

 # Menambahkan kolom label kluster ke DataFrame
 X['Labels'] = kmeans.labels_

 # Tampilkan DataFrame setelah penambahan label kluster
 print(X)
```
``` bash
plt.figure(figsize=(6, 4))

# Membuat scatter plot dengan warna berdasarkan label kluster
sns.scatterplot(data=X1, x="Alcohol", y="Ash", hue="Labels", palette='bright', markers=True)

# Menambahkan legenda
plt.legend(title='Clusters', loc='upper right', bbox_to_anchor=(1.2, 1))

# Menambahkan judul dan label sumbu
plt.title('Kluster berdasarkan Alcohol & Ash')
plt.xlabel('Alcohol')
plt.ylabel('Ash')

# Menampilkan plot
plt.show()
```
``` bash
plt.figure(figsize=(6, 4))

# Membuat scatter plot dengan warna berdasarkan label kluster
sns.scatterplot(data=X1, x=X1['Alcohol'], y=X1['Magnesium'], hue=X1['Labels'], palette='bright', markers=True)

# Menambahkan legenda
plt.legend(title='Clusters', loc='upper right', bbox_to_anchor=(1.2, 1))

# Menambahkan judul dan label sumbu
plt.title('Distribusi dari Kluster Ash dan Ash_Alcanity')
plt.xlabel('Alcohol')
plt.ylabel('Magnesium')

# Menampilkan plot
plt.show()
```
``` bash
# Menghilangkan kolom 'Hue' dan 'Labels' sebelum melakukan klastering
X1 = X1.drop(['Hue', 'Labels'], axis=1)

silhouette_scores = []

for num_clusters in range(2, min(10, len(X1))):
    kmeans = KMeans(n_clusters=num_clusters)
    labels = kmeans.fit_predict(X1)

    silhouette_avg = silhouette_score(X1, labels)
    silhouette_scores.append(silhouette_avg)

plt.figure(figsize=(8, 5))
plt.plot(range(2, min(10, len(X1))), silhouette_scores, marker='o')
plt.title('Silhouette Scores for Different Numbers of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()
## Visualisasi Hasil Algoritma
``` bash
   for k in range(2, min(10, len(X1) - 1)):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X1)
    silhouette_avg = silhouette_score(X1, labels)
    sample_silhouette_values = silhouette_samples(X1, labels)

    plt.figure(figsize=(6, 4))
    plt.title(f'KMeans Clustering dengan {k} Klaster\nSkor Siluet: {silhouette_avg:.2f}')

    y_lower = 10
    for i in range(k):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / k)
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)

        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    plt.xlabel("Nilai Koefisien Siluet")
    plt.ylabel("Label Klaster")
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")
    plt.yticks([])
    plt.show()
```
``` bash
for k in range(2, min(10, len(X1) - 1)):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X1)
    silhouette_avg = silhouette_score(X1, labels)
    sample_silhouette_values = silhouette_samples(X1, labels)

    plt.figure(figsize=(6, 4))
    plt.title(f'KMeans Clustering dengan {k} Klaster\nSkor Siluet: {silhouette_avg:.2f}')

    y_lower = 10
    for i in range(k):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / k)
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)

        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    plt.xlabel("Nilai Koefisien Siluet")
    plt.ylabel("Label Klaster")
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")
    plt.yticks([])
    plt.show()
```

## Deployment
https://github.com/auliamarshanda12/Wine-Dataset-For-Clustering/new/main?filename=README.md
