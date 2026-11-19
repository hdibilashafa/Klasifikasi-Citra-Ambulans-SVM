# Klasifikasi Citra Ambulans dengan SVM dan Fitur HSV, HOG, LBP

![Python](https://img.shields.io/badge/Python-3-blue)
![Google Colab](https://img.shields.io/badge/Google-Colab-F9AB00)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8)
![scikit-learn](https://img.shields.io/badge/scikit--learn-SVM_RBF-F7931E)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hdibilashafa/Klasifikasi-Citra-Ambulans-SVM/blob/main/klasifikasi_ambulans_svm.ipynb)

Klasifikasi gambar kendaraan menjadi **ambulans** atau **bukan ambulans** menggunakan **Support Vector Machine (kernel RBF)**. Setiap gambar diubah menjadi 8.638 fitur yang menggabungkan **warna (HSV)**, **bentuk dan tepi (HOG)**, dan **tekstur (LBP)**.

**Hasil: akurasi 93,47%** pada 950 gambar uji.

---

## Dataset

| Kelas | Jumlah gambar | Proporsi |
|---|---|---|
| Ambulance | 2.121 | 44,7% |
| Non-Ambulance | 2.628 | 55,3% |
| **Total** | **4.749** | |

Dataset berupa `ambulance.zip` (± 93 MB) yang **diunduh otomatis** dari Google Drive lewat `gdown` pada sel "Download Dataset". Isinya diekstrak ke `dataset/carsdataset/` dengan dua subfolder: `ambulance` dan `noambulance`.

---

## Pipeline

```
Gambar → Resize 128×128 → Ekstraksi fitur (HSV + HOG + LBP) → Gabung 8.638 fitur
       → StandardScaler → SVM (RBF) → Ambulance / Non-Ambulance
```

### Ekstraksi fitur

| Fitur | Metode | Jumlah fitur |
|---|---|---|
| Warna | Histogram HSV 3 dimensi (8×8×8 bin), dinormalisasi | 512 |
| Bentuk dan tepi | HOG (9 orientasi, 8×8 piksel per sel, blok 2×2, `L2-Hys`) | 8.100 |
| Tekstur | LBP uniform (P=24, R=3), histogram ternormalisasi | 26 |
| **Total** | | **8.638** |

### Pelatihan model

| Tahap | Detail |
|---|---|
| Split data | 80% latih (3.799) dan 20% uji (950), stratified, `random_state=42` |
| Normalisasi | `StandardScaler`, di-fit hanya pada data latih |
| Model | `SVC(kernel='rbf', C=10, gamma='scale')` |

---

## Hasil Evaluasi

**Akurasi: 93,47%** (950 data uji)

| Kelas | Precision | Recall | F1-score | Jumlah data |
|---|---|---|---|---|
| Ambulance | 0,94 | 0,91 | 0,93 | 424 |
| Non-Ambulance | 0,93 | 0,96 | 0,94 | 526 |
| Macro avg | 0,94 | 0,93 | 0,93 | 950 |
| Weighted avg | 0,93 | 0,93 | 0,93 | 950 |

Notebook juga menyediakan fungsi **prediksi gambar baru**: upload foto, lalu model menampilkan gambar beserta label prediksinya.

---

## Cara Menjalankan

Notebook ini dibuat untuk **Google Colab** (memakai `google.colab` dan `gdown`).

1. Klik badge **Open In Colab** di atas, atau buka file `.ipynb` lewat Colab
2. Klik **Runtime → Run all**. Dataset akan diunduh otomatis (butuh koneksi internet)
3. Di sel paling akhir, klik **Choose Files** dan pilih gambar yang ingin diprediksi

> Untuk menjalankan di komputer sendiri, unduh dataset secara manual, sesuaikan path `ambulance.zip` pada sel ekstraksi, dan hapus `from google.colab import files` beserta sel upload gambar.

---

## Struktur Repository

```
.
├── klasifikasi_ambulans_svm.ipynb
└── README.md
```

---

## Tech Stack

Python · Google Colab · OpenCV · scikit-image (HOG, LBP) · scikit-learn (SVM, StandardScaler) · NumPy · Matplotlib · gdown

---
