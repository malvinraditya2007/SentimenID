# SentimenID - Analisis Komparatif Model Machine Learning untuk Deteksi Hate Speech dan Abusive Language

Aplikasi dan studi komparasi model Natural Language Processing (NLP) untuk mendeteksi **Ujaran Kebencian (Hate Speech)** dan **Bahasa Kasar (Abusive Language)** pada tweet/teks media sosial berbahasa Indonesia. Proyek ini membandingkan empat algoritma Machine Learning: **Logistic Regression**, **Random Forest**, **Multinomial Naive Bayes**, dan **K-Nearest Neighbors (KNN)** yang dilengkapi dengan antarmuka web interaktif berbasis **Streamlit**.

---

## Daftar Isi
- [Latar Belakang](#latar-belakang)
- [Fitur Utama](#fitur-utama)
- [Arsitektur dan Pipeline NLP](#arsitektur-dan-pipeline-nlp)
- [Dataset dan Definisi Label](#dataset-dan-definisi-label)
- [Hasil Evaluasi dan Komparasi Model](#hasil-evaluasi-dan-komparasi-model)
- [Struktur Proyek](#struktur-proyek)
- [Panduan Instalasi dan Penggunaan](#panduan-instalasi-dan-penggunaan)
- [Aplikasi Web Streamlit](#aplikasi-web-streamlit)
- [Sitasi dan Referensi](#sitasi-dan-referensi)

---

## Latar Belakang

Penyebaran ujaran kebencian (*hate speech*) dan kata-kata kasar (*abusive language*) di ruang publik digital (khususnya Twitter/X) dapat memicu konflik horizontal, perundungan siber (*cyberbullying*), serta polarisasi isu sosial dan politik. Mengidentifikasi nuansa bahasa Indonesia di media sosial memiliki tantangan tersendiri:
1. Penggunaan bahasa gaul (*slang* / kata alay).
2. Singkatan tidak baku dan kesalahan pengetikan (*typo*).
3. Perbedaan batasan antara kata makian tanpa kebencian target tertentu (*abusive*) dengan serangan berbasis identitas kelompok/SARA (*hate speech*).

Proyek ini menghadirkan analisis komparatif performa 4 model klasifikasi dengan teknik pembobotan teks **TF-IDF** dan **CountVectorizer (Bag of Words)**, serta menyediakan antarmuka web langsung untuk pengujian teks secara *real-time*.

---

## Fitur Utama

- **End-to-End Preprocessing**: Membersihkan mention, URL, hashtag, simbol non-alfabetik, normalisasi kamus alay (`new_kamusalay.csv`), stopword removal (NLTK), serta *stemming* dengan PySastrawi.
- **Ekstraksi Fitur N-Gram**: Implementasi *unigram* dan *bigram* (`ngram_range=(1,2)`) berbasis TF-IDF (*sublinear term frequency*) dan CountVectorizer menghasilkan lebih dari 78.000 fitur linguistik.
- **Hyperparameter Tuning**: Optimasi hyperparameter otomatis dengan `GridSearchCV` menggunakan `StratifiedKFold` (5-split) dengan metrik `f1_macro`.
- **Evaluasi Multi-Metrik**: Pengukuran komprehensif mencakup *Accuracy*, *Macro Precision*, *Macro Recall*, dan *Macro F1-Score*.
- **Web App Interaktif**: Aplikasi Streamlit siap pakai untuk menguji kalimat pengguna dengan model pilihan secara instan.

---

## Arsitektur dan Pipeline NLP

Alur pemrosesan data dirancang secara terstruktur mulai dari teks mentah hingga menghasilkan label prediksi akhir:

```plaintext
[Input Teks] ──> [Text Preprocessing] ──> [Ekstraksi Fitur] ──> [Model Machine Learning] ──> [Label Prediksi]
```

### Rincian Tahapan Pipeline:

| Tahap | Proses | Deskripsi & Contoh |
| :--- | :--- | :--- |
| **1. Input** | Teks Mentah / Tweet | Kalimat atau tweet berbahasa Indonesia yang akan diuji. |
| **2. Preprocessing** | Case Folding & Regex Cleaning | Mengubah ke huruf kecil (*lowercase*), menghapus tautan URL (`http/www`), mention (`@username`), hashtag (`#tag`), token sistem (`USER`, `URL`, `RT`), karakter heksadesimal (`\x..`), serta simbol tanda baca dan angka. |
| | Normalisasi Slang / Kata Alay | Mengonversi kata singkatan atau bahasa tidak baku menjadi kata formal menggunakan kamus `new_kamusalay.csv` (misal: *beud* &rarr; *banget*, *jgn* &rarr; *jangan*, *loe* &rarr; *kamu*). |
| | Stopword Removal | Menghapus kata-kata umum yang minim nilai sentimen/makna berdasarkan korpus stopword bahasa Indonesia dari NLTK. |
| | Lemmatization / Stemming | Mengubah kata berimbuhan menjadi bentuk kata dasar dengan algoritma Nazief & Adriani melalui library **PySastrawi**. |
| | Filtering Kata Pendek | Mengeliminasi token kata yang memiliki panjang $\le 2$ karakter setelah proses pembersihan. |
| **3. Ekstraksi Fitur** | CountVectorizer (Unigram & Bigram) | Mengonversi teks menjadi matriks frekuensi kemunculan token (`ngram_range=(1,2)`), khusus digunakan untuk model **Multinomial Naive Bayes**. |
| | TF-IDF Vectorizer (Unigram & Bigram) | Menghitung bobot kepentingan kata berdasarkan frekuensi dokumen (*sublinear TF*), digunakan untuk **Logistic Regression**, **Random Forest**, dan **KNN**. |
| **4. Klasifikasi Model** | Evaluasi Model Terbaik | Teks yang sudah menjadi vektor numerik diproses oleh model machine learning yang telah dioptimasi hyperparameternya via `GridSearchCV`. |
| **5. Output** | Prediksi Kategori | Mengeluarkan 1 dari 3 kategori: **Neutral**, **Abusive**, atau **Hate Speech**. |

---

## Dataset dan Definisi Label

Dataset yang digunakan bersumber dari penelitian deteksi ujaran kebencian di Twitter Indonesia (*Ibrohim & Budi, 2019*).

### Skema Anotasi & Pemetaan Label (Multi-Class):
Dataset mentah dipetakan menjadi 3 kelas utama:
- **`Hate Speech`**: Tweet yang mengandung kebencian/serangan terhadap individu, kelompok, ras, agama, gender, atau disabilitas (`HS == 1`).
- **`Abusive`**: Tweet yang mengandung kata kotor/makian/bahasa kasar, tetapi tidak ditargetkan sebagai ujaran kebencian (`Abusive == 1` & `HS == 0`).
- **`Neutral`**: Tweet normal dan aman yang tidak mengandung unsur ujaran kebencian maupun kata kasar (`HS == 0` & `Abusive == 0`).

### Distribusi Data:
| Label | Jumlah Tweet | Persentase |
| :--- | :---: | :---: |
| **Neutral** | 5.860 | 44.50% |
| **Hate Speech** | 5.561 | 42.23% |
| **Abusive** | 1.748 | 13.27% |
| **Total** | **13.169** | **100%** |

> *Pembagian data dilakukan secara **Stratified Split (80% Train : 20% Test)**: Data Latih = 10.535 sampel, Data Uji = 2.634 sampel.*

---

## Hasil Evaluasi dan Komparasi Model

Evaluasi dilakukan pada subset pengujian (*test set*) 2.634 tweet yang tidak pernah dilihat sebelumnya oleh model.

| Algoritma Model | Representasi Fitur | Parameter Terbaik | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | TF-IDF (1,2-gram) | `n_estimators=100`, `min_samples_split=5` | **0.7954** | 0.7900 | 0.7453 | 0.7627 |
| **Logistic Regression** | TF-IDF (1,2-gram) | `C=100`, `solver='lbfgs'` | 0.7935 | 0.7725 | **0.7562** | **0.7636** |
| **Multinomial Naive Bayes** | CountVectorizer (1,2-gram) | `alpha=1.0` | 0.7779 | **0.8045** | 0.7015 | 0.7290 |
| **K-Nearest Neighbors (KNN)** | TF-IDF (1,2-gram) | `n_neighbors=9`, `metric='cosine'` | 0.7198 | 0.6922 | 0.6831 | 0.6864 |

### Analisis dan Temuan:
1. **Random Forest** mencatatkan **akurasi tertinggi (79.54%)**, tangguh dalam menangkap kombinasi non-linear dari fitur teks n-gram.
2. **Logistic Regression** memberikan performa paling **seimbang dan stabil** pada seluruh kelas dengan **F1-Score Macro tertinggi (76.36%)** dan **Recall tertinggi (75.62%)**.
3. **Multinomial Naive Bayes** unggul dalam hal **Precision (80.45%)**, menjadikannya sangat selektif dan meminimalisir kesalahan *false positive* saat menandai teks sebagai ujaran kebencian.
4. **KNN** memiliki performa terendah (F1-score 68.64%) akibat fenomena *curse of dimensionality* pada ruang vektor teks sparse berdimensi 78.000+.

---

## Struktur Proyek

```plaintext
├── Dataset/
│   ├── abusive.csv             # Kamus leksikon kata-kata kasar
│   ├── citation.bib            # Format sitasi bibtex dataset
│   ├── data.csv                # Dataset utama tweet beranotasi
│   ├── new_kamusalay.csv       # Kamus normalisasi kata alay/slang
│   └── README.md               # Dokumentasi resmi dataset sumber
├── code.ipynb                  # Notebook lengkap (EDA, Preprocessing, Modeling, Evaluasi)
├── model_lr.pkl                # Serialized model Logistic Regression
├── model_nb.pkl                # Serialized model Naive Bayes
├── model_rf.pkl                # Serialized model Random Forest
├── model_knn.pkl               # Serialized model K-Nearest Neighbors
├── vectorizer_tfidf.pkl        # Serialized TF-IDF Vectorizer
├── vectorizer_count.pkl        # Serialized CountVectorizer
├── streamlit_app.py            # Script antarmuka aplikasi web Streamlit
├── requirements.txt            # Daftar pustaka dependensi Python
└── README.md                   # Dokumentasi proyek
```

---

## Panduan Instalasi dan Penggunaan

### 1. Prasyarat
- Python 3.8, 3.9, atau 3.10
- Git & pip

### 2. Clone Repositori
```bash
git clone https://github.com/username-kamu/NLP-Deteksi-Hate-Speech-dan-Abusive.git
cd "NLP-Deteksi-Hate-Speech-dan-Abusive"
```

### 3. Buat dan Aktifkan Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 4. Pasang Dependensi
```bash
pip install -r requirements.txt
```

### 5. Unduh Korpus NLTK (Jika Diperlukan)
Jalankan perintah singkat berikut di terminal untuk memastikan data NLTK terunduh:
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

---

## Aplikasi Web Streamlit

Untuk menjalankan antarmuka grafis interaktif deteksi teks:

```bash
streamlit run streamlit_app.py
```

Setelah server aktif, buka peramban (*browser*) di alamat `http://localhost:8501`.

### Cara Menggunakan Aplikasi:
1. **Pilih Algoritma Model** pada sidebar kiri (*Logistic Regression*, *Naive Bayes*, *Random Forest*, atau *K-Nearest Neighbors*).
2. **Ketik atau tempel teks** tweet / kalimat pada area input yang disediakan.
3. Klik tombol **"Analisis Teks"**.
4. Sistem akan menampilkan:
   - **Hasil Preprocessing**: Teks bersih setelah filtering, normalisasi slang, stopword, dan stemming.
   - **Status Deteksi**: Label prediksi akhir (Neutral, Abusive, atau Hate Speech).

---

## Sitasi dan Referensi

Dataset dan leksikon slang yang digunakan dalam proyek ini dipublikasikan oleh:

```bibtex
@inproceedings{ibrohim-budi-2019-multi,
    title = "Multi-label Hate Speech and Abusive Language Detection in {I}ndonesian {T}witter",
    author = "Ibrohim, Muhammad Okky  and
      Budi, Indra",
    booktitle = "Proceedings of the Third Workshop on Abusive Language Online",
    month = aug,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W19-3506",
    doi = "10.18653/v1/W19-3506",
    pages = "46--57"
}
```

---


  Dibuat untuk keperluan riset NLP, edukasi, dan eksplorasi Machine Learning berbahasa Indonesia.

