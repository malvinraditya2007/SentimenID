import streamlit as st
import joblib
import re
import nltk
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords

# --- KONFIGURASI NLTK & SASTRAWI ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

factory = StemmerFactory()
stemmer = factory.create_stemmer()
list_stopwords = set(stopwords.words('indonesian'))

# --- MEMUAT KAMUS ALAY ---
@st.cache_resource
def load_slang_dict():
    try:
        slang_df = pd.read_csv('Dataset/new_kamusalay.csv', encoding='latin-1', header=None)
        return dict(zip(slang_df[0], slang_df[1]))
    except Exception as e:
        st.warning(f"Kamus alay tidak ditemukan. Preprocessing akan berjalan tanpa kamus slang. Error: {e}")
        return {}

SLANG_DICT = load_slang_dict()

# --- FUNGSI PREPROCESSING ---
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'\\x[a-f0-9]{2}', ' ', text)
    text = re.sub(r'\b(user|url|rt)\b', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text) 
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    text = ' '.join([SLANG_DICT.get(w, w) for w in text.split()])
    text = ' '.join([w for w in text.split() if w not in list_stopwords])
    text = stemmer.stem(text)
    text = ' '.join([w for w in text.split() if len(w) > 2])
    return text

# --- MEMUAT MODEL DAN VECTORIZER (.PKL) ---
@st.cache_resource
def load_resources():
    vectorizer_tfidf = joblib.load('vectorizer_tfidf.pkl') 
    vectorizer_count = joblib.load('vectorizer_count.pkl') 
    model_lr = joblib.load('model_lr.pkl')
    model_nb = joblib.load('model_nb.pkl')
    model_rf = joblib.load('model_rf.pkl')
    model_knn = joblib.load('model_knn.pkl')
    return vectorizer_tfidf, vectorizer_count, model_lr, model_nb, model_rf, model_knn

# --- TAMPILAN UTAMA STREAMLIT ---
st.set_page_config(page_title="SentimenID | Detektor Ujaran Kebencian", layout="centered")

try:
    tfidf, cvect, lr, nb, rf, knn = load_resources()
except Exception as e:
    st.error(f"Gagal memuat resource model! Pastikan semua file `.pkl` berada di direktori yang sama. Error: {e}")
    st.stop()

st.title("SentimenID 🛡️")
st.write("Aplikasi Deteksi Multi-Class untuk Indonesia Hate Speech & Abusive Language.")
st.markdown("---")

# --- SIDEBAR PENGATURAN ---
st.sidebar.title("SentimenID")
st.sidebar.header("Pengaturan Model")
selected_model = st.sidebar.selectbox(
    "Pilih Algoritma Model:",
    ["Logistic Regression", "Naive Bayes", "Random Forest", "K-Nearest Neighbors"]
)

st.sidebar.markdown("""
---
**Keterangan Target Label:**
1. **Neutral**: Teks aman/normal.
2. **Abusive**: Mengandung kata kasar/kotor namun tidak menyerang kelompok SARA.
3. **Hate Speech**: Mengandung ujaran kebencian terhadap individu/SARA.
""")

# --- AREA INPUT ---
user_input = st.text_area(
    "Masukkan teks Tweet / Kalimat yang ingin dianalisis:",
    placeholder="Ketik kalimat di sini..."
)

# --- LOGIKA PREDIKSI ---
if st.button("Analisis Teks"):
    if user_input.strip() == "":
        st.warning("Silakan masukkan teks terlebih dahulu!")
    else:
        with st.spinner("Sedang melakukan preprocessing dan prediksi..."):
            
            cleaned_text = preprocess_text(user_input)
            
            if selected_model == "Naive Bayes":
                text_vectorized = cvect.transform([cleaned_text])
                prediction = nb.predict(text_vectorized)[0]
            else:
                text_vectorized = tfidf.transform([cleaned_text])
                if selected_model == "Logistic Regression":
                    prediction = lr.predict(text_vectorized)[0]
                elif selected_model == "Random Forest":
                    prediction = rf.predict(text_vectorized)[0]
                elif selected_model == "K-Nearest Neighbors":
                    prediction = knn.predict(text_vectorized)[0]
            
            st.subheader("Hasil Preprocessing:")
            st.info(f"**Teks Bersih:** {cleaned_text if cleaned_text else '[Teks menjadi kosong setelah penyaringan kata]'}")
            
            st.subheader("Hasil Prediksi Model:")
            
            if prediction == "Hate Speech":
                st.error(f"🤬 **Terdeteksi:** Kalimat ini mengandung **Hate Speech** (Berdasarkan model {selected_model})")
            elif prediction == "Abusive":
                st.warning(f"🚨 **Terdeteksi:** Kalimat ini mengandung **Abusive Language / Kata Kasar** (Berdasarkan model {selected_model})")
            else:
                st.success(f"✅ **Aman:** Kalimat ini dikategorikan **Neutral / Normal** (Berdasarkan model {selected_model})")