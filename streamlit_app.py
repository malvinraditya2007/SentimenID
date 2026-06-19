import streamlit as st
import joblib
import re
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

factory = StemmerFactory()
stemmer = factory.create_stemmer()
list_stopwords = set(stopwords.words('indonesian'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'https?://\s+|www\.\s+', '', text) 
    text = re.sub(r'@[^\s]+', '', text)               
    text = re.sub(r'#[^\s]+', '', text)               
    text = re.sub(r'[^a-zA-Z\s]', '', text)           
    text = re.sub(r'\s+', ' ', text).strip()          
    
    tokens = text.split()
    clean_tokens = []
    for token in tokens:
        if token not in list_stopwords:
            stemmed_token = stemmer.stem(token)
            if stemmed_token:
                clean_tokens.append(stemmed_token)
                
    return " ".join(clean_tokens)

@st.cache_resource
def load_resources():
    vectorizer = joblib.load('vectorizer_tfidf.pkl') 
    model_lr = joblib.load('model_lr.pkl')
    model_nb = joblib.load('model_nb.pkl')
    model_rf = joblib.load('model_rf.pkl')
    model_knn = joblib.load('model_knn.pkl')
    return vectorizer, model_lr, model_nb, model_rf, model_knn

st.set_page_config(page_title="Malvin | Hate Speech Detector", layout="centered")

try:
    tfidf, lr, nb, rf, knn = load_resources()
except Exception as e:
    st.error(f"Gagal memuat model/vectorizer. Pastikan semua file .pkl ada di folder proyek! Error: {e}")
    st.stop()

st.title("SentimenID")
st.write("Deteksi Indonesia Hate Speech & Abusive Language..")
st.markdown("---")

st.sidebar.title("SentimenID")
st.sidebar.header("Pengaturan Model")

selected_model = st.sidebar.selectbox(
    "Pilih Algoritma Model:",
    ["Logistic Regression", "Naive Bayes", "Random Forest", "K-Nearest Neighbors"]
)

user_input = st.text_area(
    "Masukkan teks Tweet / Kalimat yang ingin dianalisis:",
    placeholder="Ketik kalimat di sini..."
)

if st.button("Analisis Sentimen Teks"):
    if user_input.strip() == "":
        st.warning("Silakan masukkan teks terlebih dahulu!")
    else:
        with st.spinner("Sedang memproses teks dan melakukan prediksi..."):
            
            cleaned_text = preprocess_text(user_input)
            text_vectorized = tfidf.transform([cleaned_text])
            
            if selected_model == "Logistic Regression":
                prediction = lr.predict(text_vectorized)[0]
            elif selected_model == "Naive Bayes":
                prediction = nb.predict(text_vectorized)[0]
            elif selected_model == "Random Forest":
                prediction = rf.predict(text_vectorized)[0]
            elif selected_model == "K-Nearest Neighbors":
                prediction = knn.predict(text_vectorized)[0]
            
            st.subheader("Hasil Preprocessing:")
            st.info(f"**Teks Bersih (Setelah Stemming):** {cleaned_text if cleaned_text else '[Teks menjadi kosong setelah filter stopword]'}")
            
            st.subheader("Hasil Prediksi Model:")
            
            if prediction == 1 or prediction == "Hate Speech" or prediction == "Abusive":
                st.error(f"**Terdeteksi:** Kalimat ini mengandung **Hate Speech / Abusive Language** (Berdasarkan model {selected_model})")
            else:
                st.success(f"**Aman:** Kalimat ini dikategorikan **Normal / Netral** (Berdasarkan model {selected_model})")