import streamlit as st
import pandas as pd
import joblib
import os
import re
import nltk
from nltk.stem.porter import PorterStemmer

# --- Page Config ---
st.set_page_config(
    page_title="Airline Sentiment Analysis",
    page_icon="✈️",
    layout="wide"
)

# --- 1. Tokenizer Fonksiyonu (Model Yükleme için ZORUNLU) ---
# Model bu fonksiyonla eğitildiği için, yüklerken bu fonksiyonun
# tanımlı olması gerekir.
stemmer = PorterStemmer()

def stemming_tokenizer(text):
    # Sadece harfleri al
    text = re.sub(r"[^a-zA-Z]", " ", str(text))
    # Kelimelere böl
    tokens = text.split()
    # Stemming uygula
    stems = [stemmer.stem(word) for word in tokens]
    return stems

# --- 2. Akıllı Yol Bulucu (Local vs Hugging Face) ---
def resolve_model_path(filename):
    """
    Model dosyasını scriptin bulunduğu konuma göre arar.
    En garantili yöntem budur.
    """
    # 1. Şu an çalışan dosyanın (streamlit_app.py) tam yolunu bul
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    possible_paths = [
        # Seçenek 1: Script ile aynı klasörde (Sizin durumunuz: src/best_model.pkl)
        os.path.join(current_script_dir, filename), 
        
        # Seçenek 2: Çalışma dizininde (Root'ta ise)
        filename,
        
        # Seçenek 3: Localde üst klasördeki models altında
        os.path.join(current_script_dir, "..", "models", filename),
        
        # Seçenek 4: Çalışma dizininde models/ altında
        os.path.join("models", filename),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

# --- 3. Modeli Yükle ---
@st.cache_resource
def load_model():
    # Model ismini buraya yazıyoruz
    model_filename = "best_model.pkl"
    model_path = resolve_model_path(model_filename)
    
    if model_path:
        try:
            model = joblib.load(model_path)
            return model
        except Exception as e:
            st.error(f"Model yüklenirken hata oluştu: {e}")
            return None
    else:
        st.error(f"Model dosyası ('{model_filename}') bulunamadı! Lütfen dosya yolunu kontrol edin.")
        return None

# --- Main App ---
def main():
    st.title("✈️ Airline Sentiment Classification")
    st.markdown("Bu uygulama, havayolu tweetlerinin duygu durumunu (Positive, Negative, Neutral) tahmin eder.")
    
    model = load_model()

    if model is None:
        st.warning("Model dosyası bulunamadığı için uygulama başlatılamadı.")
        st.info("İpucu: Eğer localde `src` klasöründeysen modelin `../models/` altında olduğundan emin ol.")
        return

    tab1, tab2 = st.tabs(["✍️ Manuel Giriş", "📂 Toplu Tahmin (CSV)"])

    # --- TAB 1: Manuel Giriş ---
    with tab1:
        st.subheader("Tekli Tahmin")
        user_input = st.text_area("Tweet Metni Girin:", "The flight was amazing and the crew was helpful!")
        
        if st.button("Analiz Et"):
            if user_input:
                try:
                    # Model pipeline olduğu için doğrudan metni veriyoruz
                    # (Tokenizer pipeline içinde çalışacak)
                    prediction = model.predict([user_input])[0]
                    
                    st.divider()
                    
                    # Renklendirme
                    if prediction == "positive":
                        st.success(f"Tahmin: {prediction.upper()} 😃")
                    elif prediction == "negative":
                        st.error(f"Tahmin: {prediction.upper()} 😡")
                    else:
                        st.warning(f"Tahmin: {prediction.upper()} 😐")
                    
                    # Olasılık değerleri (Eğer model destekliyorsa)
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba([user_input])[0]
                        classes = model.classes_
                        proba_df = pd.DataFrame([proba], columns=classes)
                        st.bar_chart(proba_df.T)
                        
                except Exception as e:
                    st.error(f"Tahmin Hatası: {e}")
            else:
                st.warning("Lütfen bir metin girin.")

    # --- TAB 2: Toplu İşlem ---
    with tab2:
        st.subheader("CSV ile Toplu Tahmin")
        uploaded_file = st.file_uploader("CSV Dosyası Yükle", type=["csv"])

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Yüklenen Veri (İlk 5 satır):")
                st.dataframe(df.head())
                
                # Text kolonunu otomatik bulmaya çalış veya kullanıcıya seçtir
                object_cols = list(df.select_dtypes(include=['object']).columns)
                text_col = st.selectbox("Analiz edilecek metin sütununu seçin:", object_cols)
                
                if st.button("Tahminleri Çalıştır"):
                    with st.spinner('Tahmin yapılıyor...'):
                        # Boş verileri doldur
                        input_data = df[text_col].fillna('')
                        
                        # Tahmin
                        df['sentiment_prediction'] = model.predict(input_data)
                        
                        st.success("Tamamlandı!")
                        st.dataframe(df[[text_col, 'sentiment_prediction']].head())
                        
                        # İndirme Butonu
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Sonuçları İndir (CSV)",
                            data=csv,
                            file_name="tahmin_sonuclari.csv",
                            mime="text/csv"
                        )
                        
            except Exception as e:
                st.error(f"Hata: {e}")

if __name__ == "__main__":
    main()