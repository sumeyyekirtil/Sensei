# veri_seti_rag.py (.cvs dosyaları - Veri Seti Hazırlama ve Vektör Veritabanı Oluşturma)
# veri_seti_rag.py (Veri Seti Hazırlama ve Vektör Veritabanı Oluşturma)

import os
import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
# from dotenv import load_dotenv # Artık gerek yok, hardcoding (içine api key yazma) yapıyoruz

# --- YAPILANDIRMA ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 

# (Geri kalan API kontrol bloklarını ve değişkenleri doğru şekilde tanımlayın)

FILM_METADATA_DOSYASI = "tmdb_5000_movies.csv"
CHROMA_DB_PATH = "./chroma_db"
# --- YAPILANDIRMA SONU ---


def df_to_documents(df: pd.DataFrame) -> list[Document]:
    """Pandas DataFrame'i LangChain Document objelerine çevirir."""
    documents = []
    
    # ... (Temizleme ve Document oluşturma kısmı, önceki kodunuzla aynı ve doğru)
    print(f"Başlangıç DataFrame satır sayısı: {len(df)}")
    df = df.dropna(subset=['tmdbId', 'ozet']) 
    df = df[df['ozet'].str.strip() != ''] 
    print(f"Temizlemeden sonra kalan satır sayısı: {len(df)}")

    for index, row in df.iterrows():
        content = (
            f"Başlık: {row['baslik']}\n"
            f"Özet: {row['ozet']}\n"
            f"Türler: {row['genres']}"
        )

        doc = Document(
            page_content=content,
            metadata={
                "tmdbId": row['tmdbId'],
                "title": row['baslik'],
                "genres": row['genres']
            }
        )
        documents.append(doc)
    
    print(f"*** SONUÇ: Toplam {len(documents)} adet film belgesi hazırlandı. ***")
    
    if not documents:
        print("KRİTİK HATA: Hiçbir Document objesi oluşturulamadı. Veri setinizdeki 'ozet' sütununu kontrol edin.")
        return None
        
    return documents


def veriyi_hazirla():
    """CSV dosyasını okur, temizler, Documents'a çevirir ve vektör veritabanına kaydeder."""
    
    # ... (Veri okuma ve sütun eşleştirme kısmı, önceki kodunuzla aynı ve doğru)
    try:
        df_meta = pd.read_csv(FILM_METADATA_DOSYASI, low_memory=False)
    except Exception as e:
        print(f"HATA: Dosya okunurken beklenmedik bir hata oluştu. Hata: {e}")
        return

    try:
        df_meta = df_meta[['id', 'title', 'overview', 'genres']]
        df_meta = df_meta.rename(columns={'id': 'tmdbId', 'title': 'baslik', 'overview': 'ozet'})
    except KeyError as e:
        print(f"HATA: Sütun adları bulunamadı. Eksik sütun: {e}")
        return

    # veri_seti_rag.py dosyasında, veriyi_hazirla() fonksiyonunun son kısmı:

    documents = df_to_documents(df_meta)
    if not documents:
        return

    # --- 4. Vektör Veritabanı (Chroma) Oluşturma ---
    if not GEMINI_API_KEY:
        # API anahtarı yoksa burada durur
        print("HATA: GEMINI_API_KEY bulunamadı. Lütfen anahtarınızı kontrol edin.")
        return # <-- BURASI DOĞRU

    print("\n--- ChromaDB Oluşturuluyor ---") # <-- Artık if bloğunun dışında

    try: # <-- Artık if bloğunun dışında ve doğru seviyede
        # Embeddings Modelini Tanımla (SADECE GEREKLİ OLAN EMBEDDINGS)
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            # API anahtarını sadece bu parametre üzerinden iletelim:
            google_api_key=GEMINI_API_KEY
        )

        # Document'ları veritabanına ekle
        Chroma.from_documents(
            documents,
            embeddings, 
            persist_directory=CHROMA_DB_PATH
        )

        print(f"Başarı: Vektör veritabanı {CHROMA_DB_PATH} konumunda başarıyla oluşturuldu.")
        
    except Exception as e:
        # Hata ayıklamayı kolaylaştırmak için chromadb hatası hariç tutulabilir, ancak şimdilik kalsın.
        print(f"KRİTİK HATA: Vektör veritabanı oluşturulamadı. Hata: {e}")



if __name__ == "__main__":
    # ChromaDB'nin kurulu olduğundan emin olun (önceki adımda kuruldu)
    # pip install chromadb

    veriyi_hazirla()