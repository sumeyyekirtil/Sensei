# veri_seti_rag.py (.cvs dosyaları - Veri Seti Hazırlama ve Vektör Veritabanı Oluşturma)

import os
import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings # <-- YENİ SATIR - rag sorgu hatası alındı ve api değişimine rağmen kota düzelmedi
from dotenv import load_dotenv

# --- YAPILANDIRMA ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # Sadece Chatbot için kontrol amaçlı tutuldu

FILM_METADATA_DOSYASI = "tmdb_5000_movies.csv"
CHROMA_DB_PATH = "./chroma_db"
# --- YAPILANDIRMA SONU ---


def df_to_documents(df: pd.DataFrame) -> list[Document]:
    """Pandas DataFrame'i LangChain Document objelerine çevirir."""
    documents = []
    
    print(f"Başlangıç DataFrame satır sayısı: {len(df)}")
    # Veri temizleme adımları
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
    return documents


def veriyi_hazirla():
    """CSV dosyasını okur, temizler, Documents'a çevirir ve vektör veritabanına kaydeder."""
    
    try:
        df_meta = pd.read_csv(FILM_METADATA_DOSYASI, low_memory=False)
    except Exception as e:
        print(f"HATA: Dosya okunurken beklenmedik bir hata oluştu. Hata: {e}")
        return

    try:
        # Sütunları eşleştirme ve yeniden adlandırma
        df_meta = df_meta[['id', 'title', 'overview', 'genres']]
        df_meta = df_meta.rename(columns={'id': 'tmdbId', 'title': 'baslik', 'overview': 'ozet'})
    except KeyError as e:
        print(f"HATA: Sütun adları bulunamadı. Eksik sütun: {e}")
        return

    documents = df_to_documents(df_meta)
    if not documents:
        return

    # --- 4. Vektör Veritabanı (Chroma) Oluşturma (YEREL EMBEDDINGS KULLANILARAK) ---
    print("\n--- ChromaDB Oluşturuluyor (YEREL MODEL İLE) ---")
    
    try:
        # Embeddings Modelini Tanımla (YEREL HUGGINGFACE MODELİ - API BAĞIMSIZ)
        embeddings_model = HuggingFaceEmbeddings(
            model_name="paraphrase-multilingual-mpnet-base-v2"
        )

        # Document'ları veritabanına ekle
        Chroma.from_documents(
            documents,
            embeddings_model, 
            persist_directory=CHROMA_DB_PATH
        )

        print(f"Başarı: Vektör veritabanı {CHROMA_DB_PATH} konumunda başarıyla oluşturuldu.")
        
    except Exception as e:
        print(f"KRİTİK HATA: Vektör veritabanı oluşturulamadı. Hata: {e}")


if __name__ == "__main__":
    # Gerekli bağımlılıklar: pip install pandas langchain-community sentence-transformers chromadb
    veriyi_hazirla()