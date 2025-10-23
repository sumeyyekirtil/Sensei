# chatbot_project.py (rag sistem ve arayüz entegre)

import os
import streamlit as st
from dotenv import load_dotenv

# LangChain Bileşenleri
from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings # <-- YENİ SATIR - api yenilenmesine rağmen limit 0 da olduğu için kütüphane import işlemi yapıldı
# chatbot_project.py (nihai, temiz, duygu entegrasyonlu versiyon)
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda # <--- BU SATIR - sorgulama zinciri pipe hatası için

# --- YAPILANDIRMA ---
load_dotenv()
CHROMA_DB_PATH = "./chroma_db" 

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY bulunamadı. Lütfen .env dosyanızı kontrol edin.")
    st.stop()
# --------------------

# --- 1. FONKSİYON: RAG Zincirini Kurma (MANUEL ZİNCİR) ---
@st.cache_resource
def setup_rag_chain():
    """RAG sistemini stabil LangChain bileşenleriyle kurar."""
    try:
        # 1. Embeddings Modeli (Çok Dilli Yerel Model)
        embeddings_model = HuggingFaceEmbeddings(
            model_name="paraphrase-multilingual-mpnet-base-v2"
            #"yeni model - sorgu - rag - veri seti değişiminde chroma db silinmeli yeniden oluşturulmalıdır"
            #"all-MiniLM-L6-v2" - yabancı dil modeli
        )
            
        # Vektör Deposu Yükleniyor
        vector_store = Chroma(
            persist_directory=CHROMA_DB_PATH, 
            embedding_function=embeddings_model
        )

        # Geri Çekici (Retriever) Tanımlama
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})

        # 2. LLM Modeli (Asıl Cevaplayıcı)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            temperature=0.4, 
            google_api_key=GEMINI_API_KEY
        )

        # 3. DUYGU TERCÜMANI (PRE-PROCESSING) ZİNCİRİ
        
        # 3a. Duygu Tercümanı Prompt'u
        emotion_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
         "Sen bir duygu durum tercümanısın. Kullanıcının verdiği ifadeyi sadece bir veya birkaç film türüne çevir. "
         "ÖRNEK: 'Neşeliyim' -> 'Komedi, Romantik', 'Üzgünüm' -> 'Dram', 'Korku filmi öner' -> 'Korku'. "
         "Eğer kullanıcı hem duygu hem de tür belirtiyor ve bunlar çelişiyorsa (Örnek: 'Mutluyum ama korku filmi izlemek istiyorum'), "
         "**LÜTFEN TÜR İSTEĞİNİ ÖNCELİKLENDİR VE DÖNDÜR (yani 'Korku' dön).** "
         "SADECE virgülle ayrılmış film türlerini döndür. BAŞKA HİÇBİR CÜMLE KURMA. "
         "Eğer hiçbir duygu veya tür çıkaramıyorsan 'Genel' kelimesini döndür."
        ),
        ("human", "{emotion_input}"),
    ]
)

        # 3b. Duygu Tercümanı LLM'i
        emotion_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            temperature=0.0, # Yaratıcılığı sıfırlıyoruz
            google_api_key=GEMINI_API_KEY
        )

        # 3c. Tür Çeviri Zinciri
        genre_translator_chain = (
            {"emotion_input": RunnablePassthrough()}
            | emotion_prompt
            | emotion_llm
            | StrOutputParser()
        )

        # 4. YANIT OLUŞTURMA (GENERATION) PROMPT'U
        system_prompt = (
            "Sen bir film öneri asistanısın. Yalnızca sağlanan 'Bağlam' içerisindeki filmlerle ilgili yanıt ver."
            "Yanıtını nazik ve kısa tut, film başlıklarını vurgula."
            "Eğer yanıtı BAĞLAM'da bulamıyorsan, 'Üzgünüm, aradığın filmi veya bilgiyi veritabanımda bulamadım.' de."
            "\n\nBağlam: {context}"
        )
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{question}"),
            ]
        )

        # 5. MANUEL RAG ZİNCİRİNİ OLUŞTURMA (Duygu Entegre)
        def format_docs(docs):
            # Geri çekilen belgeleri tek bir string'de birleştirir
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            # 1. KULLANICI SORUSUNU ÖNCE TÜR TERCÜMANINA GÖNDER
            RunnablePassthrough.assign(
                query_to_search=genre_translator_chain
            )
            # 2. RETRIEVER'A GİTMEDEN ÖNCE ÇEVRİLEN TÜRÜ YENİ SORGU OLARAK KULLAN
            .assign(
                # Çevrilen türü (query_to_search) retriever'a gönder,
                # ardından sonucu RunnableLambda ile format_docs'a aktar.
                context=lambda x: (retriever | RunnableLambda(format_docs)).invoke(x["query_to_search"]),
                
                # Orijinal kullanıcı sorusunu LLM'e göndermek için sakla
                question=lambda x: x["emotion_input"]
            )
            # 3. YANIT OLUŞTURMA
            | prompt 
            | llm
            | StrOutputParser()
        )
        
        # SİSTEM BURADA RAG ZİNCİRİNİ BAŞARILI BİR ŞEKİLDE DÖNDÜRMELİDİR
        return rag_chain

    except Exception as e:
        st.error(f"RAG Zinciri kurulurken bir hata oluştu: {e}")
        st.stop()


# --- 2. FONKSİYON: Streamlit Arayüzü ---
def main():
    """Ana Streamlit arayüz fonksiyonu"""
    st.set_page_config(page_title="Sensei - Film Öneri Asistanı (RAG)", layout="centered")
    st.title("🎬 Duygulardan Filmlere Açılan Kapı - Sensei")
    st.caption("Veritabanımızdaki filmler hakkında sorular sorun (örn: 'Çok mutluyum', 'Aksiyon filmi öner').")

    qa_chain = setup_rag_chain() 

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Bugün ruh halin nasıl? Hadi birlikte film gecesi düzenleyelim! Eğlenceye hazır mısın?"):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Sensei düşünüyor..."):
                answer = ""
                try:
                    # Zinciri sorgulama
                    # qa_chain, Runnable zinciri olduğu için invoke() metodu kullanılır.
                    answer = qa_chain.invoke({"emotion_input": prompt})
                    #answer = qa_chain.invoke(prompt) --eski prompt
                    
                    st.markdown(answer)
                except Exception as e:
                    # Hata yakalama
                    st.error(f"Sorgulama Zincirinde Hata Oluştu: {e}")
                    answer = "Üzgünüm, şu anda bir hata oluştu veya aradığın bilgiyi veritabanımda bulamadım. (Hata Kodu: {e})"
                    st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()