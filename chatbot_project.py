# chatbot_project.py (rag sistem ve arayüz entegre)

import os
import streamlit as st
from dotenv import load_dotenv

# LangChain Bileşenleri (Stabil ve Yeni Yapı)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

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

        # Embeddings Modelini Tanımla (Veri hazırlama dosyası ile AYNI OLMALI!)
        embeddings_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", # BU SATIRI DÜZELTİN!
        google_api_key=GEMINI_API_KEY
        )
          
        # Vektör Deposu Yükleniyor
        vector_store = Chroma(
            persist_directory=CHROMA_DB_PATH, 
            embedding_function=embeddings_model
        )

        # Geri Çekici (Retriever) Tanımlama
        # k=2, daha az ve daha alakalı belge çekerek LLM'in kafasını karıştırmasını engeller
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # LLM Tanımlama (API ile stabil çalışan model)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", # En stabil model adı seçildi
            temperature=0.4, 
            google_api_key=GEMINI_API_KEY
        )
        
        
        # 1. ADIM: PROMPT ŞABLONUNU OLUŞTURMA
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

        # 2. ADIM: MANUEL RAG ZİNCİRİNİ OLUŞTURMA
        def format_docs(docs):
            # Geri çekilen belgeleri tek bir string'de birleştirir
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            # 1. Retriever'ı çalıştırıp sonucu 'context'e atar
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            # 2. Prompt'u LLM'e gönderir
            | prompt
            | llm
            # 3. Yanıtı düz metne çevirir
            | StrOutputParser()
        )
        
        return rag_chain

    except Exception as e:
        # Gerçek hatayı gösteren hata ayıklama mesajı korundu
        st.error(f"RAG Zinciri kurulurken bir hata oluştu: {e}")
        st.stop()


# --- 2. FONKSİYON: Streamlit Arayüzü ---
def main():
    """Ana Streamlit arayüz fonksiyonu"""
    st.set_page_config(page_title="Film Öneri Asistanı (RAG)", layout="centered")
    st.title("🎬 Filmler Hakkında Sohbet Asistanı")
    st.caption("Veritabanımızdaki filmler hakkında sorular sorun.")

    qa_chain = setup_rag_chain() 

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Hangi film türlerini seversin veya ne önerebilirim?"):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Sensei düşünüyor..."):
                answer = ""
                try:
                    # Zinciri sorgulama
                    answer = qa_chain.invoke(prompt) 
                    
                    st.markdown(answer)
                except Exception as e:
                    # Gerçek hatayı gösteren hata ayıklama mesajı
                    st.error(f"Sorgulama Zincirinde Hata Oluştu: {e}")
                    answer = "Üzgünüm, şu anda bir hata oluştu veya aradığın bilgiyi veritabanımda bulamadım.^_^"
                    st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()