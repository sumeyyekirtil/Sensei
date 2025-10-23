# Sensei
RAG Chatbot, developed for Akbank GAIB and powered by Gemini, that recommends movies based on "sense".
# 🎬 Akbank GenAI Bootcamp Projesi: RAG Tabanlı Film Öneri Asistanı

## 🌟 1. Projenin Amacı

Bu proje, Akbank GenAI Bootcamp kapsamında geliştirilmiş, **Retrieval Augmented Generation (RAG)** mimarisine dayalı bir chatbot uygulamasıdır. Amacı, büyük bir film veri setini (metadata, özetler vb.) kullanarak, kullanıcıların sorduğu filmler hakkında doğru, hızlı ve bağlamsal olarak sınırlı yanıtlar sunmaktır.

* **Temel Hedef:** Kullanıcılardan gelen sorulara (örneğin: "En popüler romantik filmler hangileri?") sadece veritabanında (Vektör Deposu) bulunan bilgileri kullanarak cevap vermek ve dışarıdan bilgi uydurmayı (hallucination) engellemektir.
* **Sonuç:** Kullanıcıların film arama ve keşfetme deneyimini geliştiren akıllı bir sohbet arayüzü sunmaktır.

## 💾 2. Veri Seti Hakkında Bilgi

Projede kullanılan veri seti, popüler film platformlarından türetilmiş (TMDB/MovieLens) iki ana dosyadan oluşmaktadır:

* **`movies_metadata.csv`**: Filmlere ait temel bilgiler (Başlık, Özet, Türler, Yayın Tarihi, TMDB ID).
* **`links.csv`**: Filmlerin izleyici linklerini (IMDb ID) içerir.
* **Hazırlık Metodolojisi**: İki dosya, ortak TMDB ID'si üzerinden birleştirilmiş, özet ve tür bilgileri eksik olan satırlar temizlenmiştir. `genres` (türler) sütunu, RAG zinciri tarafından daha iyi okunabilmesi için JSON formatından temizlenmiş ve formatlanmıştır.

## ⚙️ 3. Kullanılan Yöntemler ve Çözüm Mimarisi

### Çekirdek Teknolojiler

| Kategori | Teknoloji | Görevi |
| :--- | :--- | :--- |
| **Büyük Dil Modeli (LLM)** | Google **Gemini API (`gemini-pro`)** | Kullanıcı sorusunu ve bağlamı yorumlayarak nihai cevabı üretmek. |
| **RAG Çatısı** | **LangChain** (Gelişmiş Manuel Zincir) | Tüm bileşenleri (Retriever, Prompt, LLM) bir araya getirip veri akışını yönetmek. |
| **Embeddings** | **`GoogleGenerativeAIEmbeddings(model="text-embedding-004")`** | Metinleri (özetler, sorgular) yüksek boyutlu vektörlere dönüştürmek. |
| **Vektör Veritabanı** | **ChromaDB** | Film özetlerinin vektörlerini depolamak ve sorgu anında en alakalı belgeleri çekmek. |
| **Arayüz** | **Streamlit** | Chatbot için modern ve etkileşimli bir web arayüzü sunmak. |

### RAG Mimarisi (Gelişmiş Manuel Zincir)

Geleneksel modüller yerine, uyumluluk ve stabilite için LangChain'in **"Runnable"** bileşenleri kullanılarak manuel bir zincir kurulmuştur:

1.  **Retrieval (Çekme):** Soru vektörleştirilir ve `ChromaDB`'de aranır. `Retriever` (k=2) en alakalı 2 film belgesini çeker.
2.  **Prompt Formatlama:** Çekilen belgeler (`context`) ve orijinal soru, Gemini'ın yönergelerini içeren `system_prompt` içine yerleştirilir.
3.  **Üretim (Generation):** Hazırlanan Prompt, `ChatGoogleGenerativeAI(gemini-pro)` modeline gönderilir ve yanıt alınır.

## 🚀 4. Çalıştırma Kılavuzu (Local Kurulum)

Bu projenin yerel olarak çalıştırılması için gerekli adımlar aşağıdadır.

### Ön Koşullar

* Python 3.8+
* `GEMINI_API_KEY` (Google AI Studio'dan alınmış)
* `movies_metadata.csv` ve `links.csv` veri dosyaları (proje klasöründe olmalı).

### Adımlar

1