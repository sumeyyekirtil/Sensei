# Sensei
RAG Chatbot, developed for Akbank GAIB and powered by Gemini, that recommends movies based on "sense".
## Proje Fikri
Proje özellikle duygu yoğunluğu yaşanılan zamanlarda film izleyerek kafa dağıtma hobisi olanlara özel hazırlanmıştır.
Projedeki istenilen sonuç ; Kullanıcının ruh halini stabil(normal) duruma indirgemek, özellikle stres seviyesini kısaltmaktır.

# 🎬 Akbank GenAI Bootcamp Projesi: RAG Tabanlı Film Öneri Asistanı

## 🌟 1. Projenin Amacı

Bu proje, Akbank GenAI Bootcamp kapsamında geliştirilmiş, **Retrieval Augmented Generation (RAG)** mimarisine dayalı bir chatbot uygulamasıdır. Amacı, büyük bir film veri setini (metadata, özetler vb.) kullanarak, kullanıcıların sorduğu filmler hakkında doğru, hızlı ve bağlamda kalarak sınırlı yanıtlar sunmaktır.

* **Temel Hedef:** Kullanıcılardan gelen sorulara (örneğin: "En popüler romantik filmler hangileri?") sadece veritabanında (Vektör Deposu) bulunan bilgileri kullanarak cevap vermek ve dışarıdan bilgi uydurmayı(hallucination) engellemektir.
* **Sonuç:** Kullanıcıların duygu durumlarına göre film arama ve keşfetme deneyimini geliştiren akıllı bir sohbet arayüzü sunmaktır.

## 💾 2. Veri Seti Hakkında Bilgi

Projede kullanılan veri seti, kaggle platformundan alınıp, popüler film platformlarından türetilmiş (tmdb_5000_movies) ana dosyadan oluşmaktadır:

* **`tmdb_5000_movies.csv`**: Filmlere ait temel bilgiler "4799 film belge verisi"(Başlık, Türler).
* **Hazırlık Metodolojisi**: Bu dosya, `genres` (türler) sütunu, RAG zinciri tarafından daha iyi okunabilmesi için JSON formatından temizlenmiş ve formatlanmıştır.

## ⚙️ 3. Kullanılan Yöntemler ve Çözüm Mimarisi

### Çekirdek Teknolojiler

| Kategori | Teknoloji | Görevi |
| :--- | :--- | :--- |
| **Büyük Dil Modeli (LLM)** | Google **Gemini API (`gemini-2.0-flash`)** | Kullanıcı sorusunu ve bağlamı yorumlayarak nihai cevabı üretmek. |
| **RAG Çatısı** | **LangChain** (Gelişmiş Manuel Zincir) | Tüm bileşenleri (Retriever, Prompt, LLM) bir araya getirip veri akışını yönetmek. |
| **Embeddings** | **`GoogleGenerativeAIEmbeddings(model="paraphrase-multilingual-mpnet-base-v2")`** | Metinleri (özetler, sorgular) yüksek boyutlu vektörlere dönüştürmek. Özellikle türkçe dil modeli kullanılmıştır. |
| **Vektör Veritabanı** | **ChromaDB** | Film özetlerinin vektörlerini depolamak ve sorgu anında en alakalı belgeleri çekmek. |
| **Arayüz** | **Streamlit** | Chatbot için modern ve etkileşimli bir web arayüzü sunmak. |

### RAG Mimarisi (Gelişmiş Manuel Zincir)

Geleneksel modüller yerine, uyumluluk ve stabilite için LangChain'in **"Runnable"** bileşenleri kullanılarak manuel bir zincir kurulmuştur:

1.  **Retrieval (Çekme):** Soru vektörleştirilir ve `ChromaDB`'de aranır. `Retriever` (k=5) en alakalı 5 film belgesini çeker.
2.  **Prompt Formatlama:** Çekilen belgeler (`context`) ve orijinal soru, Gemini'ın yönergelerini içeren `system_prompt` içine yerleştirilir.
3.  **Üretim (Generation):** Hazırlanan Prompt, `ChatGoogleGenerativeAI(gemini-pro)` modeline gönderilir ve yanıt alınır.

## 🚀 4. Çalıştırma Kılavuzu (Local Kurulum)

Bu projenin yerel olarak çalıştırılması için gerekli adımlar aşağıdadır.

### Ön Koşullar

* Python 3.8+
* `GEMINI_API_KEY` (Google AI Studio'dan alınmış)
* `tmdb_5000_movies.csv`  veri dosyaları (proje klasöründe olmalı).

### Adımlar

1.  **Gerekli Kütüphaneleri Kurma:**
    Terminalde sanal ortam (`.\venv\Scripts\activate`) etkinleştirdikten sonra gerekli tüm kütüphaneleri kurma yolu:
    ```bash
    pip install pandas langchain-google-genai chromadb streamlit python-dotenv
    ```

2.  **API Anahtarını Ayarlama:**
    Projenin kök dizinine **`.env`** adında bir dosya oluştur ve API anahtarı aşağıdaki gibi eklenmeli ("" içinde olup olmaması sorun teşkil etmez):
    ```
    GEMINI_API_KEY="YOUR_API_KEY_HERE"
    ```

3.  **Vektör Veritabanını Oluşturma:**
    Veri dosyalarınızın kod ile aynı klasörde olmalı ve terminalde çalıştır. Bu adım, `chroma_db` klasörünü oluşturur.
    ```bash
    python veri_seti_rag.py
    ```
    (Bu adımda `*** SONUÇ: Toplam [5000'e yakın bir sayı] adet film belgesi hazırlandı. ***` çıktısını görmelisiniz.)

4.  **Web Arayüzünü Başlatma:**
    ```bash
    streamlit run chatbot_project.py
    ```
    (eğer çalıştırılmada hatalar olursa yapılması gerekenler sırayla:
    .1 İmport kütüphanelerin kontrolü eğer emin olunmazsa tekrar silip baştan kurma
    .2 İmport hatalarında "comminity(güvenli bağlan seçenekleri)" denenmeli
    .3 Veri çekememe - chroma db klasör yoluyla silinmeli, tekrar import edilmeli (veri dosyası çalıştırılarak)
       - eski vt silmek için windows powershell/cmd yolları farklıdır!
    .4 venv dosyası kontrolü hatalar çözülmez ise tekrar silinip kurulabilir
    .5 proje açılıp tarayıcıda alınan hatalardan biri rag temelli retriever ın yanlış bilgi çekmesi =>> {k=5} ayarlanabilir
    .6 tarayıcıda alınan hatalardan bir diğeri api kullanım kotası dolmuş olması >> sıfırlanması için (24 saat) bekleyiniz
    .7 API key tanıyamama hatası =>> !!hardcore tekniği ile önce veri seti dosyasına gömüp, sonra terminale api_key aracılığıyla tanımlanıp, güvenlik kontrolü için çalıştıktan sonra silinmeli!!
    .8 En Basit Hata >> python kodları tam yerinde istiyor dolayısıyla kod doğru bile olsa problem kısmında hata fırlatır kontrol sağlayınız.
    )

## 📈 5. Elde Edilen Sonuçlar Özeti

* **Stabil Çalışma:** Tüm API, `import` ve veri işleme hataları giderilmiş; proje, Gemini API'si ile hatasız iletişim kuran stabil bir RAG zincirine sahiptir.
* **Doğruluk ve Bağlam:** Embeddings modellerinin eşleştirilmesi ve manuel RAG zincirinin optimize edilmesi sayesinde, chatbot artık **veritabanındaki filmlerle ilgili doğru ve bağlamsal yanıtlar** verebilmektedir.
* **Kullanıcı Deneyimi:** Streamlit arayüzü, kullanıcıların kolayca soru sormasını ve anlık film önerileri almasını sağlar.

## 🌐 6. Web Arayüzü & Deploy Linki

Projenin çalışan versiyonuna aşağıdaki linkten ulaşabilirsiniz:
# Uygulama Ekran Örneği

![Sensei Chatbot Ekran Görüntüsü](app_screenshot.png/Ekran%20görüntüsü%202025-10-24%20000113.png)
![Sensei Chatbot Ekran Görüntüsü](app_screenshot.png/Ekran%20görüntüsü%202025-10-23%20223756.png)

**Deploy Linki:** [Proje Dağıtım Bağlantınız Buraya Gelecek]