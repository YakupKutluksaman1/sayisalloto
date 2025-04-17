from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from typing import Dict

class LotoYorumlayici:
    def __init__(self):
        # .env dosyasını yükle
        load_dotenv()
        
        # OpenAI API anahtarını al
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # GPT-3.5-turbo modelini kullan (daha yaygın erişilebilir)
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",  # gpt-4 yerine gpt-3.5-turbo kullan
            temperature=0.8,  # 0.4'ten 0.8'e çıkaralım
            openai_api_key=self.openai_api_key
        )
        
        # Prompt şablonunu güncelle
        self.prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                "Sayısal loto için 6 FARKLI kupon hazırla. Her kupon TAMAMEN FARKLI sayılardan oluşmalıdır!"
                "\n\n"
                "ÖNEMLİ SINIRLAMALAR:"
                "- Tüm sayılar 1 ile 90 arasında olmalıdır"
                "- Her sayı sadece bir kez kullanılabilir"
                "- Her kupon birbirinden farklı olmalıdır"
                "\n\n"
                "1. **SICAK SAYILAR BAZLI KUPONLAR:**"
                "   Kupon 1: En sık çıkan ilk 6 sayıyı kullan (en_cok_cikan_sayilar listesinden)"
                "   Kupon 2: Şu şekilde oluştur (tüm sayılar 1-90 arası):"
                "           - Sıcak sayılardan ilk 3 sayıyı al"
                "           - En çok birlikte çıkan ikili gruptan 2 sayı al"
                "           - Ardışık sayılardan 1 tane al"
                "   Kupon 3: Şu şekilde oluştur (tüm sayılar 1-90 arası):"
                "           - Sıcak sayılardan 4. ve 5. sayıyı al"
                "           - En çok çıkan üçlü gruptan 3 sayı al"
                "           - Tek-çift dengesini gözeterek 1 sayı ekle"
                "\n\n"
                "2. **KOMBİNASYON BAZLI KUPONLAR:**"
                "   Kupon 4: Şu şekilde oluştur (tüm sayılar 1-90 arası):"
                "           - En çok tekrar eden ikili gruptan 2 sayı"
                "           - En çok tekrar eden üçlü gruptan 2 sayı"
                "           - Düşük (1-45) ve yüksek (46-90) denge için 2 sayı"
                "   Kupon 5: Garanti kupon - şu şekilde oluştur (tüm sayılar 1-90 arası):"
                "           - En sık çıkan sayılardan 2 tane"
                "           - En çok birlikte çıkan çiftlerden 2 tane"
                "           - Tekrar eden üçlü gruptan 2 tane"
                "   Kupon 6: Garanti kupon - şu şekilde oluştur (tüm sayılar 1-90 arası):"
                "           - Kupon 5'ten 3-4 sayıyı al"
                "           - Kalan sayıları sıcak sayılardan seç"
                "           - Tek-çift dengesini gözet"
                "\n\n"
                "SUPER STAR SEÇİMİ (1-90 arası):"
                "- Her kupon için farklı bir Super Star seç"
                "- Super Star'ları super_star_analizi'nden al"
                "- Her kuponda farklı Super Star kullan"
                "\n\n"
                "KUPON FORMATI:"
                "- [S1 S2 S3 S4 S5 S6] + SS: [X]"
                "- Tüm sayılar 1-90 arasında olmalı"
                "- Sayılar küçükten büyüğe sıralı olmalı"
                "- Her sayı sadece bir kez kullanılabilir"
                "\n\n"
                "ÖRNEK GEÇERLI KUPON:"
                "Kupon 1: [12 23 45 56 78 89] + SS: [34] (Açıklama...)"
                "\n\n"
                "ÖRNEK GEÇERSİZ KUPON:"
                "Kupon 1: [12 91 187 234 300] + SS: [95] (Sayılar 90'dan büyük!)"
                "\n\n"
                "KURALLAR:"
                "1. Her sayı 1-90 arasında olmalı"
                "2. Her kuponda tam 6 farklı sayı olmalı"
                "3. Her kuponda farklı bir Super Star olmalı (1-90 arası)"
                "4. Sayılar sıralı verilmeli"
                "5. Kupon 5 ve 6'da 3-4 ortak sayı olmalı"
            ),
            ("user", "{analyses}")
        ])
        
        self.chat_history = ChatMessageHistory()

    def analiz_yorumla(self, analiz_sonuclari: Dict) -> str:
        prompt = f"""Bir istatistik ve olasılık uzmanı olarak, aşağıdaki Çılgın Sayısal Loto analiz sonuçlarını detaylıca inceleyerek kapsamlı bir değerlendirme yap ve tahminlerde bulun.

Analiz Verileri:
{analiz_sonuclari}

Lütfen aşağıdaki başlıklar altında detaylı bir analiz sun:

1. İstatistiksel Örüntüler:
- Sayıların beklenen değerleri ve varyans analizini yorumla
- Ki-kare testi sonuçlarına göre rastgelelik değerlendirmesi
- Korelasyon analizinden çıkan önemli ilişkileri belirt

2. Zaman Serisi Analizi:
- Trend ve mevsimsellik değerlendirmesi
- ARIMA modeli sonuçlarının yorumu
- Gelecek tahminlerinin güvenilirliği

3. Olasılık Dağılımları:
- Poisson ve Binom dağılımlarından çıkan sonuçlar
- Ardışık sayıların analizi
- Sıcak ve soğuk sayıların değerlendirmesi

4. Monte Carlo Simülasyonları:
- Farklı stratejilerin başarı oranları
- En başarılı kombinasyonların analizi
- Risk değerlendirmesi

5. Markov Zinciri Analizi:
- Geçiş olasılıklarının yorumu
- Durağan dağılım sonuçları
- Tahmin güvenilirliği

6. Genel Değerlendirme ve Tahminler:
- Tüm analizlerden çıkan ortak örüntüler
- Bir sonraki çekiliş için 6 sayı tahmini (gerekçeleriyle)
- Süper Star tahmini ve nedeni
- Tahminlerin güven aralığı

7. Öneriler:
- Oyuncular için strateji önerileri
- Dikkat edilmesi gereken sayı grupları
- Risk yönetimi tavsiyeleri

Lütfen yanıtını matematiksel ve istatistiksel terimlerle destekle, ancak anlaşılır bir dille açıkla. Tahminlerini somut verilere dayandır ve neden bu sonuçlara vardığını detaylıca açıkla."""

        try:
            # LLM'den yanıt al
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Analiz sırasında bir hata oluştu: {str(e)}"

    def _dict_to_text(self, analiz_dict: Dict) -> str:
        """
        Analiz sözlüğünü metne dönüştürür
        """
        text_parts = []
        
        # Her bir analiz türü için özel formatlama
        if "sayisal_istatistikler" in analiz_dict:
            stats = analiz_dict["sayisal_istatistikler"]
            text_parts.append("İstatistiksel Analiz:\n" + 
                f"Ortalama: {stats['ortalama']}\n" +
                f"Varyans: {stats['varyans']}\n" +
                f"Standart Sapma: {stats['standart_sapma']}\n")
        
        if "ki_kare_analizi" in analiz_dict:
            ki_kare = analiz_dict["ki_kare_analizi"]
            text_parts.append("Ki-Kare Testi:\n" +
                f"Test İstatistiği: {ki_kare['test_istatistigi']}\n" +
                f"P-değeri: {ki_kare['p_deger']}\n" +
                f"Sonuç: {ki_kare['sonuc']}\n")
        
        if "korelasyon_analizi" in analiz_dict:
            korelasyon = analiz_dict["korelasyon_analizi"]
            text_parts.append("Korelasyon Analizi:\n" +
                "En Güçlü İlişkiler:\n" +
                "\n".join([f"{k}: {v}" for k, v in korelasyon["guclu_iliskiler"].items()]))
        
        if "zaman_serisi_analizi" in analiz_dict:
            zaman = analiz_dict["zaman_serisi_analizi"]
            text_parts.append("Zaman Serisi Analizi:\n" +
                f"Durağanlık: {zaman['durağanlık_testi']['durum']}\n" +
                f"ARIMA Parametreleri: {zaman['arima_model']['parametreler']}\n" +
                "Tahminler:\n" +
                "\n".join([f"Tahmin {k}: {v}" for k, v in zaman["tahminler"].items()]))
        
        if "olasilik_dagilim_analizi" in analiz_dict:
            olasilik = analiz_dict["olasilik_dagilim_analizi"]
            text_parts.append("Olasılık Dağılımları:\n" +
                f"Poisson Lambda: {olasilik['poisson_analizi']['lambda']}\n" +
                "En Yüksek Olasılıklı Sayılar:\n" +
                "\n".join([f"Sayı {k}: {v['olasilik']}" for k, v in olasilik["poisson_analizi"]["sayilar"].items()]))
        
        if "monte_carlo_analizi" in analiz_dict:
            monte = analiz_dict["monte_carlo_analizi"]
            text_parts.append("Monte Carlo Simülasyonları:\n" +
                "Strateji Sonuçları:\n" +
                "\n".join([f"{k}:\n{v['kazanma_olasiliklari']['altili']}" 
                          for k, v in monte["strateji_sonuclari"].items()]))
        
        if "markov_zinciri_analizi" in analiz_dict:
            markov = analiz_dict["markov_zinciri_analizi"]
            text_parts.append("Markov Zinciri Analizi:\n" +
                "En Yüksek Geçiş Olasılıkları:\n" +
                "\n".join([f"{k}: {v['olasilik']} ({v['yorum']})" 
                          for k, v in markov["gecis_olasiliklari"].items()]) +
                "\n\nDurağan Dağılım En Olası Sayılar:\n" +
                "\n".join([f"Sayı {k}: {v}" for k, v in markov["duragan_dagilim"].items()]) +
                "\n\nMarkov Tahminleri:\n" +
                f"Son Çekiliş: {markov['markov_tahminleri']['son_cekilis']}\n" +
                f"Tahmin Edilen Sayılar: {markov['markov_tahminleri']['tahminler']}\n" +
                f"Güven Skoru: {markov['markov_tahminleri']['guven_skoru']}\n\n" +
                "Genel Analiz:\n" +
                f"En Kararlı Sayı: {markov['genel_analiz']['en_kararli_sayi']}\n" +
                f"En Yüksek Geçiş: {markov['genel_analiz']['en_yuksek_gecis']}\n" +
                f"Ortalama Geçiş Olasılığı: {markov['genel_analiz']['ortalama_gecis_olasiligi']}")
        
        return "\n\n".join(text_parts)

    def sohbet_et(self, user_message: str) -> str:
        """
        Kullanıcı ile interaktif sohbet için
        """
        self.chat_history.add_user_message(user_message)
        response = self.llm.invoke(self.chat_history.messages)
        self.chat_history.add_ai_message(response.content)
        return response.content

# Kullanım örneği:
def main():
    yorumlayici = LotoYorumlayici()
    
    # Gerçek analiz sonuçlarını kullanalım
    analiz_sonuclari = {
        "en_cok_cikan_sayilar": {
            "45": 51, "69": 50, "60": 49, "56": 48, 
            "71": 48, "89": 47, "86": 45, "38": 44
        },
        "tek_cift_analizi": {
            "tek": "49.6%", 
            "cift": "50.4%"
        },
        "sicak_sayilar": [1, 88, 56, 38, 40],
        "soguk_sayilar": [81, 80, 60, 58, 72],
        "en_cok_birlikte": {
            "47-57": 9,
            "87-88": 8,
            "1-56": 8
        },
        "gecikme_sayilari": [96, 82, 29, 76, 28]
    }
    
    try:
        # Hata ayıklama için print ekleyelim
        print("API Anahtarı:", yorumlayici.openai_api_key[:10] + "...")
        print("Analiz sonuçları:", analiz_sonuclari)
        
        # Yorumları al
        yorum = yorumlayici.analiz_yorumla(analiz_sonuclari)
        print("\nLLM Yorumu:")
        print(yorum)
        
    except Exception as e:
        print("Hata oluştu:", str(e))

if __name__ == "__main__":
    main() 