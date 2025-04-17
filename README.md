# 🎲 Sayısal Loto Analiz ve Tahmin Sistemi

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen" alt="Status">
</div>

<br>

> 🔮 Yapay zeka destekli, istatistiksel analizlerle güçlendirilmiş Sayısal Loto tahmin sistemi

## 📋 İçindekiler

- [Özellikler](#-özellikler)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Proje Yapısı](#-proje-yapısı)
- [Katkıda Bulunma](#-katkıda-bulunma)
- [Lisans](#-lisans)
- [İletişim](#-iletişim)

## ✨ Özellikler

- 📊 **Gelişmiş Analizler**
  - Geçmiş çekiliş sonuçlarının detaylı analizi
  - İstatistiksel dağılım analizleri
  - Olasılık hesaplamaları ve simülasyonlar
  - Yapay zeka destekli tahmin sistemi

- 🔥 **Sayı Analizleri**
  - Sıcak/soğuk sayı tespiti
  - Tek/çift sayı dengesi
  - Birlikte çıkma olasılıkları
  - Ardışık sayı analizleri

- 🤖 **Yapay Zeka Özellikleri**
  - GPT-3.5 destekli tahmin sistemi
  - Örüntü tanıma ve analiz
  - Akıllı kupon önerileri
  - Risk değerlendirmesi

## 🚀 Kurulum

1. **Projeyi Klonlayın**
   ```bash
   git clone https://github.com/YakupKutluksaman1/sayisalloto.git
   cd sayisalloto
   ```

2. **Sanal Ortam Oluşturun (Önerilen)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   ```

3. **Gerekli Paketleri Yükleyin**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ortam Değişkenlerini Ayarlayın**
   ```bash
   cp .env.example .env
   # .env dosyasını düzenleyin ve API anahtarınızı ekleyin
   ```

## 💻 Kullanım

### Temel Kullanım
```bash
# Çekiliş sonuçlarını güncelle
python cekilisSonucları.py

# Analiz ve tahminleri görüntüle
python lotoYorumlayici.py
```

### Gelişmiş Özellikler
- 📈 İstatistiksel analizler
- 🔍 Örüntü tespiti
- 🎯 Tahmin oluşturma
- 📊 Raporlama

## 📁 Proje Yapısı

```
sayisalloto/
├── cekilisSonucları.py    # Ana analiz modülü
├── lotoYorumlayici.py     # Yapay zeka tahmin modülü
├── Kitap1.csv            # Geçmiş çekiliş verileri
├── .env                  # Ortam değişkenleri
├── .env.example          # Örnek ortam değişkenleri
└── requirements.txt      # Bağımlılıklar
```

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! İşte nasıl katkıda bulunabileceğiniz:

1. Bu depoyu fork edin
2. Yeni bir branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 📞 İletişim

**Yakup Kutluksaman** - [GitHub Profili](https://github.com/YakupKutluksaman1)

📧 E-posta: yakup.kutluksaman1mail.com

🔗 LinkedIn: [Yakup Kutluksaman](https://www.linkedin.com/in/yakup-kutluksaman/)

🌐 Proje Linki: [https://github.com/YakupKutluksaman1/sayisalloto](https://github.com/YakupKutluksaman1/sayisalloto)

---

<div align="center">
  <sub>Built with ❤️ by Yakup Kutluksaman</sub>
</div> 