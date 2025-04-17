import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from itertools import combinations
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
import datetime
import random
from lotoYorumlayici import LotoYorumlayici  # Yeni import
from scipy.stats import chi2
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import poisson, binom

# CSV dosyasını okuma
df = pd.read_csv('Kitap1.csv', sep=';')

# Mevcut sütun isimlerini görelim
print("CSV dosyasındaki sütun isimleri:")
print(df.columns.tolist())

# Tarih sütununu kontrol edelim
print("\nTarih sütunu örnek veriler:")
print(df['Tarih'].head())

def en_cok_cikan_sayilar():
    """En çok çıkan sayıları döndürür"""
    sayac = Counter(df['1'].tolist() + df['2'].tolist() + df['3'].tolist() + 
                   df['4'].tolist() + df['5'].tolist() + df['6'].tolist())
    return sorted(sayac.items(), key=lambda x: x[1], reverse=True)

def cikis_oruntusu_analizi():
    # Bir sayı çıktığında bir sonraki çekilişte gelen sayıların analizi
    onceki_sayilar = set()
    sonraki_sayilar = []
    
    for i in range(len(df)-1):
        mevcut_cekilisteki_sayilar = set()
        for j in range(1, 7):
            mevcut_cekilisteki_sayilar.add(df.iloc[i][str(j)])
            if i > 0:  # İlk çekiliş değilse
                if df.iloc[i][str(j)] in onceki_sayilar:
                    sonraki_sayilar.append(df.iloc[i][str(j)])
        onceki_sayilar = mevcut_cekilisteki_sayilar
    
    return Counter(sonraki_sayilar).most_common(5)

def en_cok_birlikte_cikan_sayilar():
    """En çok birlikte çıkan sayı ikililerini bulur"""
    ikili_sayac = defaultdict(int)
    for _, row in df.iterrows():
        sayilar = [row[str(i)] for i in range(1, 7)]
        for i, j in combinations(sayilar, 2):
            if i < j:
                ikili_sayac[(i, j)] += 1
            else:
                ikili_sayac[(j, i)] += 1
    return sorted(ikili_sayac.items(), key=lambda x: x[1], reverse=True)

def ardisik_sayilar_analizi():
    ardisik_sayilar_count = 0
    for _, row in df.iterrows():
        sayilar = sorted([row[str(i)] for i in range(1, 7)])
        for i in range(len(sayilar)-1):
            if sayilar[i+1] - sayilar[i] == 1:
                ardisik_sayilar_count += 1
    return ardisik_sayilar_count

def sayi_gruplari_dengesi():
    dusuk_sayilar = 0  # 1-45 arası
    yuksek_sayilar = 0  # 46-90 arası
    
    for _, row in df.iterrows():
        for i in range(1, 7):
            if row[str(i)] <= 45:
                dusuk_sayilar += 1
            else:
                yuksek_sayilar += 1
    
    return {
        'Düşük Sayılar (1-45)': dusuk_sayilar,
        'Yüksek Sayılar (46-90)': yuksek_sayilar
    }

def tek_cift_analizi():
    tek_sayilar = 0
    cift_sayilar = 0
    
    for _, row in df.iterrows():
        for i in range(1, 7):
            if row[str(i)] % 2 == 0:
                cift_sayilar += 1
            else:
                tek_sayilar += 1
    
    return {'Tek Sayılar': tek_sayilar, 'Çift Sayılar': cift_sayilar}

def super_star_analizi():
    """Super Star sayılarının analizini yapar"""
    sayac = Counter(df['Super Star'])
    return sorted(sayac.items(), key=lambda x: x[1], reverse=True)

def cikmamis_sayilar():
    tum_sayilar = set(range(1, 91))
    cikan_sayilar = set()
    
    for i in range(1, 7):
        cikan_sayilar.update(df[str(i)].unique())
    
    return sorted(list(tum_sayilar - cikan_sayilar))

def sanslı_gun_analizi():
    try:
        df['Hafta_Gunu'] = pd.to_datetime(df['Tarih'], format='%d.%m.%Y').dt.day_name()
        gun_dagilimi = df.groupby('Hafta_Gunu').size()
        
        if gun_dagilimi.empty:
            print("  Günlere göre dağılım analizi yapılamadı - yeterli veri yok")
            return None
            
        print("\nGünlere Göre Çekiliş Dağılımı:")
        for gun, sayi in gun_dagilimi.items():
            print(f"  {gun}: {sayi} çekiliş")
        return gun_dagilimi
    except Exception as e:
        print(f"  Günlere göre dağılım analizi hatası: {str(e)}")
        return None

def grafikleri_ciz():
    # plt.figure(figsize=(15, 10))
    # ... diğer kodlar ...
    # plt.show()
    pass

def makine_ogrenmesi_tahmin():
    """Makine öğrenmesi ile tahmin yapar"""
    # Son 10 çekilişin verilerini al
    son_10_cekilis = df.head(10)
    
    # Basit bir tahmin stratejisi: son 10 çekilişte en çok çıkan 6 sayı
    tum_sayilar = []
    for i in range(1, 7):
        tum_sayilar.extend(son_10_cekilis[str(i)].tolist())
    
    sayac = Counter(tum_sayilar)
    en_cok_cikanlar = [sayi for sayi, _ in sayac.most_common(6)]
    
    return sorted(en_cok_cikanlar)

def zaman_serisi_analizi():
    try:
        df['Ortalama'] = df[[str(i) for i in range(1, 7)]].mean(axis=1)
        df['Tarih_Converted'] = pd.to_datetime(df['Tarih'], format='%d.%m.%Y')
        
        if len(df) < 10:  # Minimum veri kontrolü
            print("  Zaman serisi analizi yapılamadı - yeterli veri yok (en az 10 çekiliş gerekli)")
            return
            
        df_copy = df.copy()
        df_copy.set_index('Tarih_Converted', inplace=True)
        
        # Mevsimsel ayrıştırma
        decomposition = seasonal_decompose(df_copy['Ortalama'], period=52)
        
        # plt.figure(figsize=(15, 10))
        # plt.subplot(411)
        # plt.plot(df_copy['Ortalama'])
        # plt.title('Orijinal Zaman Serisi')
        
        # plt.subplot(412)
        # plt.plot(decomposition.trend)
        # plt.title('Trend')
        
        # plt.subplot(413)
        # plt.plot(decomposition.seasonal)
        # plt.title('Mevsimsellik')
        
        # plt.subplot(414)
        # plt.plot(decomposition.resid)
        # plt.title('Artıklar')
        
        # plt.tight_layout()
        # plt.show()
    except Exception as e:
        print(f"  Zaman serisi analizi hatası: {str(e)}")

def monte_carlo_simulasyonu(n_simulasyon=1000):
    # Monte Carlo simülasyonu ile olası kombinasyonlar üretme
    simulasyon_sonuclari = []
    for _ in range(n_simulasyon):
        simulasyon = sorted(np.random.choice(range(1, 91), size=6, replace=False))
        simulasyon_sonuclari.append(simulasyon)
    
    # En çok üretilen kombinasyonları bulma
    kombinasyonlar = [tuple(x) for x in simulasyon_sonuclari]
    en_cok_uretilen = Counter(kombinasyonlar).most_common(5)
    return en_cok_uretilen

def mevsimsel_analiz():
    try:
        df['Ay'] = pd.to_datetime(df['Tarih'], format='%d.%m.%Y').dt.month
        aylik_ortalamalar = df.groupby('Ay')[[str(i) for i in range(1, 7)]].mean()
        
        if aylik_ortalamalar.empty:
            print("  Mevsimsel analiz yapılamadı - yeterli veri yok")
            return
            
        # plt.figure(figsize=(12, 6))
        # plt.plot(aylik_ortalamalar.mean(axis=1))
        # plt.title('Aylara Göre Çıkan Sayıların Ortalaması')
        # plt.xlabel('Ay')
        # plt.ylabel('Ortalama Sayı')
        # plt.show()
    except Exception as e:
        print(f"  Mevsimsel analiz hatası: {str(e)}")

def korelasyon_analizi():
    try:
        korelasyon_matrisi = df[[str(i) for i in range(1, 7)]].corr()
        
        if korelasyon_matrisi.empty:
            print("  Korelasyon analizi yapılamadı - yeterli veri yok")
            return
            
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(korelasyon_matrisi, annot=True, cmap='coolwarm')
        # plt.title('Sayılar Arası Korelasyon Matrisi')
        # plt.show()
    except Exception as e:
        print(f"  Korelasyon analizi hatası: {str(e)}")

def korelasyon_analizi_detayli():
    """
    Sayılar arasındaki korelasyonları detaylı olarak analiz eder
    """
    # Tüm sayıları matrise dönüştür
    sayilar_matrix = df[[str(i) for i in range(1, 7)]].values
    
    # 90x90'lık bir korelasyon matrisi oluştur
    korelasyon_matrisi = np.zeros((90, 90))
    
    # Her sayı çifti için korelasyon hesapla
    for i in range(90):
        for j in range(90):
            # i+1 ve j+1 sayılarının birlikte çıkma durumunu kontrol et
            sayi1_varmi = (sayilar_matrix == (i + 1)).any(axis=1)
            sayi2_varmi = (sayilar_matrix == (j + 1)).any(axis=1)
            
            # Phi katsayısı hesapla
            n11 = np.sum(sayi1_varmi & sayi2_varmi)  # Her ikisi de var
            n10 = np.sum(sayi1_varmi & ~sayi2_varmi)  # Sadece 1. var
            n01 = np.sum(~sayi1_varmi & sayi2_varmi)  # Sadece 2. var
            n00 = np.sum(~sayi1_varmi & ~sayi2_varmi)  # Her ikisi de yok
            
            try:
                phi = (n11 * n00 - n10 * n01) / np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
                korelasyon_matrisi[i, j] = phi
            except:
                korelasyon_matrisi[i, j] = 0
    
    # En güçlü pozitif korelasyonlar
    pozitif_korelasyonlar = []
    for i in range(90):
        for j in range(i + 1, 90):
            if korelasyon_matrisi[i, j] > 0:
                pozitif_korelasyonlar.append(((i + 1, j + 1), korelasyon_matrisi[i, j]))
    
    # En güçlü negatif korelasyonlar
    negatif_korelasyonlar = []
    for i in range(90):
        for j in range(i + 1, 90):
            if korelasyon_matrisi[i, j] < 0:
                negatif_korelasyonlar.append(((i + 1, j + 1), korelasyon_matrisi[i, j]))
    
    # En güçlü korelasyonları sırala
    en_guclu_pozitif = sorted(pozitif_korelasyonlar, key=lambda x: x[1], reverse=True)[:5]
    en_guclu_negatif = sorted(negatif_korelasyonlar, key=lambda x: x[1])[:5]
    
    # Sonuçları yazdır
    print("\n3. Korelasyon Analizi:")
    print("\nEn Güçlü Pozitif Korelasyonlar:")
    for (sayi1, sayi2), corr in en_guclu_pozitif:
        print(f"Sayılar: {sayi1}-{sayi2}, Korelasyon: {corr:.3f}")
    
    print("\nEn Güçlü Negatif Korelasyonlar:")
    for (sayi1, sayi2), corr in en_guclu_negatif:
        print(f"Sayılar: {sayi1}-{sayi2}, Korelasyon: {corr:.3f}")
    
    # Görselleştirme için heatmap
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(korelasyon_matrisi, cmap='coolwarm', center=0)
    # plt.title('Sayılar Arası Korelasyon Matrisi')
    # plt.xlabel('Sayı')
    # plt.ylabel('Sayı')
    # plt.show()
    
    # LLM için sonuçları sözlük formatında döndür
    return {
        "pozitif_korelasyonlar": {
            f"{sayi1}-{sayi2}": {
                "korelasyon": f"{corr:.3f}",
                "yorum": "Güçlü pozitif ilişki" if corr > 0.5 else "Orta pozitif ilişki"
            } for (sayi1, sayi2), corr in en_guclu_pozitif
        },
        "negatif_korelasyonlar": {
            f"{sayi1}-{sayi2}": {
                "korelasyon": f"{corr:.3f}",
                "yorum": "Güçlü negatif ilişki" if corr < -0.5 else "Orta negatif ilişki"
            } for (sayi1, sayi2), corr in en_guclu_negatif
        },
        "genel_yorum": {
            "en_guclu_iliski": f"{en_guclu_pozitif[0][0][0]}-{en_guclu_pozitif[0][0][1]} ({en_guclu_pozitif[0][1]:.3f})",
            "en_zayif_iliski": f"{en_guclu_negatif[0][0][0]}-{en_guclu_negatif[0][0][1]} ({en_guclu_negatif[0][1]:.3f})"
        }
    }

def coklu_grup_analizi():
    # Üçlü ve dörtlü kombinasyonları analiz etme
    uclu_kombinasyonlar = []
    dortlu_kombinasyonlar = []
    
    for _, row in df.iterrows():
        sayilar = [row[str(i)] for i in range(1, 7)]
        # Üçlü kombinasyonlar
        uclu_kombinasyonlar.extend(list(combinations(sorted(sayilar), 3)))
        # Dörtlü kombinasyonlar
        dortlu_kombinasyonlar.extend(list(combinations(sorted(sayilar), 4)))
    
    # En çok tekrar eden grupları bulma
    en_cok_uclu = Counter(uclu_kombinasyonlar).most_common(5)
    en_cok_dortlu = Counter(dortlu_kombinasyonlar).most_common(5)
    
    return en_cok_uclu, en_cok_dortlu

def detayli_yillik_analiz():
    """
    Her yıl için detaylı analiz yapar
    """
    df['Yil'] = pd.to_datetime(df['Tarih'], format='%d.%m.%Y').dt.year
    yillik_analizler = {}
    
    for yil in sorted(df['Yil'].unique(), reverse=True):
        yil_verisi = df[df['Yil'] == yil]
        
        # 1. O yılın sıcak sayıları
        tum_sayilar = []
        for i in range(1, 7):
            tum_sayilar.extend(yil_verisi[str(i)].tolist())
        sicak_sayilar = Counter(tum_sayilar).most_common(10)
        
        # 2. O yılın grup analizleri
        ikili_gruplar = []
        uclu_gruplar = []
        ardisik_ikili = []  # Ardışık ikili gruplar
        ardisik_uclu = []   # Ardışık üçlü gruplar
        
        for _, row in yil_verisi.iterrows():
            sayilar = sorted([row[str(i)] for i in range(1, 7)])
            
            # Normal gruplar
            ikili_gruplar.extend(combinations(sayilar, 2))
            uclu_gruplar.extend(combinations(sayilar, 3))
            
            # Ardışık grupları bul
            for i in range(len(sayilar)-1):
                if sayilar[i+1] - sayilar[i] == 1:
                    ardisik_ikili.append((sayilar[i], sayilar[i+1]))
                if i < len(sayilar)-2 and sayilar[i+2] - sayilar[i+1] == 1 and sayilar[i+1] - sayilar[i] == 1:
                    ardisik_uclu.append((sayilar[i], sayilar[i+1], sayilar[i+2]))
        
        # Grup analizleri
        grup_analizleri = {
            'tekrar_eden_ikili': Counter(ikili_gruplar).most_common(5),
            'tekrar_eden_uclu': Counter(uclu_gruplar).most_common(3),
            'ardisik_ikili': Counter(ardisik_ikili).most_common(3),
            'ardisik_uclu': Counter(ardisik_uclu).most_common(2)
        }
        
        # 3. O yılın Super Star analizi
        super_star_sayilari = Counter(yil_verisi['Super Star']).most_common(5)
        
        # 4. O yılın trend analizi
        tek_sayilar = sum(1 for x in tum_sayilar if x % 2 != 0)
        cift_sayilar = sum(1 for x in tum_sayilar if x % 2 == 0)
        dusuk_sayilar = sum(1 for x in tum_sayilar if x <= 45)
        yuksek_sayilar = sum(1 for x in tum_sayilar if x > 45)
        
        # Sonuçları kaydet
        yillik_analizler[yil] = {
            'sicak_sayilar': sicak_sayilar,
            'grup_analizleri': grup_analizleri,
            'super_star': super_star_sayilari,
            'tek_cift_orani': (tek_sayilar, cift_sayilar),
            'dusuk_yuksek_orani': (dusuk_sayilar, yuksek_sayilar),
            'cekilisler': len(yil_verisi)
        }
    
    # Sonuçları yazdır
    for yil, analiz in yillik_analizler.items():
        print(f"\n{yil} Yılı Detaylı Analizi:")
        print(f"Toplam Çekiliş: {analiz['cekilisler']}")
        
        print("\nEn Sıcak 10 Sayı:")
        for sayi, tekrar in analiz['sicak_sayilar']:
            print(f"  {sayi}: {tekrar} kez")
        
        print("\nGrup Analizleri:")
        print("  En Çok Tekrar Eden İkili Gruplar:")
        for (sayi1, sayi2), tekrar in analiz['grup_analizleri']['tekrar_eden_ikili']:
            print(f"    {sayi1}-{sayi2}: {tekrar} kez")
        
        print("\n  En Çok Tekrar Eden Üçlü Gruplar:")
        for (sayi1, sayi2, sayi3), tekrar in analiz['grup_analizleri']['tekrar_eden_uclu']:
            print(f"    {sayi1}-{sayi2}-{sayi3}: {tekrar} kez")
        
        print("\n  En Çok Tekrar Eden Ardışık İkililer:")
        for (sayi1, sayi2), tekrar in analiz['grup_analizleri']['ardisik_ikili']:
            print(f"    {sayi1}-{sayi2}: {tekrar} kez")
        
        print("\n  En Çok Tekrar Eden Ardışık Üçlüler:")
        for (sayi1, sayi2, sayi3), tekrar in analiz['grup_analizleri']['ardisik_uclu']:
            print(f"    {sayi1}-{sayi2}-{sayi3}: {tekrar} kez")
        
        print("\nEn Çok Çıkan Super Star Sayıları:")
        for sayi, tekrar in analiz['super_star']:
            print(f"  {sayi}: {tekrar} kez")
        
        tek, cift = analiz['tek_cift_orani']
        print(f"\nTek-Çift Dağılımı:")
        print(f"  Tek: %{(tek/(tek+cift))*100:.1f}")
        print(f"  Çift: %{(cift/(tek+cift))*100:.1f}")
        
        dusuk, yuksek = analiz['dusuk_yuksek_orani']
        print(f"\nDüşük-Yüksek Dağılımı:")
        print(f"  1-45: %{(dusuk/(dusuk+yuksek))*100:.1f}")
        print(f"  46-90: %{(yuksek/(dusuk+yuksek))*100:.1f}")
    
    return yillik_analizler

def sicak_soguk_sayilar():
    """Son 20 çekilişte en çok ve en az çıkan sayıları bulur"""
    son_20_cekilis = df.head(20)
    tum_sayilar = []
    for i in range(1, 7):
        tum_sayilar.extend(son_20_cekilis[str(i)].tolist())
    
    sayac = Counter(tum_sayilar)
    sicak_sayilar = sayac.most_common(10)
    soguk_sayilar = sayac.most_common()[:-11:-1]
    
    return sicak_sayilar, soguk_sayilar

def gecikme_analizi():
    """En uzun süredir çıkmayan sayıları bulur"""
    son_gorulme = {i: 0 for i in range(1, 91)}
    for index, row in df.iterrows():
        for i in range(1, 7):
            sayi = row[str(i)]
            son_gorulme[sayi] = index
    
    gecikme_sureleri = {sayi: len(df) - son_gorulme[sayi] 
                       for sayi in son_gorulme}
    return sorted(gecikme_sureleri.items(), key=lambda x: x[1], reverse=True)[:10]

def gelismis_tahmin_analizi():
    # 1. Sıcak-Soğuk Sayılar Analizi
    def sicak_soguk_sayilar():
        """Son 20 çekilişte en çok ve en az çıkan sayıları bulur"""
        son_20_cekilis = df.head(20)
        tum_sayilar = []
        for i in range(1, 7):
            tum_sayilar.extend(son_20_cekilis[str(i)].tolist())
        
        sayac = Counter(tum_sayilar)
        sicak_sayilar = sayac.most_common(10)
        soguk_sayilar = sayac.most_common()[:-11:-1]
        
        return sicak_sayilar, soguk_sayilar
    
    # 2. Gecikme Analizi (En uzun süredir çıkmayan sayılar)
    def gecikme_analizi():
        son_gorulme = {i: 0 for i in range(1, 91)}
        for index, row in df.iterrows():
            for i in range(1, 7):
                sayi = row[str(i)]
                son_gorulme[sayi] = index
        
        gecikme_sureleri = {sayi: len(df) - son_gorulme[sayi] 
                           for sayi in son_gorulme}
        return sorted(gecikme_sureleri.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # 3. Örüntü Bazlı Tahmin
    def oruntu_tahmini():
        # Son 5 çekilişte tekrar eden sayı grupları
        son_5_cekilis = []
        for _, row in df.head(5).iterrows():
            son_5_cekilis.append(sorted([row[str(i)] for i in range(1, 7)]))
        
        # Tekrar eden ikili ve üçlü grupları bul
        tekrar_eden_gruplar = []
        for cekilisler in combinations(son_5_cekilis, 2):
            ortak_sayilar = set(cekilisler[0]) & set(cekilisler[1])
            if len(ortak_sayilar) >= 2:
                tekrar_eden_gruplar.extend(combinations(sorted(ortak_sayilar), 2))
        
        return Counter(tekrar_eden_gruplar).most_common(5)
    
    # 4. Dengeli Dağılım Tahmini
    def dengeli_dagilim_tahmini():
        # Sayıları 15'erli gruplara böl (1-15, 16-30, ..., 76-90)
        grup_sayilari = {i: 0 for i in range(6)}
        for _, row in df.head(20).iterrows():
            for i in range(1, 7):
                sayi = row[str(i)]
                grup = (sayi - 1) // 15
                grup_sayilari[grup] += 1
        
        # En az temsil edilen gruplardan sayı seç
        az_temsil_gruplar = sorted(grup_sayilari.items(), key=lambda x: x[1])
        return [random.randint(g*15+1, (g+1)*15) for g, _ in az_temsil_gruplar[:3]]
    
    # Sonuçları yazdır
    print("\nGelişmiş Tahmin Analizi:")
    
    print("\n1. Sıcak-Soğuk Sayılar:")
    sicak, soguk = sicak_soguk_sayilar()
    print("Sıcak Sayılar (Son 20 çekilişte en çok çıkanlar):")
    for sayi, tekrar in sicak:
        print(f"  Sayı: {sayi}, Tekrar: {tekrar}")
    print("\nSoğuk Sayılar (Son 20 çekilişte en az çıkanlar):")
    for sayi, tekrar in soguk:
        print(f"  Sayı: {sayi}, Tekrar: {tekrar}")
    
    print("\n2. En Uzun Süredir Çıkmayan Sayılar:")
    for sayi, gecikme in gecikme_analizi():
        print(f"  Sayı: {sayi}, Çekilişten beri çıkmadı: {gecikme}")
    
    print("\n3. Son Çekilişlerde Tekrar Eden Örüntüler:")
    for (sayi1, sayi2), tekrar in oruntu_tahmini():
        print(f"  Sayı çifti: {sayi1}-{sayi2}, Tekrar: {tekrar}")
    
    print("\n4. Dengeli Dağılım İçin Önerilen Sayılar:")
    print("  ", dengeli_dagilim_tahmini())
    
    # Tüm analizleri birleştirerek tahmin üret
    tahmin_seti = set()
    
    # Sıcak sayılardan 2 tane al
    tahmin_seti.update([sayi for sayi, _ in sicak[:2]])
    
    # Soğuk sayılardan 1 tane al
    tahmin_seti.update([sayi for sayi, _ in soguk[:1]])
    
    # Gecikme analizinden 1 tane al
    tahmin_seti.update([sayi for sayi, _ in gecikme_analizi()[:1]])
    
    # Örüntü tahmininden 1 çift al
    if oruntu_tahmini():
        tahmin_seti.update(oruntu_tahmini()[0][0])
    
    # Dengeli dağılımdan eksik sayıları tamamla
    while len(tahmin_seti) < 6:
        tahmin_seti.add(random.choice(dengeli_dagilim_tahmini()))
    
    print("\nBirleştirilmiş Tahmin Seti (Sıcak ve Soğuk Sayılar Dahil):")
    print(sorted(list(tahmin_seti)))
    print("\nTahmin Seti Dağılımı:")
    print("- Sıcak Sayılar:", [s for s in tahmin_seti if any(s == sayi for sayi, _ in sicak[:2])])
    print("- Soğuk Sayılar:", [s for s in tahmin_seti if any(s == sayi for sayi, _ in soguk[:1])])
    print("- Gecikme Sayıları:", [s for s in tahmin_seti if any(s == sayi for sayi, _ in gecikme_analizi()[:1])])

def tekrar_eden_gruplar_analizi():
    """
    Tekrar eden sayı gruplarını analiz eder
    2'li, 3'lü ve 4'lü grupları bulur
    """
    # Son 100 çekilişi al
    son_cekilisler = df.head(100)
    
    # Tüm kombinasyonları topla
    ikili_gruplar = []
    uclu_gruplar = []
    dortlu_gruplar = []
    
    
    for _, row in son_cekilisler.iterrows():
        sayilar = sorted([row[str(i)] for i in range(1, 7)])
        # İkili gruplar
        ikili_gruplar.extend(combinations(sayilar, 2))
        # Üçlü gruplar
        uclu_gruplar.extend(combinations(sayilar, 3))
        
        # Dörtlü gruplar
        dortlu_gruplar.extend(combinations(sayilar, 4))
    
    # En çok tekrar edenleri bul
    ikili_tekrar = Counter(ikili_gruplar).most_common(10)
    uclu_tekrar = Counter(uclu_gruplar).most_common(10)
    dortlu_tekrar = Counter(dortlu_gruplar).most_common(5)
    
    # Sonuçları yazdır
    print("\nEn çok tekrar eden ikili gruplar:")
    for grup, tekrar in ikili_tekrar:
        if tekrar > 2:  # En az 3 kez tekrar edenler
            print(f"Sayılar: {grup}, {tekrar} kez tekrar etmiş")
    
    print("\nEn çok tekrar eden üçlü gruplar:")
    for grup, tekrar in uclu_tekrar:
        if tekrar > 1:  # En az 2 kez tekrar edenler
            print(f"Sayılar: {grup}, {tekrar} kez tekrar etmiş")
    
    print("\nEn çok tekrar eden dörtlü gruplar:")
    for grup, tekrar in dortlu_tekrar:
        if tekrar > 1:  # En az 2 kez tekrar edenler
            print(f"Sayılar: {grup}, {tekrar} kez tekrar etmiş")
    
    return {
        'ikili_gruplar': ikili_tekrar,
        'uclu_gruplar': uclu_tekrar,
        'dortlu_gruplar': dortlu_tekrar
    }

def sicak_sayilar_iliskisi():
    """
    Sıcak sayılar arasındaki ilişkileri analiz eder:
    - Birlikte çıkma sıklığı
    - Ardışık çıkma durumu
    - Grup oluşturma eğilimi
    """
    # Son 50 çekilişi al
    son_cekilisler = df.head(50)
    
    # Sıcak sayıları bul
    sicak_sayilar, _ = sicak_soguk_sayilar()
    sicak_sayilar = [sayi for sayi, _ in sicak_sayilar[:10]]  # İlk 10 sıcak sayı
    
    # Sıcak sayılar arası ilişkiler
    iliskiler = {
        'birlikte_cikma': [],
        'ardisik_cikma': [],
        'uclu_gruplar': []
    }
    
    # Birlikte çıkma analizi
    for i, sayi1 in enumerate(sicak_sayilar):
        for sayi2 in sicak_sayilar[i+1:]:
            birlikte_sayisi = 0
            for _, row in son_cekilisler.iterrows():
                sayilar = [row[str(i)] for i in range(1, 7)]
                if sayi1 in sayilar and sayi2 in sayilar:
                    birlikte_sayisi += 1
            if birlikte_sayisi > 2:  # En az 3 kez birlikte çıkanlar
                iliskiler['birlikte_cikma'].append((sayi1, sayi2, birlikte_sayisi))
    
    # Ardışık çıkma analizi
    for sayi in sicak_sayilar:
        onceki_cekilis = None
        ardisik_sayisi = 0
        for _, row in son_cekilisler.iterrows():
            sayilar = [row[str(i)] for i in range(1, 7)]
            if sayi in sayilar:
                if onceki_cekilis and onceki_cekilis - row.name == 1:
                    ardisik_sayisi += 1
                onceki_cekilis = row.name
        if ardisik_sayisi > 1:  # En az 2 kez ardışık çıkanlar
            iliskiler['ardisik_cikma'].append((sayi, ardisik_sayisi))
    
    # Üçlü grup analizi
    for i, sayi1 in enumerate(sicak_sayilar):
        for j, sayi2 in enumerate(sicak_sayilar[i+1:], i+1):
            for sayi3 in sicak_sayilar[j+1:]:
                uclu_sayisi = 0
                for _, row in son_cekilisler.iterrows():
                    sayilar = set([row[str(i)] for i in range(1, 7)])
                    if sayi1 in sayilar and sayi2 in sayilar and sayi3 in sayilar:
                        uclu_sayisi += 1
                if uclu_sayisi > 1:  # En az 2 kez birlikte çıkan üçlüler
                    iliskiler['uclu_gruplar'].append((sayi1, sayi2, sayi3, uclu_sayisi))
    
    # Sonuçları yazdır
    print("\nSıcak Sayılar Arası İlişki Analizi:")
    print("\n1. Sık Birlikte Çıkan Sıcak Sayılar:")
    for sayi1, sayi2, tekrar in sorted(iliskiler['birlikte_cikma'], key=lambda x: x[2], reverse=True):
        print(f"  {sayi1}-{sayi2}: {tekrar} kez birlikte")
    
    print("\n2. Ardışık Çekilişlerde Tekrar Eden Sıcak Sayılar:")
    for sayi, tekrar in sorted(iliskiler['ardisik_cikma'], key=lambda x: x[1], reverse=True):
        print(f"  {sayi}: {tekrar} kez ardışık çekilişlerde")
    
    print("\n3. Sık Birlikte Çıkan Üçlü Sıcak Sayı Grupları:")
    for sayi1, sayi2, sayi3, tekrar in sorted(iliskiler['uclu_gruplar'], key=lambda x: x[3], reverse=True):
        print(f"  {sayi1}-{sayi2}-{sayi3}: {tekrar} kez birlikte")
    
    return iliskiler

def sayisal_istatistikler():
    """
    Sayıların beklenen değerlerini ve varyansını hesaplar
    """
    # Tüm sayıları bir listede topla
    tum_sayilar = []
    for i in range(1, 7):
        tum_sayilar.extend(df[str(i)].tolist())
    
    # Sayıların frekansını hesapla (1-90 arası)
    frekanslar = {i: tum_sayilar.count(i) for i in range(1, 91)}
    
    # Toplam çekiliş sayısı
    toplam_cekilis = len(df)
    
    # Beklenen değer hesaplaması
    teorik_beklenen = toplam_cekilis * 6 / 90  # Her sayının eşit olasılıkla çıkması durumu
    gercek_ortalama = sum(tum_sayilar) / len(tum_sayilar)
    
    # Varyans hesaplaması
    teorik_varyans = sum((x - gercek_ortalama) ** 2 for x in tum_sayilar) / len(tum_sayilar)
    
    # Standart sapma
    standart_sapma = teorik_varyans ** 0.5
    
    # En çok ve en az çıkan sayılar
    en_cok = sorted(frekanslar.items(), key=lambda x: x[1], reverse=True)[:5]
    en_az = sorted(frekanslar.items(), key=lambda x: x[1])[:5]
    
    # Sonuçları yazdır
    print("\n1. Sayıların Beklenen Değerleri ve Varyansı Analizi:")
    print("\nGenel İstatistikler:")
    print(f"Toplam Çekiliş Sayısı: {toplam_cekilis}")
    print(f"Teorik Beklenen Değer (her sayı için): {teorik_beklenen:.2f}")
    print(f"Gerçek Ortalama: {gercek_ortalama:.2f}")
    print(f"Varyans: {teorik_varyans:.2f}")
    print(f"Standart Sapma: {standart_sapma:.2f}")
    
    print("\nEn Çok Çıkan 5 Sayı:")
    for sayi, frekans in en_cok:
        sapma = ((frekans - teorik_beklenen) / teorik_beklenen) * 100
        print(f"Sayı: {sayi}, Frekans: {frekans}, Beklenen Değerden Sapma: %{sapma:.2f}")
    
    print("\nEn Az Çıkan 5 Sayı:")
    for sayi, frekans in en_az:
        sapma = ((frekans - teorik_beklenen) / teorik_beklenen) * 100
        print(f"Sayı: {sayi}, Frekans: {frekans}, Beklenen Değerden Sapma: %{sapma:.2f}")
    
    # LLM için sonuçları sözlük formatında döndür
    return {
        "istatistikler": {
            "toplam_cekilis": toplam_cekilis,
            "teorik_beklenen": f"{teorik_beklenen:.2f}",
            "gercek_ortalama": f"{gercek_ortalama:.2f}",
            "varyans": f"{teorik_varyans:.2f}",
            "standart_sapma": f"{standart_sapma:.2f}"
        },
        "en_cok_cikanlar": {
            str(sayi): {
                "frekans": frekans,
                "sapma_yuzdesi": f"{((frekans - teorik_beklenen) / teorik_beklenen) * 100:.2f}%"
            } for sayi, frekans in en_cok
        },
        "en_az_cikanlar": {
            str(sayi): {
                "frekans": frekans,
                "sapma_yuzdesi": f"{((frekans - teorik_beklenen) / teorik_beklenen) * 100:.2f}%"
            } for sayi, frekans in en_az
        }
    }

def ki_kare_testi():
    """
    Sayıların dağılımının rastgeleliğini test eder
    """
    # Tüm sayıları bir listede topla
    tum_sayilar = []
    for i in range(1, 7):
        tum_sayilar.extend(df[str(i)].tolist())
    
    # Sayıların frekansını hesapla (1-90 arası)
    gozlenen_frekanslar = {i: tum_sayilar.count(i) for i in range(1, 91)}
    
    # Toplam çekiliş sayısı
    toplam_cekilis = len(df)
    
    # Beklenen frekans (eşit dağılım varsayımı altında)
    beklenen_frekans = toplam_cekilis * 6 / 90
    
    # Ki-kare istatistiği hesaplama
    ki_kare = sum(((gozlenen_frekanslar[i] - beklenen_frekans) ** 2) / beklenen_frekans 
                  for i in range(1, 91))
    
    # Serbestlik derecesi
    serbestlik_derecesi = 90 - 1
    
    # Kritik değer (0.05 anlamlılık düzeyi için)
    kritik_deger = chi2.ppf(0.95, serbestlik_derecesi)
    
    # p-değeri hesaplama
    p_deger = 1 - chi2.cdf(ki_kare, serbestlik_derecesi)
    
    # En büyük sapmalar
    sapmalar = {i: ((gozlenen_frekanslar[i] - beklenen_frekans) ** 2) / beklenen_frekans 
                for i in range(1, 91)}
    en_buyuk_sapmalar = sorted(sapmalar.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Sonuçları yazdır
    print("\n2. Ki-Kare Testi Analizi:")
    print("\nTest İstatistikleri:")
    print(f"Ki-Kare Değeri: {ki_kare:.2f}")
    print(f"Serbestlik Derecesi: {serbestlik_derecesi}")
    print(f"Kritik Değer (0.05): {kritik_deger:.2f}")
    print(f"p-değeri: {p_deger:.4f}")
    
    print("\nTest Sonucu:")
    if ki_kare > kritik_deger:
        print("Dağılım rastgele DEĞİLDİR (H0 hipotezi reddedilir)")
    else:
        print("Dağılım rastgeledir (H0 hipotezi reddedilemez)")
    
    print("\nEn Büyük Sapmalar:")
    for sayi, sapma in en_buyuk_sapmalar:
        gozlenen = gozlenen_frekanslar[sayi]
        sapma_yuzdesi = ((gozlenen - beklenen_frekans) / beklenen_frekans) * 100
        print(f"Sayı: {sayi}, Gözlenen: {gozlenen}, "
              f"Beklenen: {beklenen_frekans:.2f}, "
              f"Sapma: {sapma:.2f}, "
              f"Sapma Yüzdesi: %{sapma_yuzdesi:.2f}")
    
    # LLM için sonuçları sözlük formatında döndür
    return {
        "test_istatistikleri": {
            "ki_kare": f"{ki_kare:.2f}",
            "serbestlik_derecesi": serbestlik_derecesi,
            "kritik_deger": f"{kritik_deger:.2f}",
            "p_deger": f"{p_deger:.4f}",
            "sonuc": "Rastgele Değil" if ki_kare > kritik_deger else "Rastgele"
        },
        "en_buyuk_sapmalar": {
            str(sayi): {
                "gozlenen": gozlenen_frekanslar[sayi],
                "beklenen": f"{beklenen_frekans:.2f}",
                "sapma": f"{sapma:.2f}",
                "sapma_yuzdesi": f"{((gozlenen_frekanslar[sayi] - beklenen_frekans) / beklenen_frekans) * 100:.2f}%"
            } for sayi, sapma in en_buyuk_sapmalar
        }
    }

def zaman_serisi_analizi_detayli():
    """
    Sayıların zaman içindeki değişimlerini analiz eder ve ARIMA modeli ile tahmin yapar
    """
    # Tarihleri düzgün formata çevir
    df['Tarih'] = pd.to_datetime(df['Tarih'], format='%d.%m.%Y')
    
    # Her çekilişin ortalama değerini hesapla
    df['Ortalama'] = df[[str(i) for i in range(1, 7)]].mean(axis=1)
    
    # Zaman serisini oluştur
    zaman_serisi = df.set_index('Tarih')['Ortalama']
    
    # Durağanlık testi (Augmented Dickey-Fuller)
    adf_test = adfuller(zaman_serisi)
    
    # ARIMA modeli için parametreleri belirle
    p = range(0, 3)  # AR parametresi
    d = range(0, 2)  # Fark alma derecesi
    q = range(0, 3)  # MA parametresi
    
    # En iyi parametreleri bul
    en_iyi_aic = float('inf')
    en_iyi_parametreler = None
    
    for i in p:
        for j in d:
            for k in q:
                try:
                    model = ARIMA(zaman_serisi, order=(i,j,k))
                    sonuc = model.fit()
                    if sonuc.aic < en_iyi_aic:
                        en_iyi_aic = sonuc.aic
                        en_iyi_parametreler = (i,j,k)
                except:
                    continue
    
    # En iyi model ile tahmin yap
    model = ARIMA(zaman_serisi, order=en_iyi_parametreler)
    sonuc = model.fit()
    
    # Gelecek 5 çekiliş için tahmin
    tahminler = sonuc.forecast(steps=5)
    
    # Sonuçları yazdır
    print("\n4. Zaman Serisi Analizi:")
    
    print("\nDurağanlık Testi (ADF):")
    print(f"Test İstatistiği: {adf_test[0]:.3f}")
    print(f"p-değeri: {adf_test[1]:.3f}")
    print("Durağanlık: ", "Seri Durağan" if adf_test[1] < 0.05 else "Seri Durağan Değil")
    
    print("\nEn İyi ARIMA Modeli:")
    print(f"Parametreler (p,d,q): {en_iyi_parametreler}")
    print(f"AIC: {en_iyi_aic:.2f}")
    
    print("\nGelecek 5 Çekiliş için Tahminler:")
    for i, tahmin in enumerate(tahminler, 1):
        print(f"Tahmin {i}: {tahmin:.2f}")
    
    # LLM için sonuçları sözlük formatında döndür
    return {
        "durağanlık_testi": {
            "test_istatistigi": f"{adf_test[0]:.3f}",
            "p_deger": f"{adf_test[1]:.3f}",
            "durum": "Durağan" if adf_test[1] < 0.05 else "Durağan Değil"
        },
        "arima_model": {
            "parametreler": f"p={en_iyi_parametreler[0]}, d={en_iyi_parametreler[1]}, q={en_iyi_parametreler[2]}",
            "aic": f"{en_iyi_aic:.2f}"
        },
        "tahminler": {
            f"tahmin_{i}": f"{tahmin:.2f}"
            for i, tahmin in enumerate(tahminler, 1)
        },
        "trend_analizi": {
            "ortalama": f"{zaman_serisi.mean():.2f}",
            "std": f"{zaman_serisi.std():.2f}",
            "min": f"{zaman_serisi.min():.2f}",
            "max": f"{zaman_serisi.max():.2f}"
        }
    }

def olasilik_dagilim_analizi():
    """
    Sayıların çıkma olasılıklarını Poisson ve Binom dağılımları ile analiz eder
    """
    # Tüm sayıları bir listede topla
    tum_sayilar = []
    for i in range(1, 7):
        tum_sayilar.extend(df[str(i)].tolist())
    
    # Toplam çekiliş sayısı
    toplam_cekilis = len(df)
    
    # Her sayının frekansını hesapla
    frekanslar = {i: tum_sayilar.count(i) for i in range(1, 91)}
    
    # Poisson Dağılımı Analizi
    # Ortalama çıkma sayısı (lambda)
    lambda_poisson = toplam_cekilis * 6 / 90  # Her sayı için beklenen ortalama
    
    # En çok çıkan 5 sayı için Poisson olasılıkları
    en_cok_cikan = sorted(frekanslar.items(), key=lambda x: x[1], reverse=True)[:5]
    poisson_olasiliklar = {}
    
    for sayi, gercek_frekans in en_cok_cikan:
        # Poisson olasılığı hesapla
        poisson_olasilik = poisson.pmf(gercek_frekans, lambda_poisson)
        poisson_olasiliklar[sayi] = {
            "gercek_frekans": gercek_frekans,
            "beklenen_frekans": lambda_poisson,
            "poisson_olasilik": poisson_olasilik
        }
    
    # Binom Dağılımı Analizi
    # Her çekilişte bir sayının çıkma olasılığı p = 6/90
    p_binom = 6/90
    n_denemeler = toplam_cekilis
    
    # Seçili sayılar için Binom olasılıkları
    binom_olasiliklar = {}
    
    for sayi, gercek_frekans in en_cok_cikan:
        # Binom olasılığı hesapla
        binom_olasilik = binom.pmf(gercek_frekans, n_denemeler, p_binom)
        binom_olasiliklar[sayi] = {
            "gercek_frekans": gercek_frekans,
            "beklenen_frekans": n_denemeler * p_binom,
            "binom_olasilik": binom_olasilik
        }
    
    # Ardışık sayıların analizi
    ardisik_ciftler = []
    for _, row in df.iterrows():
        sayilar = sorted([row[str(i)] for i in range(1, 7)])
        for i in range(len(sayilar)-1):
            if sayilar[i+1] - sayilar[i] == 1:
                ardisik_ciftler.append((sayilar[i], sayilar[i+1]))
    
    ardisik_olasilik = len(ardisik_ciftler) / (toplam_cekilis * 5)  # Her çekilişte 5 komşu çift var
    
    # Sonuçları yazdır
    print("\n5. Olasılık Dağılımları Analizi:")
    print("\nPoisson Dağılımı Analizi:")
    print(f"Beklenen Ortalama (lambda): {lambda_poisson:.2f}")
    
    print("\nEn Çok Çıkan Sayılar için Poisson Olasılıkları:")
    for sayi, degerler in poisson_olasiliklar.items():
        print(f"\nSayı: {sayi}")
        print(f"Gerçek Frekans: {degerler['gercek_frekans']}")
        print(f"Beklenen Frekans: {degerler['beklenen_frekans']:.2f}")
        print(f"Poisson Olasılığı: {degerler['poisson_olasilik']:.6f}")
    
    print("\nBinom Dağılımı Analizi:")
    print(f"Başarı Olasılığı (p): {p_binom:.4f}")
    print(f"Deneme Sayısı (n): {n_denemeler}")
    
    print("\nEn Çok Çıkan Sayılar için Binom Olasılıkları:")
    for sayi, degerler in binom_olasiliklar.items():
        print(f"\nSayı: {sayi}")
        print(f"Gerçek Frekans: {degerler['gercek_frekans']}")
        print(f"Beklenen Frekans: {degerler['beklenen_frekans']:.2f}")
        print(f"Binom Olasılığı: {degerler['binom_olasilik']:.6f}")
    
    print("\nArdışık Sayılar Analizi:")
    print(f"Ardışık Sayı Çiftleri Olasılığı: {ardisik_olasilik:.4f}")
    
    # LLM için sonuçları sözlük formatında döndür
    return {
        "poisson_analizi": {
            "lambda": f"{lambda_poisson:.2f}",
            "sayilar": {
                str(sayi): {
                    "gercek_frekans": degerler["gercek_frekans"],
                    "beklenen_frekans": f"{degerler['beklenen_frekans']:.2f}",
                    "olasilik": f"{degerler['poisson_olasilik']:.6f}"
                } for sayi, degerler in poisson_olasiliklar.items()
            }
        },
        "binom_analizi": {
            "p": f"{p_binom:.4f}",
            "n": n_denemeler,
            "sayilar": {
                str(sayi): {
                    "gercek_frekans": degerler["gercek_frekans"],
                    "beklenen_frekans": f"{degerler['beklenen_frekans']:.2f}",
                    "olasilik": f"{degerler['binom_olasilik']:.6f}"
                } for sayi, degerler in binom_olasiliklar.items()
            }
        },
        "ardisik_analiz": {
            "olasilik": f"{ardisik_olasilik:.4f}",
            "yorum": "Yüksek" if ardisik_olasilik > 0.2 else "Normal" if ardisik_olasilik > 0.1 else "Düşük"
        }
    }

def monte_carlo_analizi_detayli(n_simulasyon=1_000_000):
    """
    Detaylı Monte Carlo simülasyonları ile farklı stratejileri test eder
    """
    # Gerçek verilerden istatistikler
    tum_sayilar = []
    for i in range(1, 7):
        tum_sayilar.extend(df[str(i)].tolist())
    
    # En sık çıkan sayılar
    sicak_sayilar = [sayi for sayi, _ in Counter(tum_sayilar).most_common(20)]
    
    # Farklı stratejiler için sonuçlar
    stratejiler = {
        "rastgele": [],
        "sicak_sayilar": [],
        "dengeli": [],
        "ardisik": []
    }
    
    # Simülasyonları gerçekleştir
    print(f"\nMonte Carlo simülasyonu başlıyor ({n_simulasyon:,} iterasyon)...")
    for i in range(n_simulasyon):
        if i % 100_000 == 0:  # Her 100,000 iterasyonda ilerleme göster
            print(f"İlerleme: {i:,}/{n_simulasyon:,} ({(i/n_simulasyon)*100:.1f}%)")
            
        # 1. Tamamen rastgele strateji
        rastgele = sorted(np.random.choice(range(1, 91), size=6, replace=False))
        stratejiler["rastgele"].append(rastgele)
        
        # 2. Sıcak sayılardan seçme stratejisi
        sicak = sorted(np.random.choice(sicak_sayilar, size=6, replace=False))
        stratejiler["sicak_sayilar"].append(sicak)
        
        # 3. Dengeli strateji (düşük-yüksek, tek-çift dengesi)
        dusuk = np.random.choice(range(1, 46), size=3, replace=False)
        yuksek = np.random.choice(range(46, 91), size=3, replace=False)
        dengeli = sorted(np.concatenate([dusuk, yuksek]))
        stratejiler["dengeli"].append(dengeli)
        
        # 4. Ardışık sayıları içeren strateji
        baslangic = np.random.randint(1, 85)
        ardisik = sorted(list(range(baslangic, baslangic + 3)) + 
                        list(np.random.choice(range(1, 91), size=3, replace=False)))
        stratejiler["ardisik"].append(ardisik)
    
    print("Monte Carlo simülasyonu tamamlandı!")
    
    # Her strateji için kazanma olasılıklarını hesapla
    gercek_cekilisler = df[[str(i) for i in range(1, 7)]].values
    kazanma_oranlari = {}
    
    print("\nStrateji Bazlı Kazanma Olasılıkları:")
    for strateji_adi, kombinasyonlar in stratejiler.items():
        altili = 0
        besli = 0
        dortlu = 0
        
        for kombinasyon in kombinasyonlar:
            for cekilis in gercek_cekilisler:
                eslesen = len(set(kombinasyon) & set(cekilis))
                if eslesen == 6:
                    altili += 1
                elif eslesen == 5:
                    besli += 1
                elif eslesen == 4:
                    dortlu += 1
        
        kazanma_oranlari[strateji_adi] = {
            "altili": altili / n_simulasyon,
            "besli": besli / n_simulasyon,
            "dortlu": dortlu / n_simulasyon
        }
        
        print(f"\n{strateji_adi.upper()} Strateji:")
        print(f"6 Tutturma Olasılığı: {kazanma_oranlari[strateji_adi]['altili']:.6f}")
        print(f"5 Tutturma Olasılığı: {kazanma_oranlari[strateji_adi]['besli']:.6f}")
        print(f"4 Tutturma Olasılığı: {kazanma_oranlari[strateji_adi]['dortlu']:.6f}")
    
    # En başarılı kombinasyonları bul
    en_basarili = {
        strateji: Counter([tuple(k) for k in komb]).most_common(3)
        for strateji, komb in stratejiler.items()
    }
    
    print("\nEn Çok Üretilen Kombinasyonlar:")
    for strateji, kombinasyonlar in en_basarili.items():
        print(f"\n{strateji.upper()} Strateji:")
        for komb, tekrar in kombinasyonlar:
            print(f"Kombinasyon: {komb}, Tekrar: {tekrar}")
    
    # LLM için sonuçları sözlük formatında döndür
    return {
        "simulasyon_detaylari": {
            "toplam_simulasyon": n_simulasyon,
            "analiz_tarihi": datetime.datetime.now().strftime("%d.%m.%Y")
        },
        "strateji_sonuclari": {
            strateji: {
                "kazanma_olasiliklari": {
                    "altili": f"{oranlar['altili']:.6f}",
                    "besli": f"{oranlar['besli']:.6f}",
                    "dortlu": f"{oranlar['dortlu']:.6f}"
                },
                "en_basarili_kombinasyonlar": {
                    f"kombinasyon_{i+1}": {
                        "sayilar": list(komb),
                        "tekrar": tekrar
                    } for i, (komb, tekrar) in enumerate(en_basarili[strateji])
                }
            } for strateji, oranlar in kazanma_oranlari.items()
        },
        "genel_degerlendirme": {
            "en_iyi_strateji": max(kazanma_oranlari.items(), 
                                key=lambda x: x[1]['altili'])[0],
            "en_kotu_strateji": min(kazanma_oranlari.items(), 
                                key=lambda x: x[1]['altili'])[0]
        }
    }

def markov_zinciri_analizi():
    """
    Sayıların çıkış örüntülerini Markov Zincirleri ile analiz eder
    """
    # Tüm sayıları sıralı olarak al ve 1-90 aralığında olduğundan emin ol
    sayilar_sirayla = []
    for _, row in df.iterrows():
        sayilar = [row[str(i)] for i in range(1, 7)]
        # Sadece 1-90 arasındaki sayıları al
        sayilar = [s for s in sayilar if 1 <= s <= 90]
        sayilar_sirayla.extend(sorted(sayilar))
    
    # Geçiş matrisi oluştur (90x90)
    gecis_matrisi = np.zeros((90, 90))
    
    # Geçiş sayılarını hesapla
    for i in range(len(sayilar_sirayla) - 1):
        try:
            mevcut = sayilar_sirayla[i] - 1  # 0-89 arası indeks için
            sonraki = sayilar_sirayla[i + 1] - 1
            if 0 <= mevcut < 90 and 0 <= sonraki < 90:  # İndeks kontrolü
                gecis_matrisi[mevcut][sonraki] += 1
        except IndexError:
            continue  # Hatalı indeksleri atla
    
    # Geçiş olasılıklarını hesapla
    satir_toplamlari = gecis_matrisi.sum(axis=1)
    satir_toplamlari[satir_toplamlari == 0] = 1  # 0'a bölmeyi önle
    gecis_olasiliklari = gecis_matrisi / satir_toplamlari[:, np.newaxis]
    
    # En yüksek geçiş olasılıklarını bul
    en_yuksek_gecisler = []
    for i in range(90):
        for j in range(90):
            if gecis_olasiliklari[i, j] > 0:
                en_yuksek_gecisler.append(((i+1, j+1), gecis_olasiliklari[i, j]))
    
    en_yuksek_gecisler = sorted(en_yuksek_gecisler, key=lambda x: x[1], reverse=True)[:10]
    
    # Durağan dağılımı hesapla
    try:
        eigvals, eigvecs = np.linalg.eig(gecis_olasiliklari.T)
        duragan_dagilim = eigvecs[:, np.argmax(eigvals)].real
        duragan_dagilim = duragan_dagilim / duragan_dagilim.sum()
    except:
        duragan_dagilim = np.ones(90) / 90
    
    # En yüksek olasılıklı sayıları bul
    en_olasilikli = sorted([(i+1, p) for i, p in enumerate(duragan_dagilim)], 
                          key=lambda x: x[1], reverse=True)[:5]
    
    # Markov tahminleri
    son_cekilis = sorted([row[str(i)] for i in range(1, 7) if 1 <= row[str(i)] <= 90])
    tahminler = []
    
    for sayi in son_cekilis:
        if 1 <= sayi <= 90:  # Sayı kontrolü
            # Her sayı için en olası sonraki sayıyı bul
            siradaki_olasiliklar = gecis_olasiliklari[sayi-1]
            tahmin = np.argmax(siradaki_olasiliklar) + 1
            tahminler.append(tahmin)
    
    # Sonuçları yazdır
    print("\n7. Markov Zinciri Analizi:")
    
    print("\nEn Yüksek Geçiş Olasılıkları:")
    for (sayi1, sayi2), olasilik in en_yuksek_gecisler:
        print(f"{sayi1} -> {sayi2}: {olasilik:.4f}")
    
    print("\nDurağan Dağılıma Göre En Olası Sayılar:")
    for sayi, olasilik in en_olasilikli:
        print(f"Sayı: {sayi}, Olasılık: {olasilik:.4f}")
    
    print("\nSon Çekilişe Göre Markov Tahminleri:")
    print(f"Son Çekiliş: {son_cekilis}")
    print(f"Tahmin Edilen Sayılar: {sorted(tahminler)}")
    
    # LLM için sonuçları sözlük formatında döndür
    return {
        "gecis_olasiliklari": {
            f"{sayi1}->{sayi2}": {
                "olasilik": f"{olasilik:.4f}",
                "yorum": "Yüksek" if olasilik > 0.5 else "Orta" if olasilik > 0.2 else "Düşük"
            } for (sayi1, sayi2), olasilik in en_yuksek_gecisler
        },
        "duragan_dagilim": {
            str(sayi): f"{olasilik:.4f}"
            for sayi, olasilik in en_olasilikli
        },
        "markov_tahminleri": {
            "son_cekilis": son_cekilis,
            "tahminler": sorted(tahminler),
            "guven_skoru": f"{np.mean([gecis_olasiliklari[s-1].max() for s in son_cekilis]):.4f}"
        },
        "genel_analiz": {
            "en_kararli_sayi": en_olasilikli[0][0],
            "en_yuksek_gecis": f"{en_yuksek_gecisler[0][0][0]}->{en_yuksek_gecisler[0][0][1]}",
            "ortalama_gecis_olasiligi": f"{np.mean([o for _, o in en_yuksek_gecisler]):.4f}"
        }
    }

def yil_bazli_analiz(yil=2025):
    """
    Belirli bir yıla ait çekiliş verilerini analiz eder
    """
    # Tarihi datetime'a çevir
    df['Tarih'] = pd.to_datetime(df['Tarih'], format='%d.%m.%Y')
    yil_df = df[df['Tarih'].dt.year == yil].copy()
    
    if len(yil_df) == 0:
        print(f"{yil} yılına ait veri bulunamadı!")
        return None
    
    print(f"\n{yil} Yılı Özel Analizi:")
    print(f"Toplam Çekiliş Sayısı: {len(yil_df)}")
    
    # En çok çıkan sayılar
    tum_sayilar = []
    for i in range(1, 7):
        tum_sayilar.extend(yil_df[str(i)].tolist())
    sayac = Counter(tum_sayilar)
    
    print("\nEn çok çıkan 5 sayı:")
    for sayi, tekrar in sayac.most_common(5):
        print(f"Sayı: {sayi}, Tekrar: {tekrar}")
    
    # Super Star analizi
    super_star_sayac = Counter(yil_df['Super Star'])
    print("\nEn çok çıkan 3 Super Star:")
    for sayi, tekrar in super_star_sayac.most_common(3):
        print(f"Super Star: {sayi}, Tekrar: {tekrar}")
    
    # Tek-Çift analizi
    tek = sum(1 for x in tum_sayilar if x % 2 == 1)
    cift = sum(1 for x in tum_sayilar if x % 2 == 0)
    print(f"\nTek-Çift Dağılımı:")
    print(f"Tek Sayılar: {tek} ({tek/(tek+cift)*100:.1f}%)")
    print(f"Çift Sayılar: {cift} ({cift/(tek+cift)*100:.1f}%)")
    
    return {
        "sicak_sayilar": sayac.most_common(5),
        "super_star": super_star_sayac.most_common(3),
        "tek_cift": (tek, cift)
    }

def en_iyi_kuponlari_olustur(monte_carlo_sonuclari, markov_sonuclari, istatistik_sonuclari, n_kupon=5):
    """
    Farklı analiz sonuçlarına dayanarak en iyi 5 kupon oluşturur
    """
    kuponlar = []
    
    # 2025 yılı analizini al
    yil_2025_analiz = yil_bazli_analiz(2025)
    
    # 1. Kupon: Sıcak sayılar + Markov tahminleri kombinasyonu
    sicak_sayilar = [sayi for sayi, _ in en_cok_cikan_sayilar()[:10]]
    markov_tahminler = markov_sonuclari['markov_tahminleri']['tahminler']
    kupon1 = sorted(list(set(sicak_sayilar[:3] + markov_tahminler[:3])))[:6]
    kuponlar.append({
        'sayilar': kupon1,
        'super_star': super_star_analizi()[0][0],  # En çok çıkan Super Star
        'strateji': 'Sıcak Sayılar + Markov Tahminleri',
        'guven_skoru': markov_sonuclari['markov_tahminleri']['guven_skoru']
    })
    
    # 2. Kupon: Monte Carlo'nun en başarılı stratejisi + 2025 sıcak sayıları
    en_iyi_strateji = monte_carlo_sonuclari['strateji_sonuclari'][
        monte_carlo_sonuclari['genel_degerlendirme']['en_iyi_strateji']]
    if yil_2025_analiz:
        sicak_2025 = [sayi for sayi, _ in yil_2025_analiz['sicak_sayilar']]
        kupon2_sayilar = sorted(en_iyi_strateji['en_basarili_kombinasyonlar']['kombinasyon_1']['sayilar'])
        kupon2 = sorted(list(set(kupon2_sayilar[:3] + sicak_2025[:3])))[:6]
        super_star_2025 = yil_2025_analiz['super_star'][0][0] if yil_2025_analiz['super_star'] else super_star_analizi()[1][0]
    else:
        kupon2 = sorted(en_iyi_strateji['en_basarili_kombinasyonlar']['kombinasyon_1']['sayilar'])
        super_star_2025 = super_star_analizi()[1][0]
    
    kuponlar.append({
        'sayilar': kupon2,
        'super_star': super_star_2025,
        'strateji': 'Monte Carlo + 2025 Sıcak Sayılar',
        'guven_skoru': monte_carlo_sonuclari['strateji_sonuclari']['sicak_sayilar']['kazanma_olasiliklari']['altili']
    })
    
    # 3. Kupon: Dengeli dağılım
    dusuk_sayilar = list(range(1, 46))
    yuksek_sayilar = list(range(46, 91))
    kupon3 = sorted(
        random.sample(dusuk_sayilar, 3) +  # 3 düşük sayı
        random.sample(yuksek_sayilar, 3)   # 3 yüksek sayı
    )
    kuponlar.append({
        'sayilar': kupon3,
        'super_star': super_star_analizi()[2][0],
        'strateji': 'Dengeli Dağılım',
        'guven_skoru': monte_carlo_sonuclari['strateji_sonuclari']['dengeli']['kazanma_olasiliklari']['altili']
    })
    
    # 4. Kupon: En çok birlikte çıkan sayılar
    birlikte_cikanlar = [sayi for ikili, _ in en_cok_birlikte_cikan_sayilar() for sayi in ikili]
    kupon4 = sorted(list(set(birlikte_cikanlar)))[:6]
    kuponlar.append({
        'sayilar': kupon4,
        'super_star': super_star_analizi()[3][0],
        'strateji': 'En Çok Birlikte Çıkan Sayılar',
        'guven_skoru': monte_carlo_sonuclari['strateji_sonuclari']['rastgele']['kazanma_olasiliklari']['altili']  # Değiştirildi
    })
    
    # 5. Kupon: Makine öğrenmesi tahminleri
    ml_tahmin = makine_ogrenmesi_tahmin()
    kupon5 = sorted([int(x) for x in ml_tahmin])
    kuponlar.append({
        'sayilar': kupon5,
        'super_star': super_star_analizi()[4][0],
        'strateji': 'Makine Öğrenmesi Tahmini',
        'guven_skoru': monte_carlo_sonuclari['strateji_sonuclari']['sicak_sayilar']['kazanma_olasiliklari']['altili']  # Değiştirildi
    })
    
    # Kuponları yazdır
    print("\nÖnerilen Kuponlar:")
    for i, kupon in enumerate(kuponlar, 1):
        print(f"\nKupon {i} ({kupon['strateji']}):")
        print(f"Sayılar: {kupon['sayilar']}")
        print(f"Super Star: {kupon['super_star']}")
        print(f"Güven Skoru: {kupon['guven_skoru']}")
    
    return kuponlar

def main():
    print("1. En çok çıkan 10 sayı:")
    for sayi, tekrar in en_cok_cikan_sayilar():
        print(f"Sayı: {sayi}, Tekrar: {tekrar}")
    
    print("\n2. Çıkış örüntüsü analizi (Bir sayıdan sonra en çok gelen sayılar):")
    for sayi, tekrar in cikis_oruntusu_analizi():
        print(f"Sayı: {sayi}, Tekrar: {tekrar}")
    
    print("\n3. En çok birlikte çıkan sayı ikililer:")
    for (sayi1, sayi2), tekrar in en_cok_birlikte_cikan_sayilar():
        print(f"Sayılar: {sayi1}-{sayi2}, Tekrar: {tekrar}")
    
    print(f"\n4. Ardışık sayıların toplam görülme sayısı: {ardisik_sayilar_analizi()}")
    
    print("\n5. Sayı grupları dengesi:")
    for grup, sayi in sayi_gruplari_dengesi().items():
        print(f"{grup}: {sayi}")
    
    print("\n6. Tek-Çift analizi:")
    for tip, sayi in tek_cift_analizi().items():
        print(f"{tip}: {sayi}")
    
    print("\n7. En çok çıkan Super Star sayıları:")
    for sayi, tekrar in super_star_analizi():
        print(f"Sayı: {sayi}, Tekrar: {tekrar}")
    
    print("\n8. Hiç çıkmamış sayılar:")
    print(cikmamis_sayilar())
    
    print("\n9. Günlere göre dağılım:")
    sanslı_gun_analizi()
    
    # Grafikleri çiz
    grafikleri_ciz()
    
    print("\n10. Makine Öğrenmesi Tahmini:")
    tahmin = makine_ogrenmesi_tahmin()
    print(f"Bir sonraki çekiliş için tahmin edilen sayılar: {tahmin}")
    
    print("\n11. Monte Carlo Simülasyonu Sonuçları:")
    for kombinasyon, tekrar in monte_carlo_simulasyonu():
        print(f"Kombinasyon: {kombinasyon}, Tekrar: {tekrar}")
    
    # Görsel analizler
    zaman_serisi_analizi()
    mevsimsel_analiz()
    korelasyon_analizi()
    
    print("\n12. En çok birlikte çıkan üçlü gruplar:")
    uclu_gruplar, dortlu_gruplar = coklu_grup_analizi()
    for (sayi1, sayi2, sayi3), tekrar in uclu_gruplar:
        print(f"Sayılar: {sayi1}-{sayi2}-{sayi3}, Tekrar: {tekrar}")
    
    print("\n13. En çok birlikte çıkan dörtlü gruplar:")
    for (sayi1, sayi2, sayi3, sayi4), tekrar in dortlu_gruplar:
        print(f"Sayılar: {sayi1}-{sayi2}-{sayi3}-{sayi4}, Tekrar: {tekrar}")
    
    print("\n14. Yıllık Analiz:")
    # yillik_analiz() fonksiyonunu kaldır
    
    print("\n15. Gelişmiş Tahmin Analizi:")
    gelismis_tahmin_analizi()
    
    print("\n16. Tekrar Eden Gruplar Analizi:")
    tekrar_analizi = tekrar_eden_gruplar_analizi()
    
    # Yorumlayıcıyı oluştur
    yorumlayici = LotoYorumlayici()
    
    # Analiz sonuçlarını güncelle
    analiz_sonuclari = {
        "en_cok_cikan_sayilar": {
            str(sayi): tekrar for sayi, tekrar in en_cok_cikan_sayilar()
        },
        "tek_cift_analizi": {
            "tek": f"{(tek_cift_analizi()['Tek Sayılar'] / (tek_cift_analizi()['Tek Sayılar'] + tek_cift_analizi()['Çift Sayılar']) * 100):.1f}%",
            "cift": f"{(tek_cift_analizi()['Çift Sayılar'] / (tek_cift_analizi()['Tek Sayılar'] + tek_cift_analizi()['Çift Sayılar']) * 100):.1f}%"
        },
        "sicak_sayilar": [sayi for sayi, _ in sicak_soguk_sayilar()[0][:5]],
        "en_cok_birlikte": {
            f"{sayi1}-{sayi2}": tekrar 
            for (sayi1, sayi2), tekrar in en_cok_birlikte_cikan_sayilar()
        },
        "ardisik_sayilar": ardisik_sayilar_analizi(),
        "super_star_analizi": {
            str(sayi): tekrar for sayi, tekrar in super_star_analizi()[:5]
        },
        "sayi_gruplari_dengesi": sayi_gruplari_dengesi(),
        "uclu_gruplar": {
            f"{sayi1}-{sayi2}-{sayi3}": tekrar 
            for (sayi1, sayi2, sayi3), tekrar in coklu_grup_analizi()[0][:3]
        },
        "tekrar_eden_gruplar": {
            "ikili": {f"{g[0]}-{g[1]}": t for g, t in tekrar_analizi['ikili_gruplar'] if t > 2},
            "uclu": {f"{g[0]}-{g[1]}-{g[2]}": t for g, t in tekrar_analizi['uclu_gruplar'] if t > 1},
            "dortlu": {f"{g[0]}-{g[1]}-{g[2]}-{g[3]}": t for g, t in tekrar_analizi['dortlu_gruplar'] if t > 1}
        }
    }
    
    print("\n17. Sıcak Sayılar İlişki Analizi:")
    sicak_iliskiler = sicak_sayilar_iliskisi()
    
    # Analiz sonuçlarına ekle
    analiz_sonuclari.update({
        "sicak_sayilar_iliskisi": {
            "birlikte_cikanlar": {f"{s1}-{s2}": t for s1, s2, t in sicak_iliskiler['birlikte_cikma'][:5]},
            "ardisik_cikanlar": {str(s): t for s, t in sicak_iliskiler['ardisik_cikma'][:5]},
            "uclu_gruplar": {f"{s1}-{s2}-{s3}": t for s1, s2, s3, t in sicak_iliskiler['uclu_gruplar'][:3]}
        }
    })
    
    print("\n18. Yıllara Göre Detaylı Analiz:")
    yillik_detay = detayli_yillik_analiz()
    
    # Analiz sonuçlarına ekle
    analiz_sonuclari.update({
        "yillik_analizler": {
            str(yil): {
                "sicak_sayilar": {str(sayi): tekrar for sayi, tekrar in analiz['sicak_sayilar'][:5]},
                "grup_analizleri": {
                    "ikili": {f"{s1}-{s2}": t for (s1, s2), t in analiz['grup_analizleri']['tekrar_eden_ikili'][:3]},
                    "uclu": {f"{s1}-{s2}-{s3}": t for (s1, s2, s3), t in analiz['grup_analizleri']['tekrar_eden_uclu'][:2]},
                    "ardisik_ikili": {f"{s1}-{s2}": t for (s1, s2), t in analiz['grup_analizleri']['ardisik_ikili'][:2]},
                    "ardisik_uclu": {f"{s1}-{s2}-{s3}": t for (s1, s2, s3), t in analiz['grup_analizleri']['ardisik_uclu'][:1]}
                },
                "super_star": {str(sayi): tekrar for sayi, tekrar in analiz['super_star'][:3]},
                "tek_cift": f"{(analiz['tek_cift_orani'][0]/(analiz['tek_cift_orani'][0]+analiz['tek_cift_orani'][1]))*100:.1f}%",
            }
            for yil, analiz in yillik_detay.items()
        }
    })
    
    print("\nAnaliz 1: Sayıların Beklenen Değerleri ve Varyansı")
    istatistik_sonuclari = sayisal_istatistikler()
    
    # Analiz sonuçlarını güncelle
    analiz_sonuclari.update({
        "sayisal_istatistikler": istatistik_sonuclari
    })
    
    print("\nAnaliz 2: Ki-Kare Testi")
    ki_kare_sonuclari = ki_kare_testi()
    
    # Analiz sonuçlarını güncelle
    analiz_sonuclari.update({
        "ki_kare_analizi": ki_kare_sonuclari
    })
    
    print("\nAnaliz 3: Korelasyon Analizi")
    korelasyon_sonuclari = korelasyon_analizi_detayli()
    
    # Analiz sonuçlarını güncelle
    analiz_sonuclari.update({
        "korelasyon_analizi": korelasyon_sonuclari
    })
    
    print("\nAnaliz 4: Zaman Serisi Analizi")
    zaman_serisi_sonuclari = zaman_serisi_analizi_detayli()
    
    # Analiz sonuçlarını güncelle
    analiz_sonuclari.update({
        "zaman_serisi_analizi": zaman_serisi_sonuclari
    })
    
    print("\nAnaliz 5: Olasılık Dağılımları")
    olasilik_sonuclari = olasilik_dagilim_analizi()
    
    # Analiz sonuçlarını güncelle
    analiz_sonuclari.update({
        "olasilik_dagilim_analizi": olasilik_sonuclari
    })
    
    try:
        print("\nYapay Zeka Analizi ve Tahminler:")
        yorum = yorumlayici.analiz_yorumla(analiz_sonuclari)
        print(yorum)
    except Exception as e:
        print("Yapay zeka analizi sırasında hata oluştu:", str(e))
    
    print("\nAnaliz 6: Monte Carlo Simülasyonları")
    monte_carlo_sonuclari = monte_carlo_analizi_detayli()
    
    # Analiz sonuçlarını güncelle
    analiz_sonuclari.update({
        "monte_carlo_analizi": monte_carlo_sonuclari
    })
    
    print("\nAnaliz 7: Markov Zinciri Analizi")
    markov_sonuclari = markov_zinciri_analizi()
    
    # Analiz sonuçlarını güncelle
    analiz_sonuclari.update({
        "markov_zinciri_analizi": markov_sonuclari
    })
    
    # Monte Carlo sonuçlarını yazdır
    print("\nMonte Carlo Analizi Sonuçları:")
    print(f"En İyi Strateji: {monte_carlo_sonuclari['genel_degerlendirme']['en_iyi_strateji']}")
    print(f"En Kötü Strateji: {monte_carlo_sonuclari['genel_degerlendirme']['en_kotu_strateji']}")
    
    # En son kısma ekle:
    print("\nEn İyi 5 Kupon Önerisi:")
    en_iyi_kuponlar = en_iyi_kuponlari_olustur(
        monte_carlo_sonuclari,
        markov_sonuclari,
        analiz_sonuclari['sayisal_istatistikler']
    )
    
    # 2025 yılı özel analizi
    print("\n2025 Yılı Özel Analizi:")
    yil_2025_sonuclari = yil_bazli_analiz(2025)
    
    # En iyi kuponları oluştur
    print("\nEn İyi 5 Kupon Önerisi:")
    en_iyi_kuponlar = en_iyi_kuponlari_olustur(
        monte_carlo_sonuclari,
        markov_sonuclari,
        analiz_sonuclari['sayisal_istatistikler']
    )

if __name__ == "__main__":
    main()
