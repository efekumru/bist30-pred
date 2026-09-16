# bist30-pred

BIST30 hisseleri için **ertesi günün** kapanış yönünü (yukarı/aşağı) tahmin eden sistem. Staj proje açıklamasındaki ("Machine Learning Based Prediction Model for BIST Stocks") her cümleye birebir uyacak şekilde tasarlandı:

- **Hedef**: yarının kapanışı bugünün kapanışından yüksek mi? ("predicts... for the next day")
- **Modeller**: Logistic Regression, K-Nearest Neighbours, Decision Tree, XGBoost, LightGBM — proje açıklamasında adı geçen 5 algoritmanın tamamı
- **Değerlendirme**: her model için walk-forward (TimeSeriesSplit) cross-validation doğruluğu hesaplanır ve karşılaştırılır
- **Karar mekanizması**: sadece CV doğruluğu `BASARI_ESIGI` (%52) üzerinde olan ("başarılı") modellerin oyu sayılır; kararı bu modellerin çoğunluğu belirler — "only successful ones will be used" + "combination of the models" ifadelerinin doğrudan uygulanması
- **Çıktı**: sonuçlar `index.html` dashboard'unda gösterilir, Vercel'e deploy edilir

## Çalıştırma

Yahoo Finance (yfinance) erişimi gerektiği için Google Colab'da çalıştırılıyor.

```bash
pip install -q yfinance lightgbm xgboost
python main.py
```

Üretilen dosyalar:
- `data/bist30_tahmin.csv` — her hisse için 5 modelin de yön/güven/CV-accuracy'si + genel karar (dashboard bunu okuyor)
- `data/xu030_tahmin.json` — XU030 endeks özeti (dashboard bunu okuyor)
- `data/model_karsilastirma.csv` — BIST30 genelinde model bazında ortalama doğruluk (staj raporundaki "performance comparison across all tested algorithms" deliverable'ı için)

## Dashboard'u güncelleme

```bash
git add data/
git commit -m "tahmin guncellendi"
git push
```

Vercel, `main` branch'e her push'ta otomatik redeploy alır.

## Notlar

- `index-4.html`, `data/bist30_wave_signals.csv` adlı ayrı bir dosyayı bekliyor; bu dosya repoda hiç var olmadı, göz ardı edilebilir.
- Eski main.py sürümü (Prophet + LightGBM + XGBoost, aynı-gün hedef) staj proje açıklamasıyla örtüşmüyordu; bu sürüm başvurudaki tanıma göre yeniden yazıldı.
