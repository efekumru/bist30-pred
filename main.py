import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

XU030_TICKER = 'XU030.IS'

# "Only successful ones will be used in this system" — bir modelin oyu, walk-forward
# CV doğruluğu bu eşiği geçerse sayılır. Staj proje açıklamasındaki "only successful
# algorithms are used" ve "combination of the models" ifadelerini birebir uygular.
BASARI_ESIGI = 52.0  # %

BIST30 = [
    'AKBNK.IS', 'ARCLK.IS', 'ASELS.IS', 'BIMAS.IS', 'DOHOL.IS',
    'EKGYO.IS', 'EREGL.IS', 'FROTO.IS', 'GARAN.IS', 'GUBRF.IS',
    'HALKB.IS', 'ISCTR.IS', 'KCHOL.IS', 'KRDMD.IS', 'MGROS.IS',
    'ODAS.IS', 'PETKM.IS', 'PGSUS.IS', 'SAHOL.IS', 'SASA.IS',
    'SISE.IS', 'SOKM.IS', 'TAVHL.IS', 'TCELL.IS', 'THYAO.IS',
    'TKFEN.IS', 'TOASO.IS', 'TUPRS.IS'
]

# ─────────────────────────────────────────
# VERİ
# ─────────────────────────────────────────
def veri_cek_hisse(ticker):
    try:
        df = yf.download(ticker, period='5y', interval='1d', progress=False)
        usdtry = yf.download('USDTRY=X', period='5y', interval='1d', progress=False)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if isinstance(usdtry.columns, pd.MultiIndex):
            usdtry.columns = usdtry.columns.get_level_values(0)

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        usdtry = usdtry[['Close']].rename(columns={'Close': 'usdtry'}).dropna()

        df = df.join(usdtry, how='left').ffill().dropna()
        if len(df) < 200:
            return None
        return df
    except:
        return None

# ─────────────────────────────────────────
# TEKNİK İNDİKATÖRLER
# ─────────────────────────────────────────
def rsi(seri, period=14):
    delta = seri.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def macd_sinyal(seri, fast=12, slow=26, signal=9):
    ema_fast = seri.ewm(span=fast).mean()
    ema_slow = seri.ewm(span=slow).mean()
    macd     = ema_fast - ema_slow
    return macd - macd.ewm(span=signal).mean()

def bollinger_pozisyon(seri, period=20):
    ort = seri.rolling(period).mean()
    std = seri.rolling(period).std()
    return (seri - (ort - 2*std)) / (4*std + 1e-9)

# ─────────────────────────────────────────
# FEATURE MÜHENDİSLİĞİ
#
# Tüm feature'lar günün KAPANIŞINDA bilinen bilgiyi temsil eder (gün-içi ekstra
# gecikme yok) — çünkü hedef artık "yarının kapanışı" olduğundan, bugünün
# kapanışındaki her şeyi feature olarak kullanmak veri sızıntısı (leakage)
# yaratmaz. Tek istisna open_gap: bugünün açılışını dünün kapanışıyla kıyaslıyor,
# bu zaten kendi içinde 1 günlük bir gecikme taşıyor.
# ─────────────────────────────────────────
def feature_hazirla(df):
    df = df.copy()

    df['gun_yonu']         = (df['Close'] > df['Open']).astype(int)
    df['ardisik_yukselis'] = df['gun_yonu'].rolling(5).sum()

    oc_fark = (df['Close'] - df['Open']) / df['Open']
    df['oc_zscore']        = (oc_fark - oc_fark.rolling(20).mean()) / (oc_fark.rolling(20).std() + 1e-9)

    df['haftanin_gunu']    = pd.to_datetime(df.index).dayofweek
    df['ay_basi']          = (pd.to_datetime(df.index).day <= 3).astype(int)
    df['ay_sonu']          = (pd.to_datetime(df.index).day >= 28).astype(int)

    yillik_max = df['Close'].rolling(252).max()
    yillik_min = df['Close'].rolling(252).min()
    df['yuksege_uzaklik']  = (df['Close'] - yillik_min) / (yillik_max - yillik_min + 1e-9)

    vol_mean = df['Volume'].rolling(20).mean()
    vol_std  = df['Volume'].rolling(20).std()
    df['hacim_anomali']    = (df['Volume'] - vol_mean) / (vol_std + 1e-9)

    df['return_1']         = df['Close'].pct_change(1)
    df['return_3']         = df['Close'].pct_change(3)
    df['return_5']         = df['Close'].pct_change(5)

    df['kapanis_baskisi']  = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-9)
    df['open_gap']         = (df['Open'] - df['Close'].shift(1)) / (df['Close'].shift(1) + 1e-9)

    df['rsi']              = rsi(df['Close'])
    df['macd_hist']        = macd_sinyal(df['Close'])
    df['boll_poz']         = bollinger_pozisyon(df['Close'])

    df['usdtry_return_1']   = df['usdtry'].pct_change(1)
    df['usdtry_return_3']   = df['usdtry'].pct_change(3)
    df['usdtry_volatilite'] = df['usdtry'].pct_change().rolling(10).std()
    df['usdtry_trend']      = (df['usdtry'] > df['usdtry'].rolling(10).mean()).astype(int)
    df['dolar_bist_kor']    = df['Close'].pct_change().rolling(20).corr(df['usdtry'].pct_change())

    # HEDEF: yarının kapanışı bugünün kapanışından yüksek mi? ("for the next day")
    # Son satırda yarın henüz gerçekleşmediği için hedef NaN olur — bu satır
    # eğitimde kullanılmaz, sadece tahmin üretmek için tutulur.
    df['hedef'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    return df.dropna(subset=FEATURES)

FEATURES = [
    'ardisik_yukselis', 'oc_zscore', 'haftanin_gunu',
    'ay_basi', 'ay_sonu', 'yuksege_uzaklik', 'hacim_anomali',
    'return_1', 'return_3', 'return_5', 'kapanis_baskisi', 'open_gap',
    'rsi', 'macd_hist', 'boll_poz',
    'usdtry_return_1', 'usdtry_return_3', 'usdtry_volatilite',
    'usdtry_trend', 'dolar_bist_kor'
]

# ─────────────────────────────────────────
# CROSS VALIDATION (walk-forward)
# ─────────────────────────────────────────
def cross_val_acc(model_factory, model_params, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for tr, te in tscv.split(X):
        m = model_factory(**model_params)
        m.fit(X.iloc[tr], y.iloc[tr])
        scores.append(accuracy_score(y.iloc[te], m.predict(X.iloc[te])))
    return round(np.mean(scores) * 100, 1)

# ─────────────────────────────────────────
# MODELLER — staj proje açıklamasında adı geçen 5 algoritma:
# Logistic Regression, K-Nearest Neighbours, Decision Trees, XGBoost, LightGBM
# ─────────────────────────────────────────
def _lr_factory(**_):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))

def _knn_factory(**_):
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15))

def _dt_factory(**_):
    return DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=42)

LGBM_PARAMS = dict(
    n_estimators=200, max_depth=3,
    learning_rate=0.03, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbose=-1
)

XGB_PARAMS = dict(
    n_estimators=200, max_depth=3,
    learning_rate=0.03, min_child_weight=20,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbosity=0,
    eval_metric='logloss'
)

MODELS = {
    'LogisticRegression': (_lr_factory, {}),
    'KNN':                (_knn_factory, {}),
    'DecisionTree':       (_dt_factory, {}),
    'XGBoost':            (XGBClassifier, XGB_PARAMS),
    'LightGBM':           (LGBMClassifier, LGBM_PARAMS),
}

def model_tahmin(factory, params, X, y, bugun_X):
    cv_acc = cross_val_acc(factory, params, X, y)
    model  = factory(**params)
    model.fit(X, y)

    tahmin   = int(model.predict(bugun_X)[0])
    olasilik = float(model.predict_proba(bugun_X)[0][tahmin])

    return {'yon': tahmin, 'guven': round(olasilik * 100, 1), 'cv_acc': cv_acc}

def tum_modelleri_calistir(df):
    """5 modelin tümünü eğitir/değerlendirir ve tahminlerini döner."""
    X       = df[FEATURES].iloc[:-1]
    y       = df['hedef'].iloc[:-1].astype(int)
    bugun_X = df[FEATURES].iloc[[-1]]

    sonuclar = {}
    for isim, (factory, params) in MODELS.items():
        sonuclar[isim] = model_tahmin(factory, params, X, y, bugun_X)
    return sonuclar

def karar_ver(model_sonuclari):
    """'Only successful ones will be used' + 'combination of the models':
    sadece BASARI_ESIGI'ni geçen modellerin oyu sayılır; hiçbiri geçemezse
    tüm modellerin oyuna geri dönülür."""
    basarili = [r['yon'] for r in model_sonuclari.values() if r['cv_acc'] >= BASARI_ESIGI]
    if basarili:
        kullanilan = basarili
    else:
        kullanilan = [r['yon'] for r in model_sonuclari.values()]

    oran = sum(kullanilan) / len(kullanilan)

    if oran >= 0.8:
        return '🟢 STRONG UP', 3, oran, len(kullanilan)
    elif oran > 0.5:
        return '🟡 UP', 2, oran, len(kullanilan)
    elif oran <= 0.2:
        return '🔴 STRONG DOWN', 3, oran, len(kullanilan)
    elif oran < 0.5:
        return '🟡 DOWN', 2, oran, len(kullanilan)
    else:
        return '⚪ NEUTRAL', 1, oran, len(kullanilan)

# ─────────────────────────────────────────
# ANA DÖNGÜ
# ─────────────────────────────────────────
print("📊 BIST30 Tarama — LogisticRegression + KNN + DecisionTree + XGBoost + LightGBM")
print("   (Hedef: yarının kapanışı bugünün kapanışından yüksek mi?)\n")

sonuclar = []

for ticker in BIST30:
    isim = ticker.replace('.IS', '')
    df_raw = veri_cek_hisse(ticker)
    if df_raw is None:
        print(f"{isim:<8} | ❌ Veri alınamadı")
        continue

    try:
        df = feature_hazirla(df_raw)
        bugun_kapanis = float(df_raw['Close'].iloc[-1])

        model_sonuclari = tum_modelleri_calistir(df)
        karar, guc, oran, kullanilan_sayisi = karar_ver(model_sonuclari)

        print(f"{isim:<8} | Kapanis: {bugun_kapanis:>10.2f} | Oy oranı: %{oran*100:>5.1f} ({kullanilan_sayisi} model) | {karar}")
        for m_isim, r in model_sonuclari.items():
            yon_str = '🟢 UP  ' if r['yon'] == 1 else '🔴 DOWN'
            print(f"         · {m_isim:<18} {yon_str} | Güven: %{r['guven']:<5} | CV Acc: %{r['cv_acc']}")

        row = {'Hisse': isim, 'Kapanis': bugun_kapanis, 'Karar': karar, 'Guc': guc, 'OyOrani': round(oran*100, 1)}
        for m_isim, r in model_sonuclari.items():
            row[f'{m_isim}_yon']    = r['yon']
            row[f'{m_isim}_guven']  = r['guven']
            row[f'{m_isim}_cv_acc'] = r['cv_acc']
        sonuclar.append(row)

    except Exception as e:
        print(f"{isim:<8} | ❌ Hata: {e}")

df_s = pd.DataFrame(sonuclar)

# ─────────────────────────────────────────
# PERFORMANS KARŞILAŞTIRMASI — "a performance comparison across all tested
# algorithms" deliverable'ı: BIST30 genelinde model bazında ortalama doğruluk
# ─────────────────────────────────────────
print(f"\n{'='*85}")
print("📋 MODEL PERFORMANS KARŞILAŞTIRMASI (BIST30 ortalama walk-forward CV doğruluğu)")
print(f"{'='*85}")

model_karsilastirma = []
for m_isim in MODELS:
    ort_acc = df_s[f'{m_isim}_cv_acc'].mean()
    basarili_mi = ort_acc >= BASARI_ESIGI
    print(f"   {m_isim:<18} | Ortalama CV Acc: %{ort_acc:.1f} | {'✅ Başarılı (eşik %' + str(BASARI_ESIGI) + ')' if basarili_mi else '❌ Eşiğin altında'}")
    model_karsilastirma.append({'Model': m_isim, 'Ortalama_CV_Acc': round(ort_acc, 1), 'Basarili': basarili_mi})

df_karsilastirma = pd.DataFrame(model_karsilastirma)
karsilastirma_path = os.path.join(DATA_DIR, 'model_karsilastirma.csv')
df_karsilastirma.to_csv(karsilastirma_path, index=False, encoding='utf-8-sig')
print(f"\n💾 Yazıldı: {karsilastirma_path}")

# ─────────────────────────────────────────
# DASHBOARD ÇIKTISI — data/bist30_tahmin.csv
# ─────────────────────────────────────────
csv_path = os.path.join(DATA_DIR, 'bist30_tahmin.csv')
df_s.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"💾 Yazıldı: {csv_path} ({len(df_s)} satır)")

# ─────────────────────────────────────────
# XU030 ENDEKS TAHMİNİ — data/xu030_tahmin.json
# ─────────────────────────────────────────
print(f"\n📊 XU030 endeks tahmini hesaplanıyor...")
try:
    df_raw_idx = veri_cek_hisse(XU030_TICKER)
    if df_raw_idx is None:
        raise RuntimeError(f"{XU030_TICKER} için veri alınamadı")

    df_idx = feature_hazirla(df_raw_idx)
    bugun_kapanis_idx = float(df_raw_idx['Close'].iloc[-1])

    model_sonuclari_idx = tum_modelleri_calistir(df_idx)
    karar_idx, guc_idx, oran_idx, kullanilan_idx = karar_ver(model_sonuclari_idx)

    karar_idx_text = karar_idx.split(' ', 1)[1]  # emoji'yi at, "STRONG UP" vb. kalsın

    basarili_acc = [r['cv_acc'] for r in model_sonuclari_idx.values() if r['cv_acc'] >= BASARI_ESIGI]
    basarili_guven = [r['guven'] for r in model_sonuclari_idx.values() if r['cv_acc'] >= BASARI_ESIGI]
    if not basarili_acc:
        basarili_acc = [r['cv_acc'] for r in model_sonuclari_idx.values()]
        basarili_guven = [r['guven'] for r in model_sonuclari_idx.values()]

    endeks_json = {
        'tarih'      : pd.Timestamp.today().strftime('%Y-%m-%d'),
        'close'      : round(bugun_kapanis_idx, 2),
        'karar'      : karar_idx_text,
        'model_yon'  : 1 if oran_idx > 0.5 else 0,
        'model_guven': round(float(np.mean(basarili_guven)), 1),
        'model_acc'  : round(float(np.mean(basarili_acc)), 1),
        'model_detay': {m: r for m, r in model_sonuclari_idx.items()},
    }

    json_path = os.path.join(DATA_DIR, 'xu030_tahmin.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(endeks_json, f, ensure_ascii=False, indent=2)
    print(f"💾 Yazıldı: {json_path}")
    print(f"   Karar: {karar_idx_text} | Model yön: {'UP' if oran_idx > 0.5 else 'DOWN'} | Güven: %{endeks_json['model_guven']} | Acc: %{endeks_json['model_acc']}")

except Exception as e:
    print(f"❌ XU030 endeks tahmini başarısız: {e}")
    print(f"   (data/xu030_tahmin.json güncellenmedi)")

print(f"\n{'='*85}")
print("✅ Tamamlandı.")
print(f"{'='*85}")
