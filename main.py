import yfinance as yf
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

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
# ─────────────────────────────────────────
def feature_hazirla(df):
    df = df.copy()

    df['gun_yonu']         = (df['Close'] > df['Open']).astype(int)
    df['ardisik_yukselis'] = df['gun_yonu'].rolling(5).sum().shift(1)

    oc_fark = (df['Close'] - df['Open']) / df['Open']
    df['oc_zscore']        = ((oc_fark - oc_fark.rolling(20).mean()) /
                               (oc_fark.rolling(20).std() + 1e-9)).shift(1)

    df['haftanin_gunu']    = pd.to_datetime(df.index).dayofweek
    df['ay_basi']          = (pd.to_datetime(df.index).day <= 3).astype(int)
    df['ay_sonu']          = (pd.to_datetime(df.index).day >= 28).astype(int)

    yillik_max = df['Close'].rolling(252).max().shift(1)
    yillik_min = df['Close'].rolling(252).min().shift(1)
    df['yuksege_uzaklik']  = ((df['Close'].shift(1) - yillik_min) /
                               (yillik_max - yillik_min + 1e-9))

    vol_mean = df['Volume'].rolling(20).mean()
    vol_std  = df['Volume'].rolling(20).std()
    df['hacim_anomali']    = ((df['Volume'] - vol_mean) /
                               (vol_std + 1e-9)).shift(1)

    df['return_1']         = df['Close'].pct_change(1).shift(1)
    df['return_3']         = df['Close'].pct_change(3).shift(1)
    df['return_5']         = df['Close'].pct_change(5).shift(1)

    df['kapanis_baskisi']  = ((df['Close'] - df['Low']) /
                               (df['High'] - df['Low'] + 1e-9)).shift(1)

    df['open_gap']         = ((df['Open'] - df['Close'].shift(1)) /
                               (df['Close'].shift(1) + 1e-9)).shift(1)

    df['rsi']              = rsi(df['Close']).shift(1)
    df['macd_hist']        = macd_sinyal(df['Close']).shift(1)
    df['boll_poz']         = bollinger_pozisyon(df['Close']).shift(1)

    df['usdtry_return_1']   = df['usdtry'].pct_change(1).shift(1)
    df['usdtry_return_3']   = df['usdtry'].pct_change(3).shift(1)
    df['usdtry_volatilite'] = df['usdtry'].pct_change().rolling(10).std().shift(1)
    df['usdtry_trend']      = (df['usdtry'].shift(1) >
                                df['usdtry'].rolling(10).mean().shift(1)).astype(int)
    df['dolar_bist_kor']    = (df['Close'].pct_change()
                                .rolling(20)
                                .corr(df['usdtry'].pct_change())
                                .shift(1))

    df['hedef'] = (df['Close'] > df['Open']).astype(int)
    return df.dropna()

FEATURES = [
    'ardisik_yukselis', 'oc_zscore', 'haftanin_gunu',
    'ay_basi', 'ay_sonu', 'yuksege_uzaklik', 'hacim_anomali',
    'return_1', 'return_3', 'return_5', 'kapanis_baskisi', 'open_gap',
    'rsi', 'macd_hist', 'boll_poz',
    'usdtry_return_1', 'usdtry_return_3', 'usdtry_volatilite',
    'usdtry_trend', 'dolar_bist_kor'
]

# ─────────────────────────────────────────
# CROSS VALIDATION
# ─────────────────────────────────────────
def cross_val_acc(model_class, model_params, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for tr, te in tscv.split(X):
        m = model_class(**model_params)
        m.fit(X.iloc[tr], y.iloc[tr])
        scores.append(accuracy_score(y.iloc[te], m.predict(X.iloc[te])))
    return round(np.mean(scores) * 100, 1)

# ─────────────────────────────────────────
# MODEL 1 — PROPHET
# ─────────────────────────────────────────
def prophet_tahmin(df_raw):
    try:
        df_kisa = df_raw.tail(180).copy()
        prop_df = df_kisa[['Close', 'usdtry']].reset_index()
        prop_df.columns = ['ds', 'y', 'usdtry']
        prop_df['ds'] = pd.to_datetime(prop_df['ds']).dt.tz_localize(None)

        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.5
        )
        model.add_regressor('usdtry')
        model.fit(prop_df)

        gelecek = model.make_future_dataframe(periods=1, freq='B')
        son_usdtry = float(df_kisa['usdtry'].iloc[-1])
        gelecek['usdtry'] = prop_df['usdtry'].tolist() + [son_usdtry]

        forecast = model.predict(gelecek)
        son = forecast.iloc[-1]

        bugun_open   = float(df_raw['Open'].iloc[-1])
        tahmin_fiyat = float(son['yhat'])
        degisim      = (tahmin_fiyat - bugun_open) / bugun_open * 100

        return {
            'yon'    : 1 if tahmin_fiyat > bugun_open else 0,
            'degisim': round(degisim, 2),
            'fiyat'  : round(tahmin_fiyat, 2)
        }
    except:
        return None

# ─────────────────────────────────────────
# MODEL 2 — LIGHTGBM
# ─────────────────────────────────────────
def lgbm_tahmin(df):
    X       = df[FEATURES].iloc[:-1]
    y       = df['hedef'].iloc[:-1]
    bugun_X = df[FEATURES].iloc[[-1]]

    params = dict(
        n_estimators=200, max_depth=3,
        learning_rate=0.03, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1
    )

    cv_acc = cross_val_acc(LGBMClassifier, params, X, y)
    model  = LGBMClassifier(**params)
    model.fit(X, y)

    tahmin   = model.predict(bugun_X)[0]
    olasilik = float(model.predict_proba(bugun_X)[0][tahmin])

    return {
        'yon'   : int(tahmin),
        'guven' : round(olasilik * 100, 1),
        'cv_acc': cv_acc
    }

# ─────────────────────────────────────────
# MODEL 3 — XGBOOST
# ─────────────────────────────────────────
def xgb_tahmin(df):
    X       = df[FEATURES].iloc[:-1]
    y       = df['hedef'].iloc[:-1]
    bugun_X = df[FEATURES].iloc[[-1]]

    params = dict(
        n_estimators=200, max_depth=3,
        learning_rate=0.03, min_child_weight=20,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0,
        eval_metric='logloss'
    )

    cv_acc = cross_val_acc(XGBClassifier, params, X, y)
    model  = XGBClassifier(**params)
    model.fit(X, y)

    tahmin   = model.predict(bugun_X)[0]
    olasilik = float(model.predict_proba(bugun_X)[0][int(tahmin)])

    return {
        'yon'   : int(tahmin),
        'guven' : round(olasilik * 100, 1),
        'cv_acc': cv_acc
    }

# ─────────────────────────────────────────
# ANA DÖNGÜ
# ─────────────────────────────────────────
print("📊 BIST30 Tarama — Prophet + LightGBM + XGBoost\n")
print(f"{'Hisse':<8} | {'Open':>10} | {'Prophet':>10} | {'LightGBM':>10} | {'XGBoost':>10} | {'Karar'}")
print("─" * 85)

sonuclar = []
usdtry_indirildi = False

for ticker in BIST30:
    isim = ticker.replace('.IS', '')
    df_raw = veri_cek_hisse(ticker)
    if df_raw is None:
        print(f"{isim:<8} | ❌ Veri alınamadı")
        continue

    try:
        df = feature_hazirla(df_raw)
        bugun_open = float(df_raw['Open'].iloc[-1])

        prophet = prophet_tahmin(df_raw)
        lgbm    = lgbm_tahmin(df)
        xgb     = xgb_tahmin(df)

        if not prophet:
            continue

        # Oylar
        oylar    = [prophet['yon'], lgbm['yon'], xgb['yon']]
        yukselis = sum(oylar)

        if yukselis == 3:
            karar = '🟢 GÜÇLÜ YÜKSELİŞ'
            guc   = 3
        elif yukselis == 2:
            karar = '🟡 ORTA YÜKSELİŞ'
            guc   = 2
        elif yukselis == 1:
            karar = '🟡 ORTA DÜŞÜŞ'
            guc   = 2
        else:
            karar = '🔴 GÜÇLÜ DÜŞÜŞ'
            guc   = 3

        p_str = '🟢' if prophet['yon'] == 1 else '🔴'
        l_str = '🟢' if lgbm['yon'] == 1    else '🔴'
        x_str = '🟢' if xgb['yon'] == 1     else '🔴'

        print(
            f"{isim:<8} | {bugun_open:>10.2f} | "
            f"{p_str} {prophet['degisim']:>+5.1f}%  | "
            f"{l_str} %{lgbm['guven']:<6} | "
            f"{x_str} %{xgb['guven']:<6} | "
            f"{karar}"
        )

        sonuclar.append({
            'Hisse'     : isim,
            'Open'      : bugun_open,
            'Prophet'   : prophet['yon'],
            'P_degisim' : prophet['degisim'],
            'LightGBM'  : lgbm['yon'],
            'L_guven'   : lgbm['guven'],
            'L_acc'     : lgbm['cv_acc'],
            'XGBoost'   : xgb['yon'],
            'X_guven'   : xgb['guven'],
            'X_acc'     : xgb['cv_acc'],
            'Karar'     : karar,
            'Guc'       : guc
        })

    except Exception as e:
        print(f"{isim:<8} | ❌ Hata: {e}")

# ─────────────────────────────────────────
# ÖZET
# ─────────────────────────────────────────
df_s = pd.DataFrame(sonuclar)

guclu_yukselis = df_s[df_s['Karar'] == '🟢 GÜÇLÜ YÜKSELİŞ'].sort_values('L_guven', ascending=False)
guclu_dusus    = df_s[df_s['Karar'] == '🔴 GÜÇLÜ DÜŞÜŞ'].sort_values('L_guven', ascending=False)
orta_yukselis  = df_s[df_s['Karar'] == '🟡 ORTA YÜKSELİŞ'].sort_values('L_guven', ascending=False)
orta_dusus     = df_s[df_s['Karar'] == '🟡 ORTA DÜŞÜŞ'].sort_values('L_guven', ascending=False)

print(f"\n{'='*85}")
print(f"📋 ÖZET — {pd.Timestamp.today().strftime('%d.%m.%Y')}")
print(f"{'='*85}")

print(f"\n🟢 GÜÇLÜ YÜKSELİŞ (3/3 hemfikir) — {len(guclu_yukselis)} hisse:")
if len(guclu_yukselis) > 0:
    for _, r in guclu_yukselis.iterrows():
        print(f"   {r['Hisse']:<8} | Open: {r['Open']:>10.2f} | Prophet: {r['P_degisim']:>+5.1f}% | LGBM Güven: %{r['L_guven']}")
else:
    print("   Yok")

print(f"\n🔴 GÜÇLÜ DÜŞÜŞ (3/3 hemfikir) — {len(guclu_dusus)} hisse:")
if len(guclu_dusus) > 0:
    for _, r in guclu_dusus.iterrows():
        print(f"   {r['Hisse']:<8} | Open: {r['Open']:>10.2f} | Prophet: {r['P_degisim']:>+5.1f}% | LGBM Güven: %{r['L_guven']}")
else:
    print("   Yok")

print(f"\n🟡 ORTA YÜKSELİŞ (2/3 hemfikir) — {len(orta_yukselis)} hisse:")
print(f"   {', '.join(orta_yukselis['Hisse'].tolist()) or 'Yok'}")

print(f"\n🟡 ORTA DÜŞÜŞ (2/3 hemfikir) — {len(orta_dusus)} hisse:")
print(f"   {', '.join(orta_dusus['Hisse'].tolist()) or 'Yok'}")

print(f"\n💡 Önce 'Güçlü' sinyallere odaklan, 'Orta' sinyaller ikincil.")
print(f"{'='*85}")