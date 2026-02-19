import streamlit as st
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import pandas as pd
import requests
from datetime import datetime, timedelta
import pyield as py  # pip install pyield (B3 public data)
import requests
import zipfile
import io

# ====================== BLACK-76 (Options on Futures) ======================


def black76_call(F: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-76 European call on futures (discount factor e^{-rT} applied to forward value)."""
    if T <= 0 or sigma <= 0:
        return max(F - K, 0) * np.exp(-r * T)
    d1 = (np.log(F / K) + (sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))


def black76_put(F: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-76 European put."""
    if T <= 0 or sigma <= 0:
        return max(K - F, 0) * np.exp(-r * T)
    d1 = (np.log(F / K) + (sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def black76_iv(market_price: float, F: float, K: float, T: float, r: float,
               option_type: str = "call", tol: float = 1e-8) -> float:
    """Solve for implied volatility using Brentq (guaranteed convergence)."""
    def objective(sigma):
        if option_type.lower() == "call":
            return black76_call(F, K, T, r, sigma) - market_price
        else:
            return black76_put(F, K, T, r, sigma) - market_price

    # Bracket [0.001, 5.0] covers 0.1 % → 500 % vol
    try:
        return brentq(objective, 1e-4, 5.0, xtol=tol)
    except:
        return np.nan  # No solution (deep ITM/OTM or bad price)

# ====================== DATA FETCH (Public/Free) ======================


@st.cache_data(ttl=300)
def get_dol_futures():
    from datetime import datetime
    today_str = datetime.now().strftime("%d-%m-%Y")
    
    try:
        import pyield as py
        pl_df = py.futures(today_str, "DOL")
        
        if pl_df is None or pl_df.is_empty():
            st.warning("pyield DOL vazio → tentando WDO (mini dólar)")
            pl_df = py.futures(today_str, "WDO")
        
        if pl_df is None or pl_df.is_empty():
            raise ValueError("No DOL/WDO data today from pyield")
        
        df = pl_df.to_pandas()
        
        # Debug: mostre colunas e primeiras linhas no app
        st.write("DEBUG pyield - Colunas:", list(df.columns))
        st.write("DEBUG pyield - Primeiras linhas:", df.head(3))
        
        # Coluna de expiração (confirmada na doc: 'ExpirationDate')
        if 'ExpirationDate' in df.columns:
            df["Expiration"] = pd.to_datetime(df["ExpirationDate"])
        else:
            # Fallback se nome mudar (raro)
            exp_candidates = [c for c in df.columns if "expir" in c.lower() or "venc" in c.lower()]
            if exp_candidates:
                df["Expiration"] = pd.to_datetime(df[exp_candidates[0]])
            else:
                raise KeyError("Coluna de expiration não encontrada")
        
        df = df.sort_values("Expiration")
        active_df = df[df["Expiration"] > pd.Timestamp.now()]
        
        if active_df.empty:
            raise ValueError("No active futures found")
        
        return active_df
    
    except Exception as e:
        st.error(f"Erro pyield: {str(e)}. Usando inputs manuais.")
        return pd.DataFrame()  # fallback para manual


@st.cache_data(ttl=3600)  # 1 hora de cache, Selic muda pouco
def get_selic_rate(days_ahead: int = 0) -> float:
    """BCB Selic anualizada (série 11). Retorna em decimal (ex: 0.1175 para 11.75%)."""
    from datetime import datetime, timedelta
    import requests
    import pandas as pd

    # Calcula data final um pouco à frente para garantir dados
    today = datetime.now().date()
    end_date = (today + timedelta(days=days_ahead + 60)).strftime("%d/%m/%Y")
    start_date = (today - timedelta(days=365)).strftime("%d/%m/%Y")  # últimos 12 meses

    url = (
        f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.11/dados"
        f"?formato=json&dataInicial={start_date}&dataFinal={end_date}"
    )

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()  # Erro se != 200
        data = resp.json()

        if not data:
            raise ValueError("Resposta vazia da API BCB")

        df = pd.DataFrame(data)
        df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
        df = df.sort_values("data", ascending=True)

        # Converte 'valor' de string para float (ex: "11.75" → 11.75)
        df["valor"] = df["valor"].astype(float)

        latest_rate = df.iloc[-1]["valor"] / 100.0  # % → decimal (0.1175)
        return latest_rate

    except Exception as e:
        st.error(f"Erro ao buscar Selic: {str(e)}. Usando fallback 10.5%.")
        return 0.105  # Fallback conservador (ajuste conforme mercado atual)

def fetch_b3_dol_settlement_fallback(date_str):  # date_str = '2026-02-19'
    # URL padrão B3 para settlements diários (ajuste se mudar; cheque https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/historico/derivativos/)
    # Exemplo real: https://www.b3.com.br/data/files/... mas use busca ou known pattern
    # Para simplicidade, use endpoint conhecido ou scrape a página de download
    
    # Placeholder: baixe manual o ZIP mais recente de "Ajustes Diários" em https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/historico/derivativos/
    # Ou implemente full parser:
    url_example = "https://www.b3.com.br/data/files/XX/XX/XX/XX/ArquivosDerivativos/...zip"  # substitua pelo link real
    
    # Para testar: rode local e encontre o link atual via browser (inspecione na página de histórico)
    # Alternativa rápida: use ADVFN ou InfoMoney scrape para F atual (mais fácil para MVP)
    
    st.info("Fallback: Implementar parser B3 ZIP settlement aqui (próximo passo)")
    return None  # por enquanto


# ====================== STREAMLIT UI ======================
st.set_page_config(page_title="DOL IV Analyzer", layout="wide")
st.title("🟢 DOL IV & Fair Price Analyzer (B3 BRL/USD Futures Options)")
st.markdown("**Public data only • Black-76 • ATM auto-select**")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Live Data (pyield + BCB)")
    futures_df = get_dol_futures()
    if not futures_df.empty:
        st.dataframe(futures_df[["TickerSymbol", "SettlementRate", "Expiration"]].head(
            8), hide_index=True)

        # Nearest active future (default front-month)
        active = futures_df[futures_df["Expiration"] > datetime.now()].iloc[0]
        F = active["SettlementRate"]
        exp_date = active["Expiration"]
        T = (exp_date - datetime.now()).days / 365.25  # Actual/365.25
        st.success(f"**Underlying F** = {F:,.4f} | T = {T*365:.1f} days")
    else:
        F = st.number_input("Manual F (BRL/USD)", value=5.70, step=0.0001)
        T = st.number_input("T (years)", value=0.25, step=0.01)

    r = get_selic_rate(int(T*365))
    st.info(f"**Risk-free r** (Selic) = {r*100:.3f}%")

with col2:
    st.subheader("2. ATM Option Selection")
    option_type = st.selectbox("Type", ["Call", "Put"])
    K_input = st.number_input("Strike (auto ATM if 0)", value=0.0, step=0.0001)

    market_price = st.number_input(
        "Market Price (BRL per contract point)", value=0.0150, step=0.0001, format="%.4f")

# Auto ATM logic
if K_input == 0:
    K = round(F / 0.0005) * 0.0005   # B3 tick 0.0005 for most series
else:
    K = K_input

st.write(f"**ATM Strike used** = {K:,.4f} (closest to F)")

# ====================== CALCULATION ======================
if st.button("Calculate IV & Fair Price", type="primary"):
    iv = black76_iv(market_price, F, K, T, r, option_type.lower())

    if np.isnan(iv):
        st.error("No IV solution – check inputs")
    else:
        theo = black76_call(F, K, T, r, iv) if option_type.lower(
        ) == "call" else black76_put(F, K, T, r, iv)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Implied Volatility", f"{iv*100:.2f}%")
        col_b.metric("Theoretical Fair Price", f"{theo:.4f}")
        col_c.metric(
            "Mispricing", f"{market_price - theo:+.4f} ({(market_price - theo)/theo*100:+.2f}%)")

        # Greeks (bonus – as per your quant spec)
        d1 = (np.log(F/K) + (iv**2/2)*T) / (iv*np.sqrt(T))
        delta = np.exp(-r*T) * norm.cdf(d1) if option_type.lower() == "call" else - \
            np.exp(-r*T)*norm.cdf(-d1)
        gamma = np.exp(-r*T) * norm.pdf(d1) / (F * iv * np.sqrt(T))
        vega = F * np.exp(-r*T) * norm.pdf(d1) * np.sqrt(T)   # per 1% vol
        st.markdown("**Greeks**")
        st.write(
            f"Δ = {delta:.4f} | Γ = {gamma:.6f} | ν (1%) = {vega/100:.4f}")

# ====================== DATA NOTES (Golden Rule compliance) ======================
st.caption("""
**Data Sourcing (public & free)**  
• Futures F & expirations → `pyield.futures()` (pulls B3 settlement TXT/ZIPs)  
• r → BCB API (Selic series 11)  
• Option market prices → Daily B3 "Boletim Diário – Derivativos" (ZIP/TXT) or ADVFN scrape fallback  
• For live chain: add Selenium on https://br.advfn.com/investimentos/opcoes/dolar or B3 public quotes  
""")

st.markdown("---")
st.markdown("**Next steps you requested**: full options-chain parser from B3 boletim, React/Vue dashboard, Greeks surface plot, backtesting module.")






