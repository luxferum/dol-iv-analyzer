import streamlit as st
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import pandas as pd
import requests
from datetime import datetime, timedelta
import pyield as py
from bs4 import BeautifulSoup
import pdfplumber
import re  # Para regex no parser PDF

# BLACK-76 functions (unchanged)
def black76_call(F: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(F - K, 0) * np.exp(-r * T)
    d1 = (np.log(F / K) + (sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))

def black76_put(F: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(K - F, 0) * np.exp(-r * T)
    d1 = (np.log(F / K) + (sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

def black76_iv(market_price: float, F: float, K: float, T: float, r: float, 
               option_type: str = "call", tol: float = 1e-8) -> float:
    def objective(sigma):
        if option_type.lower() == "call":
            return black76_call(F, K, T, r, sigma) - market_price
        else:
            return black76_put(F, K, T, r, sigma) - market_price
    try:
        return brentq(objective, 1e-4, 5.0, xtol=tol)
    except:
        return np.nan

# DATA SOURCES
@st.cache_data(ttl=300)
def get_dol_futures():
    today_str = datetime.now().strftime("%d-%m-%Y")
    try:
        pl_df = py.futures(today_str, "DOL")
        if pl_df is None or pl_df.is_empty():
            st.warning("pyield DOL vazio → tentando WDO")
            pl_df = py.futures(today_str, "WDO")
        if pl_df is None or pl_df.is_empty():
            raise ValueError("No DOL/WDO data today")
        
        df = pl_df.to_pandas()
        st.write("DEBUG pyield - Colunas:", list(df.columns))
        st.write("DEBUG pyield - Primeiras linhas:", df.head(3))
        
        # Colunas do log: ExpirationDate, TickerSymbol, LastRate (settlement)
        df["Expiration"] = pd.to_datetime(df["ExpirationDate"])
        df = df.sort_values("Expiration")
        active_df = df[df["Expiration"] > pd.Timestamp.now()]
        if active_df.empty:
            raise ValueError("No active futures")
        return active_df
    except Exception as e:
        st.error(f"Erro pyield: {str(e)}. Usando manual.")
        return pd.DataFrame()

def get_dol_from_advfn():
    url = "https://br.advfn.com/investimentos/futuros/dolar/hoje"
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, 'html.parser')
        price_elem = soup.find("td", string=lambda t: "Último" in t if t else False)
        if price_elem:
            return float(price_elem.find_next_sibling("td").text.strip().replace(',', '.'))
        return None
    except:
        return None

@st.cache_data(ttl=3600)
def get_selic_rate(days_ahead: int = 0) -> float:
    today = datetime.now().date()
    end_date = (today + timedelta(days=days_ahead + 60)).strftime("%d/%m/%Y")
    start_date = (today - timedelta(days=365)).strftime("%d/%m/%Y")
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.11/dados?formato=json&dataInicial={start_date}&dataFinal={end_date}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            raise ValueError("Resposta vazia BCB")
        df = pd.DataFrame(data)
        df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
        df = df.sort_values("data")
        df["valor"] = df["valor"].astype(float)
        return df.iloc[-1]["valor"] / 100.0
    except Exception as e:
        st.error(f"Erro Selic: {str(e)}. Fallback 10.5%")
        return 0.105

@st.cache_data(ttl=3600)
def parse_b3_options_pdf(file_path: str):
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
            st.text_area("Texto bruto (busque DOL/WDO)", text[:3000])
            
            # Regex para extrair linhas de DOL/WDO (ex: DOLH26C5000 BRBMEFCEBCT0 Cambial - - - - - - 0,1200 - 5,0000 - 1,0000 - 10 50 6.000,00)
            pattern = r'(DOL|WDO)\w{3}\d{2}[CP]\d{4,5}\s+BRBMEF\w+\s+Cambial\s+.*'
            lines = re.findall(pattern, text, re.MULTILINE)
            if not lines:
                st.warning("Nenhuma linha DOL/WDO encontrada. Procure 'Cambial' no texto bruto.")
                return pd.DataFrame()
            
            # Parse linhas (ajuste baseado nas imagens/table headers)
            data = []
            for line in lines:
                parts = re.split(r'\s+', line.strip())
                if len(parts) > 5:
                    symbol = parts[0]
                    tipo = 'Call' if 'C' in symbol else 'Put'
                    strike = float(symbol[-5:]) / 1000  # ex: 5000 → 5.0
                    market_price = float(parts[7].replace(',', '.').replace('-', '0')) if len(parts) > 7 else 0.0
                    data.append({'symbol': symbol, 'type': tipo, 'strike': strike, 'market_price': market_price})
            
            dol_chain = pd.DataFrame(data)
            return dol_chain
    except Exception as e:
        st.error(f"Erro PDF: {str(e)}")
        return pd.DataFrame()

# UI
st.set_page_config(page_title="DOL IV Analyzer", layout="wide")
st.title("🟢 DOL IV & Fair Price Analyzer (B3 BRL/USD Futures Options)")
st.markdown("**Public data only • Black-76 • ATM auto-select**")

st.subheader("Upload Boletim B3 (BDI_03-4.pdf) para Chain Live")
uploaded_pdf = st.file_uploader("Carregue o PDF", type="pdf")

options_chain = pd.DataFrame()
if uploaded_pdf is not None:
    with open("temp_b3.pdf", "wb") as f:
        f.write(uploaded_pdf.getbuffer())
    options_chain = parse_b3_options_pdf("temp_b3.pdf")
    if not options_chain.empty:
        st.dataframe(options_chain)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Live Data (pyield + BCB)")
    futures_df = get_dol_futures()
    
    if not futures_df.empty:
        # Colunas do log
        display_df = futures_df[["TickerSymbol", "LastRate", "Expiration"]].head(8).copy()
        display_df.columns = ["TickerSymbol", "SettlementRate", "Expiration"]
        st.dataframe(display_df, hide_index=True)
        
        active = futures_df[futures_df["Expiration"] > pd.Timestamp.now()].iloc[0]
        F = active["LastRate"]
        exp_date = active["Expiration"]
        T = (exp_date - datetime.now()).days / 365.25
        st.success(f"**Underlying F** = {F:,.4f} | T = {T*365:.1f} days")
    else:
        F = get_dol_from_advfn() or st.number_input("Manual F (BRL/USD)", value=5.70, step=0.0001)
        T = st.number_input("T (years)", value=0.25, step=0.01)
    
    r = get_selic_rate(int(T*365))
    st.info(f"**Risk-free r** (Selic) = {r*100:.3f}%")

with col2:
    st.subheader("2. ATM Option Selection")
    option_type = st.selectbox("Type", ["Call", "Put"])
    K_input = st.number_input("Strike (auto ATM if 0)", value=0.0, step=0.0001)
    market_price = st.number_input("Market Price (BRL per contract point)", value=0.0150, step=0.0001, format="%.4f")

if not options_chain.empty and 'strike' in options_chain.columns and 'market_price' in options_chain.columns:
    closest_idx = (options_chain['strike'] - F).abs().argmin()
    closest_strike = options_chain.iloc[closest_idx]['strike']
    market_price = options_chain.iloc[closest_idx]['market_price']
    st.success(f"ATM do PDF: Strike = {closest_strike:.4f}, Price = {market_price:.4f}")
    K = closest_strike
elif K_input == 0:
    K = round(F / 0.0005) * 0.0005
else:
    K = K_input

st.write(f"**ATM Strike used** = {K:,.4f} (closest to F)")

if st.button("Calculate IV & Fair Price", type="primary"):
    iv = black76_iv(market_price, F, K, T, r, option_type.lower())
    if np.isnan(iv):
        st.error("No IV solution – check inputs")
    else:
        theo = black76_call(F, K, T, r, iv) if option_type.lower() == "call" else black76_put(F, K, T, r, iv)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Implied Volatility", f"{iv*100:.2f}%")
        col_b.metric("Theoretical Fair Price", f"{theo:.4f}")
        col_c.metric("Mispricing", f"{market_price - theo:+.4f} ({(market_price - theo)/theo*100:+.2f}%)")
        
        d1 = (np.log(F/K) + (iv**2/2)*T) / (iv*np.sqrt(T))
        delta = np.exp(-r*T) * norm.cdf(d1) if option_type.lower() == "call" else -np.exp(-r*T)*norm.cdf(-d1)
        gamma = np.exp(-r*T) * norm.pdf(d1) / (F * iv * np.sqrt(T))
        vega = F * np.exp(-r*T) * norm.pdf(d1) * np.sqrt(T)
        st.markdown("**Greeks**")
        st.write(f"Δ = {delta:.4f} | Γ = {gamma:.6f} | ν (1%) = {vega/100:.4f}")

st.caption("**Data**: pyield (B3), BCB Selic, Boletim PDF / ADVFN fallback")
st.markdown("---")
st.markdown("**Next**: Refine PDF parser for 'Cambial' section, add ADVFN options scrape")
