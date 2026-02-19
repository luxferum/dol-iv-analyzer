import streamlit as st
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import pandas as pd
import requests
from datetime import datetime, timedelta
import pyield as py  # For B3 futures data
from bs4 import BeautifulSoup  # For ADVFN scrape
import pdfplumber  # For PDF parsing

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
@st.cache_data(ttl=300)  # 5 min cache
def get_dol_futures():
    """Fetch latest DOL futures data via pyield (public B3 settlements)."""
    today_str = datetime.now().strftime("%d-%m-%Y")
    
    try:
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
        
        # Encontra colunas dinamicamente
        ticker_col = next((c for c in df.columns if "ticker" in c.lower() or "symbol" in c.lower()), None)
        settle_col = next((c for c in df.columns if "settle" in c.lower() or "rate" in c.lower() or "price" in c.lower()), None)
        exp_col = next((c for c in df.columns if "expir" in c.lower() or "venc" in c.lower() or "maturity" in c.lower()), None)
        
        if not all([ticker_col, settle_col, exp_col]):
            raise KeyError("Colunas essenciais não encontradas: ticker/symbol, settlement/rate/price, expiration/venc/maturity")
        
        # Renomeia para padronizar
        df = df.rename(columns={ticker_col: "TickerSymbol", settle_col: "SettlementRate", exp_col: "ExpirationDate"})
        
        # Converte expiração
        df["Expiration"] = pd.to_datetime(df["ExpirationDate"])
        
        df = df.sort_values("Expiration")
        active_df = df[df["Expiration"] > pd.Timestamp.now()]
        
        if active_df.empty:
            raise ValueError("No active futures found")
        
        return active_df
    
    except Exception as e:
        st.error(f"Erro pyield: {str(e)}. Usando inputs manuais.")
        return pd.DataFrame()  # fallback para manual

def get_dol_from_advfn():
    """Fallback scrape ADVFN for current DOL front-month price."""
    url = "https://br.advfn.com/investimentos/futuros/dolar/hoje"
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Selector baseado em inspeção: ajuste se ADVFN mudar layout
        # Tipicamente, preço "Último" em table class="quotes"
        price_elem = soup.find("td", string=lambda t: "Último" in t if t else False)
        if price_elem:
            F_str = price_elem.find_next_sibling("td").text.strip().replace(',', '.')
            return float(F_str)
        return None
    except Exception as e:
        st.error(f"Erro ADVFN scrape: {str(e)}")
        return None

@st.cache_data(ttl=3600)  # 1 hora de cache, Selic muda pouco
def get_selic_rate(days_ahead: int = 0) -> float:
    """BCB Selic anualizada (série 11). Retorna em decimal (ex: 0.1175 para 11.75%)."""
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

@st.cache_data(ttl=3600)
def parse_b3_options_pdf(file_path: str):
    """Parser para BDI_03-4.pdf → extrai chain DOL/WDO."""
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ""
            tables = []
            for page in pdf.pages:
                text += page.extract_text() + "\n"
                page_tables = page.extract_tables()  # auto-detect tables
                tables.extend(page_tables or [])
            
            # Debug: mostre texto para achar "Cambial" ou "DOL"
            st.text_area("Texto bruto (busque DOL/WDO)", text[:3000])
            
            # Se tables detectadas, converta para DF
            if tables:
                chain = pd.concat([pd.DataFrame(t) for t in tables if t], ignore_index=True)
                
                # Limpeza manual: assume primeira linha é header
                chain.columns = chain.iloc[0].str.strip().str.lower()  # Normalize headers
                chain = chain[1:].reset_index(drop=True)
                
                # Filtre linhas com DOL/WDO (case-insensitive)
                dol_chain = chain[chain.apply(lambda row: row.astype(str).str.contains('DOL|WDO|Dólar|Dolar', case=False, na=False).any(), axis=1)]
                
                if dol_chain.empty:
                    st.warning("Nenhuma linha com DOL/WDO encontrada. Verifique seção 'Cambial' no PDF.")
                else:
                    # Parse adicional (exemplo: strike, type, market_price)
                    # Ajuste colunas reais baseadas no BDI_03-4 (ex: 'instrumento financeiro', 'preço de fechamento')
                    if 'instrumento financeiro' in dol_chain.columns:
                        dol_chain['strike'] = dol_chain['instrumento financeiro'].str.extract(r'(\d{4,5})', expand=False).astype(float) / 1000  # ex: 5000 → 5.000
                        dol_chain['type'] = np.where(dol_chain['instrumento financeiro'].str.contains('C', case=False), 'Call', 'Put')
                        # Expiração: parse mês/ano (ex: H26 → março/26)
                        month_map = {'F':1, 'G':2, 'H':3, 'J':4, 'K':5, 'M':6, 'N':7, 'Q':8, 'U':9, 'V':10, 'X':11, 'Z':12}
                        dol_chain['month_code'] = dol_chain['instrumento financeiro'].str[3]
                        dol_chain['month'] = dol_chain['month_code'].map(month_map)
                        dol_chain['year'] = 2000 + dol_chain['instrumento financeiro'].str[4:6].astype(int)
                        dol_chain['expiration'] = pd.to_datetime(dol_chain[['year', 'month']].assign(day=1))  # Primeiro dia do mês de expiração
                    if 'preço de fechamento' in dol_chain.columns:
                        dol_chain['market_price'] = dol_chain['preço de fechamento'].str.replace(',', '.').replace('-', np.nan).astype(float)
                    
                    return dol_chain
            
            st.warning("Nenhuma tabela detectada no PDF. Tente outro boletim ou ajuste parser.")
            return pd.DataFrame()
    
    except Exception as e:
        st.error(f"Erro parsing PDF: {str(e)}")
        return pd.DataFrame()

# ====================== STREAMLIT UI ======================
st.set_page_config(page_title="DOL IV Analyzer", layout="wide")
st.title("🟢 DOL IV & Fair Price Analyzer (B3 BRL/USD Futures Options)")
st.markdown("**Public data only • Black-76 • ATM auto-select**")

# Upload PDF antes de definir colunas e F (para potencial uso em ATM)
st.subheader("Upload Boletim B3 (BDI_03-4.pdf) para Chain Live")
uploaded_pdf = st.file_uploader("Carregue o PDF", type="pdf")

options_chain = pd.DataFrame()  # Default vazio
if uploaded_pdf is not None:
    with open("temp_b3.pdf", "wb") as f:
        f.write(uploaded_pdf.getbuffer())
    options_chain = parse_b3_options_pdf("temp_b3.pdf")
    
    if not options_chain.empty:
        # Mostre colunas chave (ajuste se parser melhorar)
        display_cols = [col for col in ['instrumento financeiro', 'type', 'strike', 'expiration', 'market_price'] if col in options_chain.columns]
        st.dataframe(options_chain[display_cols])

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Live Data (pyield + BCB)")
    futures_df = get_dol_futures()
    
    if not futures_df.empty:
        st.dataframe(futures_df[["TickerSymbol", "SettlementRate", "Expiration"]].head(8), hide_index=True)
        
        # Nearest active future (default front-month)
        active = futures_df[futures_df["Expiration"] > datetime.now()].iloc[0]
        F = active["SettlementRate"]
        exp_date = active["Expiration"]
        T = (exp_date - datetime.now()).days / 365.25  # Actual/365.25
        st.success(f"**Underlying F** = {F:,.4f} | T = {T*365:.1f} days")
    else:
        # Fallback ADVFN se pyield falhar
        F = get_dol_from_advfn() or st.number_input("Manual F (BRL/USD)", value=5.70, step=0.0001)
        T = st.number_input("T (years)", value=0.25, step=0.01)
    
    r = get_selic_rate(int(T*365)) 
    st.info(f"**Risk-free r** (Selic) = {r*100:.3f}%")

with col2:
    st.subheader("2. ATM Option Selection")
    option_type = st.selectbox("Type", ["Call", "Put"])
    K_input = st.number_input("Strike (auto ATM if 0)", value=0.0, step=0.0001)
    
    market_price = st.number_input("Market Price (BRL per contract point)", value=0.0150, step=0.0001, format="%.4f")

# Auto ATM logic (use chain se disponível)
if not options_chain.empty and 'strike' in options_chain.columns and 'market_price' in options_chain.columns and 'expiration' in options_chain.columns:
    closest_idx = (options_chain['strike'] - F).abs().argmin()
    closest_strike = options_chain.iloc[closest_idx]['strike']
    market_price = options_chain.iloc[closest_idx]['market_price']
    exp_date = options_chain.iloc[closest_idx]['expiration']
    T = (exp_date - datetime.now()).days / 365.25
    st.success(f"ATM detectado do PDF: Strike = {closest_strike:.4f}, Market Price = {market_price:.4f}, T = {T:.4f} anos")
    K = closest_strike
elif K_input == 0:
    K = round(F / 0.0005) * 0.0005  # B3 tick 0.0005 for most series
else:
    K = K_input

st.write(f"**ATM Strike used** = {K:,.4f} (closest to F)")

# ====================== CALCULATION ======================
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
        
        # Greeks (bonus – as per your quant spec)
        d1 = (np.log(F/K) + (iv**2/2)*T) / (iv*np.sqrt(T))
        delta = np.exp(-r*T) * norm.cdf(d1) if option_type.lower()=="call" else -np.exp(-r*T)*norm.cdf(-d1)
        gamma = np.exp(-r*T) * norm.pdf(d1) / (F * iv * np.sqrt(T))
        vega = F * np.exp(-r*T) * norm.pdf(d1) * np.sqrt(T)  # per 1% vol
        st.markdown("**Greeks**")
        st.write(f"Δ = {delta:.4f} | Γ = {gamma:.6f} | ν (1%) = {vega/100:.4f}")

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
    # Bracket [0.001, 5.0] covers 0.1 % → 500 % vol
    try:
        return brentq(objective, 1e-4, 5.0, xtol=tol)
    except:
        return np.nan  # No solution (deep ITM/OTM or bad price)

# ====================== DATA FETCH (Public/Free) ======================
@st.cache_data(ttl=300)  # 5 min cache
def get_dol_futures():
    """Fetch latest DOL futures data via pyield (public B3 settlements)."""
    today_str = datetime.now().strftime("%d-%m-%Y")
    
    try:
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

def get_dol_from_advfn():
    """Fallback scrape ADVFN for current DOL front-month price."""
    url = "https://br.advfn.com/investimentos/futuros/dolar/hoje"
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Selector baseado em inspeção: ajuste se ADVFN mudar layout
        # Tipicamente, preço "Último" em table class="quotes"
        price_elem = soup.find("td", string=lambda t: "Último" in t if t else False)
        if price_elem:
            F_str = price_elem.find_next_sibling("td").text.strip().replace(',', '.')
            return float(F_str)
        return None
    except Exception as e:
        st.error(f"Erro ADVFN scrape: {str(e)}")
        return None

def get_dol_options_from_advfn():
    url = "https://br.advfn.com/investimentos/opcoes/dolar"
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # A tabela principal geralmente tem class="opcoes" ou id específico
        table = soup.find("table", {"class": "opcoes"})  # ajuste após inspecionar
        if not table:
            return pd.DataFrame()
        
        rows = table.find_all("tr")[1:]  # pula header
        data = []
        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 5:
                strike = float(cols[0].text.strip().replace(',', '.'))
                tipo = cols[1].text.strip()  # Call/Put
                venc = cols[2].text.strip()
                preco = float(cols[3].text.strip().replace(',', '.'))
                data.append({"strike": strike, "type": tipo, "vencimento": venc, "market_price": preco})
        
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Erro scrape ADVFN opções: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)  # 1 hora de cache, Selic muda pouco
def get_selic_rate(days_ahead: int = 0) -> float:
    """BCB Selic anualizada (série 11). Retorna em decimal (ex: 0.1175 para 11.75%)."""
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

@st.cache_data(ttl=3600)
def parse_b3_options_pdf(file_path: str):
    """Parser para BDI_03-4.pdf → extrai chain DOL/WDO."""
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ""
            tables = []
            for page in pdf.pages:
                text += page.extract_text() + "\n"
                page_tables = page.extract_tables()  # auto-detect tables
                tables.extend(page_tables or [])
            
            # Debug: mostre texto para achar "Cambial" ou "DOL"
            st.text_area("Texto bruto (busque DOL/WDO)", text[:3000])
            
            # Se tables detectadas, converta para DF
            if tables:
                chain = pd.concat([pd.DataFrame(t) for t in tables if t], ignore_index=True)
                
                # Limpeza manual: assume primeira linha é header
                chain.columns = chain.iloc[0].str.strip().str.lower()  # Normalize headers
                chain = chain[1:].reset_index(drop=True)
                
                # Filtre linhas com DOL/WDO (case-insensitive)
                dol_chain = chain[chain.apply(lambda row: row.astype(str).str.contains('DOL|WDO|Dólar|Dolar', case=False, na=False).any(), axis=1)]
                
                if dol_chain.empty:
                    st.warning("Nenhuma linha com DOL/WDO encontrada. Verifique seção 'Cambial' no PDF.")
                else:
                    # Parse adicional (exemplo: strike, type, market_price)
                    # Ajuste colunas reais baseadas no BDI_03-4 (ex: 'instrumento financeiro', 'preço de fechamento')
                    if 'instrumento financeiro' in dol_chain.columns:
                        dol_chain['strike'] = dol_chain['instrumento financeiro'].str.extract(r'(\d{4,5})', expand=False).astype(float) / 1000  # ex: 5000 → 5.000
                        dol_chain['type'] = np.where(dol_chain['instrumento financeiro'].str.contains('C', case=False), 'Call', 'Put')
                        # Expiração: parse mês/ano (ex: H26 → março/26)
                        month_map = {'F':1, 'G':2, 'H':3, 'J':4, 'K':5, 'M':6, 'N':7, 'Q':8, 'U':9, 'V':10, 'X':11, 'Z':12}
                        dol_chain['month_code'] = dol_chain['instrumento financeiro'].str[3]
                        dol_chain['month'] = dol_chain['month_code'].map(month_map)
                        dol_chain['year'] = 2000 + dol_chain['instrumento financeiro'].str[4:6].astype(int)
                        dol_chain['expiration'] = pd.to_datetime(dol_chain[['year', 'month']].assign(day=1))  # Primeiro dia do mês de expiração
                    if 'preço de fechamento' in dol_chain.columns:
                        dol_chain['market_price'] = dol_chain['preço de fechamento'].str.replace(',', '.').replace('-', np.nan).astype(float)
                    
                    return dol_chain
            
            st.warning("Nenhuma tabela detectada no PDF. Tente outro boletim ou ajuste parser.")
            return pd.DataFrame()
    
    except Exception as e:
        st.error(f"Erro parsing PDF: {str(e)}")
        return pd.DataFrame()

# ====================== STREAMLIT UI ======================
st.set_page_config(page_title="DOL IV Analyzer", layout="wide")
st.title("🟢 DOL IV & Fair Price Analyzer (B3 BRL/USD Futures Options)")
st.markdown("**Public data only • Black-76 • ATM auto-select**")

# Upload PDF antes de definir colunas e F (para potencial uso em ATM)
st.subheader("Upload Boletim B3 (BDI_03-4.pdf) para Chain Live")
uploaded_pdf = st.file_uploader("Carregue o PDF", type="pdf")

options_chain = pd.DataFrame()  # Default vazio
if uploaded_pdf is not None:
    with open("temp_b3.pdf", "wb") as f:
        f.write(uploaded_pdf.getbuffer())
    options_chain = parse_b3_options_pdf("temp_b3.pdf")
    
    if not options_chain.empty:
        # Mostre colunas chave (ajuste se parser melhorar)
        display_cols = [col for col in ['instrumento financeiro', 'type', 'strike', 'expiration', 'market_price'] if col in options_chain.columns]
        st.dataframe(options_chain[display_cols])

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Live Data (pyield + BCB)")
    futures_df = get_dol_futures()
    
    if not futures_df.empty:
        st.dataframe(futures_df[["TickerSymbol", "SettlementRate", "Expiration"]].head(8), hide_index=True)
        
        # Nearest active future (default front-month)
        active = futures_df[futures_df["Expiration"] > datetime.now()].iloc[0]
        F = active["SettlementRate"]
        exp_date = active["Expiration"]
        T = (exp_date - datetime.now()).days / 365.25  # Actual/365.25
        st.success(f"**Underlying F** = {F:,.4f} | T = {T*365:.1f} days")
    else:
        # Fallback ADVFN se pyield falhar
        F = get_dol_from_advfn() or st.number_input("Manual F (BRL/USD)", value=5.70, step=0.0001)
        T = st.number_input("T (years)", value=0.25, step=0.01)
    
    r = get_selic_rate(int(T*365)) 
    st.info(f"**Risk-free r** (Selic) = {r*100:.3f}%")

with col2:
    st.subheader("2. ATM Option Selection")
    option_type = st.selectbox("Type", ["Call", "Put"])
    K_input = st.number_input("Strike (auto ATM if 0)", value=0.0, step=0.0001)
    
    market_price = st.number_input("Market Price (BRL per contract point)", value=0.0150, step=0.0001, format="%.4f")

# Auto ATM logic (use chain se disponível)
if not options_chain.empty and 'strike' in options_chain.columns and 'market_price' in options_chain.columns and 'expiration' in options_chain.columns:
    closest_idx = (options_chain['strike'] - F).abs().argmin()
    closest_strike = options_chain.iloc[closest_idx]['strike']
    market_price = options_chain.iloc[closest_idx]['market_price']
    exp_date = options_chain.iloc[closest_idx]['expiration']
    T = (exp_date - datetime.now()).days / 365.25
    st.success(f"ATM detectado do PDF: Strike = {closest_strike:.4f}, Market Price = {market_price:.4f}, T = {T:.4f} anos")
    K = closest_strike
elif K_input == 0:
    K = round(F / 0.0005) * 0.0005  # B3 tick 0.0005 for most series
else:
    K = K_input

st.write(f"**ATM Strike used** = {K:,.4f} (closest to F)")

# ====================== CALCULATION ======================
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
        
        # Greeks (bonus – as per your quant spec)
        d1 = (np.log(F/K) + (iv**2/2)*T) / (iv*np.sqrt(T))
        delta = np.exp(-r*T) * norm.cdf(d1) if option_type.lower()=="call" else -np.exp(-r*T)*norm.cdf(-d1)
        gamma = np.exp(-r*T) * norm.pdf(d1) / (F * iv * np.sqrt(T))
        vega = F * np.exp(-r*T) * norm.pdf(d1) * np.sqrt(T)  # per 1% vol
        st.markdown("**Greeks**")
        st.write(f"Δ = {delta:.4f} | Γ = {gamma:.6f} | ν (1%) = {vega/100:.4f}")

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


