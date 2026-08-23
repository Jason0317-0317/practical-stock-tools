import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="股票風險比較", page_icon="⚖️", layout="wide")
st.title("股票報酬與風險比較")
st.caption("參考 Goldman Sachs gs-quant 的風險分析流程｜最多比較 8 檔")

symbols = st.text_input("股票代號（逗號分隔）", "AAPL, MSFT, 2330.TW").upper()
years = st.selectbox("期間", [1, 3, 5, 10], index=2)
tickers = list(dict.fromkeys(x.strip() for x in symbols.replace("，", ",").split(",") if x.strip()))[:8]

rows, curves = [], {}
for ticker in tickers:
    try:
        data = yf.download(ticker, period=f"{years}y", auto_adjust=True, progress=False, timeout=12)
        close = data["Close"].squeeze().dropna()
    except Exception:
        close = pd.Series(dtype=float)
    if len(close) < 30:
        continue
    ret = close.pct_change().dropna()
    dd = close / close.cummax() - 1
    rows.append({"股票":ticker,"年化報酬":ret.mean()*252,"年化波動":ret.std()*252**0.5,"最大回撤":dd.min(),"下跌日比例":(ret<0).mean()})
    curves[ticker] = close / close.iloc[0] * 100

if not rows:
    st.warning("查無足夠資料，請確認股票代號。")
    st.stop()
table = pd.DataFrame(rows).set_index("股票")
st.dataframe(table.style.format("{:.1%}"), width="stretch")
scatter = px.scatter(pd.DataFrame(rows), x="年化波動", y="年化報酬", text="股票", size_max=24, title="風險－報酬分布")
scatter.update_traces(textposition="top center")
st.plotly_chart(scatter, width="stretch")
curve = pd.concat(curves, axis=1).dropna(how="all")
st.line_chart(curve, height=400)
st.info("比較結果未調整匯率、股息稅與交易成本；不同市場及幣別不可直接視為同一投資條件。")
