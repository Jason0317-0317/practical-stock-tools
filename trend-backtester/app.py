import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="趨勢策略回測", page_icon="📈", layout="wide")
st.title("趨勢策略回測")
st.caption("參考 QuantConnect Lean／Zipline 的策略研究流程｜日線資料｜非投資建議")

with st.form("inputs"):
    a, b, c, d = st.columns(4)
    ticker = a.text_input("股票代號", "2330.TW").strip().upper()
    fast = b.number_input("短均線", 2, 100, 20)
    slow = c.number_input("長均線", 5, 300, 60)
    cost = d.number_input("單邊成本（%）", 0.0, 2.0, 0.15, 0.05) / 100
    run = st.form_submit_button("執行回測", width="stretch")

if run or "ready" not in st.session_state:
    st.session_state.ready = True
    if fast >= slow:
        st.error("短均線必須小於長均線。")
        st.stop()
    try:
        data = yf.download(ticker, period="10y", auto_adjust=True, progress=False, timeout=15)
        close = data["Close"].squeeze().dropna()
    except Exception:
        close = pd.Series(dtype=float)
    if len(close) <= slow:
        st.warning("資料不足，請確認股票代號或縮短均線期間。")
        st.stop()

    frame = pd.DataFrame({"收盤": close})
    frame["短均"] = close.rolling(fast).mean()
    frame["長均"] = close.rolling(slow).mean()
    frame["持有"] = (frame["短均"] > frame["長均"]).astype(float).shift(1).fillna(0)
    frame["轉換"] = frame["持有"].diff().abs().fillna(0)
    frame["策略報酬"] = frame["持有"] * close.pct_change().fillna(0) - frame["轉換"] * cost
    frame["買進持有"] = close.pct_change().fillna(0)
    frame["策略淨值"] = (1 + frame["策略報酬"]).cumprod()
    frame["持有淨值"] = (1 + frame["買進持有"]).cumprod()

    years = max((frame.index[-1] - frame.index[0]).days / 365.25, 0.01)
    total = frame["策略淨值"].iloc[-1] - 1
    cagr = frame["策略淨值"].iloc[-1] ** (1 / years) - 1
    drawdown = frame["策略淨值"] / frame["策略淨值"].cummax() - 1
    trades = int(frame["轉換"].sum() / 2)
    cols = st.columns(4)
    cols[0].metric("策略總報酬", f"{total:+.1%}")
    cols[1].metric("年化報酬", f"{cagr:+.1%}")
    cols[2].metric("最大回撤", f"{drawdown.min():.1%}")
    cols[3].metric("完整交易次數", f"{trades}")
    fig = go.Figure()
    fig.add_scatter(x=frame.index, y=frame["策略淨值"], name="趨勢策略")
    fig.add_scatter(x=frame.index, y=frame["持有淨值"], name="買進持有")
    fig.update_layout(height=440, hovermode="x unified", yaxis_title="累積淨值")
    st.plotly_chart(fig, width="stretch")
    st.info("回測不包含滑價、稅務、融資成本與流動性限制；歷史績效不代表未來結果。")
