import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="技術訊號檢查",page_icon="🔎",layout="wide")
st.title("技術訊號與簡易回測")
st.caption("參考 PyBroker｜RSI + 長期均線的規則策略｜不連接券商、不自動下單")

a,b,c=st.columns(3)
ticker=a.text_input("股票代號","AAPL").strip().upper()
rsi_buy=b.slider("RSI 進場上限",10,50,35)
rsi_exit=c.slider("RSI 出場下限",50,90,65)
data=yf.download(ticker,period="10y",auto_adjust=True,progress=False,timeout=15)
close=data["Close"].squeeze().dropna() if not data.empty else pd.Series(dtype=float)
if len(close)<220:
    st.warning("資料不足，請確認股票代號。")
    st.stop()
delta=close.diff()
gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean()
loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
rsi=100-100/(1+gain/loss.replace(0,pd.NA))
ma=close.rolling(200).mean()
entry=(rsi<rsi_buy)&(close>ma)
exit_=(rsi>rsi_exit)|(close<ma)
position=pd.Series(0.0,index=close.index)
holding=False
for i in range(len(close)):
    if not holding and bool(entry.iloc[i]): holding=True
    elif holding and bool(exit_.iloc[i]): holding=False
    position.iloc[i]=float(holding)
strategy=position.shift(1).fillna(0)*close.pct_change().fillna(0)
equity=(1+strategy).cumprod()
dd=equity/equity.cummax()-1
cols=st.columns(4)
cols[0].metric("目前訊號","持有" if position.iloc[-1] else "觀望")
cols[1].metric("目前 RSI",f"{rsi.iloc[-1]:.1f}")
cols[2].metric("策略總報酬",f"{equity.iloc[-1]-1:+.1%}")
cols[3].metric("最大回撤",f"{dd.min():.1%}")
fig=go.Figure()
fig.add_scatter(x=close.index,y=close,name="收盤")
fig.add_scatter(x=ma.index,y=ma,name="200 日均線")
fig.add_scatter(x=close.index[entry],y=close[entry],name="進場條件",mode="markers",marker_symbol="triangle-up")
fig.update_layout(height=450,hovermode="x unified")
st.plotly_chart(fig,width="stretch")
st.info("訊號僅使用收盤後資料；實盤還需考慮隔日成交、滑價、成本及停損。")
