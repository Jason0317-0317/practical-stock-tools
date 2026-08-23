import math
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

st.set_page_config(page_title="選擇權定價與損益", page_icon="🧮", layout="wide")
st.title("選擇權定價與到期損益")
st.caption("參考 Financial-Models-Numerical-Methods｜Black–Scholes 歐式選擇權模型")

a,b,c = st.columns(3)
spot=a.number_input("標的現價",0.01,1_000_000.0,100.0)
strike=b.number_input("履約價",0.01,1_000_000.0,105.0)
days=c.number_input("到期天數",1,3650,30)
d,e,f=st.columns(3)
vol=d.number_input("年化波動率（%）",0.1,500.0,25.0)/100
rate=e.number_input("無風險利率（%）",-10.0,30.0,4.0)/100
premium=f.number_input("實際權利金（每股）",0.0,1_000_000.0,2.0)
kind=st.radio("類型",["買進 Call","買進 Put"],horizontal=True)

t=days/365
d1=(math.log(spot/strike)+(rate+vol**2/2)*t)/(vol*math.sqrt(t))
d2=d1-vol*math.sqrt(t)
call=spot*norm.cdf(d1)-strike*math.exp(-rate*t)*norm.cdf(d2)
put=strike*math.exp(-rate*t)*norm.cdf(-d2)-spot*norm.cdf(-d1)
model=call if kind.endswith("Call") else put
delta=norm.cdf(d1) if kind.endswith("Call") else norm.cdf(d1)-1
gamma=norm.pdf(d1)/(spot*vol*math.sqrt(t))
cols=st.columns(4)
cols[0].metric("模型理論價",f"{model:.3f}")
cols[1].metric("市場價差",f"{premium-model:+.3f}")
cols[2].metric("Delta",f"{delta:.3f}")
cols[3].metric("Gamma",f"{gamma:.4f}")
prices=np.linspace(max(spot*.4,.01),spot*1.6,240)
payoff=np.maximum(prices-strike,0)-premium if kind.endswith("Call") else np.maximum(strike-prices,0)-premium
fig=go.Figure(go.Scatter(x=prices,y=payoff,name="到期每股損益"))
fig.add_hline(y=0,line_dash="dash")
fig.update_layout(height=420,xaxis_title="到期標的價格",yaxis_title="每股損益")
st.plotly_chart(fig,width="stretch")
st.info("模型假設固定波動率、連續交易及無摩擦市場，不適合直接視為報價或買賣建議。")
