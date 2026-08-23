import math
import streamlit as st

st.set_page_config(page_title="部位與停損規劃", page_icon="🛡️", layout="wide")
st.title("部位大小與停損規劃")
st.caption("參考 StockSharp 的交易風控概念｜先決定可承受損失，再決定張數")

with st.form("plan"):
    a, b, c = st.columns(3)
    capital = a.number_input("可運用資金", 1_000.0, 1_000_000_000.0, 1_000_000.0, 10_000.0)
    risk_pct = b.number_input("單筆風險（%）", 0.1, 10.0, 1.0, 0.1) / 100
    fee_pct = c.number_input("來回成本（%）", 0.0, 5.0, 0.585, 0.05) / 100
    d, e, f = st.columns(3)
    entry = d.number_input("預計進場價", 0.01, 1_000_000.0, 100.0)
    stop = e.number_input("停損價", 0.01, 1_000_000.0, 95.0)
    target = f.number_input("目標價", 0.01, 1_000_000.0, 115.0)
    market = st.selectbox("交易單位", ["台股整張（1000 股）", "台股零股／美股（1 股）"])
    st.form_submit_button("計算交易計畫", width="stretch")

unit = 1000 if market.startswith("台股整張") else 1
if stop >= entry:
    st.error("多單停損價必須低於進場價。")
    st.stop()
risk_budget = capital * risk_pct
risk_per_share = entry - stop + entry * fee_pct
shares = math.floor(risk_budget / risk_per_share / unit) * unit
position = shares * entry
loss = shares * risk_per_share
profit = shares * max(target - entry - entry * fee_pct, 0)
rr = profit / loss if loss else 0

cols = st.columns(5)
cols[0].metric("建議股數", f"{shares:,}")
cols[1].metric("投入金額", f"{position:,.0f}")
cols[2].metric("預估最大損失", f"{loss:,.0f}")
cols[3].metric("預估目標獲利", f"{profit:,.0f}")
cols[4].metric("風險報酬比", f"1 : {rr:.2f}")
if shares == 0:
    st.warning("依目前風險限制無法建立一個完整交易單位；請改用零股或縮小停損距離。")
elif position > capital:
    st.warning("風險允許的部位超過可用資金，實際股數還需受資金上限限制。")
st.info("成本為概算，實際手續費、交易稅、滑價及最低費用請依券商與市場調整。")
