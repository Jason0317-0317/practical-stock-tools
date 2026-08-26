import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


@st.cache_data(ttl=900, show_spinner=False)
def prices(symbols, period="5y"):
    tickers = [s.strip().upper() for s in symbols if s.strip()]
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False, threads=False)
    if raw.empty:
        return pd.DataFrame()
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    if len(tickers) == 1:
        close.columns = tickers
    return close.dropna(how="all").ffill()


def metric(label, value, delta=None):
    st.metric(label, value, delta)


def scanner():
    st.title("多股票動能掃描器")
    st.caption("參考 OpenBB 的市場篩選流程｜快速找出相對強弱與風險")
    text = st.text_input("股票代號（逗號分隔）", "2330.TW, 2454.TW, AAPL, MSFT, NVDA")
    lookback = st.selectbox("觀察期間", [21, 63, 126, 252], index=1)
    df = prices(text.split(","), "2y")
    if df.empty or len(df) <= lookback:
        st.error("目前抓不到足夠行情，請確認代號或稍後再試。")
        return
    ret = df.pct_change(lookback).iloc[-1]
    vol = df.pct_change().tail(lookback).std() * np.sqrt(252)
    score = (ret / vol.replace(0, np.nan)).sort_values(ascending=False)
    out = pd.DataFrame({"期間報酬": ret, "年化波動": vol, "風險調整動能": score}).sort_values("風險調整動能", ascending=False)
    st.dataframe(out.style.format("{:.1%}"), width="stretch")
    st.plotly_chart(px.bar(out.reset_index(), x="index", y="風險調整動能", title="風險調整動能排名"), width="stretch")


def dca():
    st.title("定期定額歷史模擬")
    st.caption("參考 backtrader 的現金流回測概念｜比較定期投入與一次投入")
    symbol = st.text_input("股票代號", "0050.TW")
    monthly = st.number_input("每月投入", 1000, 1000000, 10000, 1000)
    years = st.slider("回測年數", 1, 15, 5)
    df = prices([symbol], f"{years}y")
    if df.empty:
        st.error("查無行情資料。")
        return
    s = df.iloc[:, 0].dropna().resample("ME").last()
    shares = (monthly / s).cumsum()
    invested = pd.Series(np.arange(1, len(s)+1) * monthly, index=s.index)
    value = shares * s
    lump = invested.iloc[-1] / s.iloc[0] * s
    c1, c2, c3 = st.columns(3)
    c1.metric("累計投入", f"{invested.iloc[-1]:,.0f}")
    c2.metric("目前價值", f"{value.iloc[-1]:,.0f}", f"{value.iloc[-1]/invested.iloc[-1]-1:.1%}")
    c3.metric("一次投入價值", f"{lump.iloc[-1]:,.0f}")
    chart = pd.DataFrame({"定期定額": value, "累計投入": invested, "期初一次投入": lump})
    st.plotly_chart(px.line(chart, title="資產累積比較"), width="stretch")


def pairs():
    st.title("配對交易價差觀察")
    st.caption("參考 vectorbt 的向量化研究方法｜觀察兩檔資產的相對偏離")
    a = st.text_input("股票 A", "2330.TW")
    b = st.text_input("股票 B", "2454.TW")
    window = st.slider("Z-score 視窗", 20, 120, 60)
    df = prices([a, b], "3y").dropna()
    if df.shape[1] < 2 or len(df) < window:
        st.error("兩檔股票需要有足夠且重疊的行情。")
        return
    logp = np.log(df)
    beta = np.polyfit(logp.iloc[:, 1], logp.iloc[:, 0], 1)[0]
    spread = logp.iloc[:, 0] - beta * logp.iloc[:, 1]
    z = (spread-spread.rolling(window).mean())/spread.rolling(window).std()
    c1,c2,c3=st.columns(3)
    c1.metric("避險比率", f"{beta:.2f}")
    c2.metric("目前 Z-score", f"{z.iloc[-1]:.2f}")
    c3.metric("價格相關性", f"{df.pct_change().corr().iloc[0,1]:.2f}")
    fig=go.Figure([go.Scatter(x=z.index,y=z,name="Z-score")])
    fig.add_hline(y=2,line_dash="dash",line_color="red"); fig.add_hline(y=-2,line_dash="dash",line_color="green")
    st.plotly_chart(fig,width="stretch")


def factor():
    st.title("多因子股票評分")
    st.caption("參考 Microsoft Qlib 的特徵排名概念｜動能、低波動與趨勢綜合評分")
    text=st.text_input("股票代號（逗號分隔）","2330.TW, 2454.TW, 2308.TW, 2881.TW, 0050.TW")
    df=prices(text.split(","),"2y")
    if len(df)<260: st.error("需要至少約一年行情。"); return
    r=df.pct_change(); mom=df.iloc[-1]/df.iloc[-126]-1; vol=r.tail(126).std()*np.sqrt(252); trend=df.iloc[-1]/df.rolling(200).mean().iloc[-1]-1
    z=lambda x:(x-x.mean())/x.std(ddof=0) if x.std(ddof=0)>0 else x*0
    score=z(mom)-z(vol)+z(trend)
    out=pd.DataFrame({"半年動能":mom,"波動":vol,"長期趨勢":trend,"綜合分數":score}).sort_values("綜合分數",ascending=False)
    st.dataframe(out.style.format({"半年動能":"{:.1%}","波動":"{:.1%}","長期趨勢":"{:.1%}","綜合分數":"{:.2f}"}),width="stretch")


def montecarlo():
    st.title("蒙地卡羅價格情境")
    st.caption("參考 FinRL 的情境研究概念｜以歷史報酬模擬可能價格區間")
    symbol=st.text_input("股票代號","2330.TW"); days=st.slider("預測交易日",20,252,126); runs=st.slider("模擬次數",200,3000,1000,100)
    df=prices([symbol],"5y")
    if len(df)<100: st.error("歷史資料不足。"); return
    s=df.iloc[:,0].dropna(); lr=np.log(s/s.shift(1)).dropna(); rng=np.random.default_rng(42)
    paths=s.iloc[-1]*np.exp(np.cumsum(rng.normal(lr.mean(),lr.std(),(days,runs)),axis=0)); end=paths[-1]
    c1,c2,c3=st.columns(3); c1.metric("悲觀 10%",f"{np.quantile(end,.1):,.2f}"); c2.metric("中位數",f"{np.median(end):,.2f}"); c3.metric("樂觀 90%",f"{np.quantile(end,.9):,.2f}")
    q=pd.DataFrame({"10%":np.quantile(paths,.1,axis=1),"中位數":np.median(paths,axis=1),"90%":np.quantile(paths,.9,axis=1)})
    st.plotly_chart(px.line(q,title="模擬價格區間"),width="stretch")


def drawdown():
    st.title("回撤與復原時間分析")
    st.caption("參考 pyfolio／empyrical 的風險診斷方式")
    symbol=st.text_input("股票代號","0050.TW"); df=prices([symbol],"10y")
    if df.empty: st.error("查無行情。"); return
    s=df.iloc[:,0].dropna(); dd=s/s.cummax()-1; underwater=dd<0
    groups=(underwater!=underwater.shift()).cumsum(); spans=underwater.groupby(groups).sum(); max_days=int(spans.max()) if len(spans) else 0
    c1,c2,c3=st.columns(3); c1.metric("最大回撤",f"{dd.min():.1%}"); c2.metric("目前距高點",f"{dd.iloc[-1]:.1%}"); c3.metric("最長水下期",f"{max_days} 交易日")
    st.plotly_chart(px.area(dd,title="歷史回撤",labels={"value":"回撤"}),width="stretch")


def correlation():
    st.title("資產相關性與分散效果")
    st.caption("參考 Riskfolio-Lib 的投資組合診斷概念")
    text=st.text_input("股票代號（逗號分隔）","0050.TW, 00679B.TW, GLD, QQQ, TLT")
    df=prices(text.split(","),"3y")
    if df.shape[1]<2: st.error("至少需要兩檔有效資產。"); return
    corr=df.pct_change().corr(); st.plotly_chart(px.imshow(corr,text_auto=".2f",zmin=-1,zmax=1,color_continuous_scale="RdBu_r",title="日報酬相關性"),width="stretch")
    st.caption("相關性接近 1 代表同向程度高；分散投資通常希望資產間相關性較低。")


def valuation():
    st.title("成長股合理價試算")
    st.caption("參考 fundamental-analysis 的估值工作流｜用 EPS 與本益比做情境分析")
    eps=st.number_input("目前每股盈餘（EPS）",0.01,1000.0,10.0); growth=st.slider("預估 EPS 年成長率",-20,50,10)/100; years=st.slider("預估年數",1,10,5); pe=st.number_input("目標本益比",1.0,100.0,20.0); discount=st.slider("要求年報酬率",1,30,10)/100
    future_eps=eps*(1+growth)**years; future_price=future_eps*pe; fair=future_price/(1+discount)**years
    c1,c2,c3=st.columns(3); c1.metric("預估未來 EPS",f"{future_eps:.2f}"); c2.metric("預估未來股價",f"{future_price:.2f}"); c3.metric("折現合理價",f"{fair:.2f}")
    pes=np.arange(max(1,pe-10),pe+11); vals=future_eps*pes/(1+discount)**years; st.plotly_chart(px.line(x=pes,y=vals,labels={"x":"目標本益比","y":"折現合理價"},title="估值敏感度"),width="stretch")


def dividend():
    st.title("股息再投入分析")
    st.caption("參考 yfinance／QuantStats 的總報酬分析概念")
    symbol=st.text_input("股票代號","0056.TW")
    try:
        t=yf.Ticker(symbol); hist=t.history(period="10y",auto_adjust=False)
    except Exception: hist=pd.DataFrame()
    if hist.empty: st.error("查無行情與股息資料。"); return
    div=hist.get("Dividends",pd.Series(0,index=hist.index)); annual=div.groupby(div.index.year).sum(); last_price=float(hist["Close"].dropna().iloc[-1]); trailing=float(div.tail(252).sum()); yld=trailing/last_price if last_price else 0
    c1,c2,c3=st.columns(3); c1.metric("近一年股息",f"{trailing:.2f}"); c2.metric("估算殖利率",f"{yld:.2%}"); c3.metric("有配息年份",str(int((annual>0).sum())))
    st.plotly_chart(px.bar(x=annual.index,y=annual.values,labels={"x":"年度","y":"每股股息"},title="歷年現金股息"),width="stretch")


def rebalance():
    st.title("投資組合再平衡計算")
    st.caption("參考 bt 的權重配置流程｜算出各資產應買賣金額")
    capital=st.number_input("投資組合總市值",10000,100000000,1000000,10000)
    names=st.text_input("資產名稱（逗號分隔）","股票, 債券, 現金").split(","); current=st.text_input("目前權重 %","70, 20, 10").split(","); target=st.text_input("目標權重 %","60, 30, 10").split(",")
    try:
        c=np.array([float(x) for x in current])/100; t=np.array([float(x) for x in target])/100
        if len(names)!=len(c) or len(c)!=len(t) or not np.isclose(t.sum(),1): raise ValueError
    except ValueError: st.error("三欄數量需相同，且目標權重合計必須為 100%。"); return
    diff=(t-c)*capital; out=pd.DataFrame({"資產":[x.strip() for x in names],"目前權重":c,"目標權重":t,"建議調整金額":diff,"動作":np.where(diff>0,"買進","賣出")})
    st.dataframe(out.style.format({"目前權重":"{:.1%}","目標權重":"{:.1%}","建議調整金額":"{:,.0f}"}),width="stretch")


def journal():
    st.title("交易紀錄績效分析")
    st.caption("參考 Jesse／Freqtrade 的交易回顧流程｜可貼上每筆損益")
    raw=st.text_area("每筆交易損益（逗號分隔）","1200,-500,800,1500,-700,400,-300,2200,-600,900")
    try: pnl=pd.Series([float(x.strip()) for x in raw.split(",") if x.strip()])
    except ValueError: st.error("請輸入數字並用逗號分隔。"); return
    if pnl.empty:return
    win=pnl[pnl>0]; loss=pnl[pnl<0]; pf=win.sum()/abs(loss.sum()) if loss.sum()!=0 else math.inf; curve=pnl.cumsum(); dd=curve-curve.cummax()
    c1,c2,c3,c4=st.columns(4); c1.metric("勝率",f"{(pnl>0).mean():.1%}"); c2.metric("獲利因子",f"{pf:.2f}"); c3.metric("平均每筆",f"{pnl.mean():,.0f}"); c4.metric("最大回撤",f"{dd.min():,.0f}")
    st.plotly_chart(px.line(curve,title="累積交易損益",labels={"index":"交易序號","value":"累積損益"}),width="stretch")


def regime():
    st.title("市場多空狀態辨識")
    st.caption("參考 FinRL／Qlib 的市場狀態特徵｜趨勢與波動雙軸判讀")
    symbol=st.text_input("指數或 ETF 代號","^TWII"); df=prices([symbol],"5y")
    if len(df)<220: st.error("資料不足。"); return
    s=df.iloc[:,0]; ma=s.rolling(200).mean(); vol=s.pct_change().rolling(20).std()*np.sqrt(252); vol_mid=vol.rolling(252).median(); bull=s>ma; high=vol>vol_mid
    label=np.select([bull&~high,bull&high,~bull&~high,~bull&high],["多頭低波動","多頭高波動","空頭低波動","空頭高波動"],default="資料不足")
    st.metric("目前狀態",label[-1]); st.metric("20 日年化波動",f"{vol.iloc[-1]:.1%}")
    plot=pd.DataFrame({"價格":s,"200 日均線":ma}); st.plotly_chart(px.line(plot,title="長期趨勢"),width="stretch")


def walkforward():
    st.title("策略走勢外驗證")
    st.caption("參考 Lean／Zipline 的樣本外驗證觀念｜避免只看整段最佳參數")
    symbol=st.text_input("股票代號","2330.TW"); split=st.slider("訓練資料比例",50,85,70)/100
    df=prices([symbol],"8y")
    if len(df)<500: st.error("需要更長的歷史資料。"); return
    s=df.iloc[:,0].dropna(); cut=int(len(s)*split); candidates=[20,50,100,150,200]; rows=[]
    for w in candidates:
        pos=(s>s.rolling(w).mean()).shift(1).fillna(False); ret=s.pct_change().fillna(0); train=(1+ret.iloc[:cut]*pos.iloc[:cut]).prod()-1; test=(1+ret.iloc[cut:]*pos.iloc[cut:]).prod()-1; rows.append((w,train,test))
    out=pd.DataFrame(rows,columns=["均線天數","訓練期報酬","樣本外報酬"]); best=out.loc[out["訓練期報酬"].idxmax()]
    st.metric("訓練期最佳參數",f"{int(best['均線天數'])} 日",f"樣本外 {best['樣本外報酬']:.1%}")
    st.dataframe(out.style.format({"訓練期報酬":"{:.1%}","樣本外報酬":"{:.1%}"}),width="stretch")


def gaps():
    st.title("跳空風險與延續性分析")
    st.caption("參考 vn.py 的事件研究流程｜統計大幅跳空後的當日與後續表現")
    symbol=st.text_input("股票代號","2330.TW"); threshold=st.slider("跳空門檻（%）",1.0,10.0,3.0,0.5)/100
    try: h=yf.Ticker(symbol).history(period="10y",auto_adjust=True)
    except Exception: h=pd.DataFrame()
    if h.empty or len(h)<100: st.error("查無足夠的開高低收資料。"); return
    gap=h["Open"]/h["Close"].shift(1)-1; intraday=h["Close"]/h["Open"]-1; next5=h["Close"].shift(-5)/h["Close"]-1
    events=pd.DataFrame({"跳空":gap,"當日表現":intraday,"後續5日":next5}).loc[gap.abs()>=threshold].dropna()
    if events.empty: st.info("此門檻下沒有事件，請降低門檻。"); return
    c1,c2,c3=st.columns(3); c1.metric("事件數",str(len(events))); c2.metric("當日延續率",f"{(np.sign(events['跳空'])==np.sign(events['當日表現'])).mean():.1%}"); c3.metric("5日同向率",f"{(np.sign(events['跳空'])==np.sign(events['後續5日'])).mean():.1%}")
    st.plotly_chart(px.scatter(events,x="跳空",y="後續5日",color="當日表現",title="跳空幅度與後續 5 日報酬"),width="stretch")


APPS={"scanner":scanner,"dca":dca,"pairs":pairs,"factor":factor,"montecarlo":montecarlo,"drawdown":drawdown,"correlation":correlation,"valuation":valuation,"dividend":dividend,"rebalance":rebalance,"journal":journal,"regime":regime,"walkforward":walkforward,"gaps":gaps}


def run(name):
    st.set_page_config(page_title="實務股票工具",page_icon="📊",layout="wide")
    APPS[name]()
    st.warning("僅供研究與教育用途，不構成投資建議。行情可能延遲；實際交易需另計稅費、滑價與流動性。")
