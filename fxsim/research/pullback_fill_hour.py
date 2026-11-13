"""E048: 押し目(S048)の約定を、日足の区切り(ロールオーバー)から k 時間遅らせたときの 1 取引の損益。

    python -m fxsim.research.pullback_fill_hour

信号は日足(相場の DB)、約定は 1 時間足の始値(mid)。費用はその時刻の記録のスプレッド(入りと出の平均)+ 往復 0.2 pip。
1 時間足のある 2010〜2026 年だけ。
"""
# ruff: noqa: E402, E501, E701, E702, E731, I001, B023
import warnings

import numpy as np

warnings.simplefilter("ignore")
from fxsim.storage import prices
from fxsim.research.intraday_edge import SOURCE
from fxsim.strategies.indicators import rsi, sma
PAIRS=["USDJPY","EURUSD","GBPUSD","AUDUSD","NZDUSD","USDCHF","USDCAD","EURJPY","GBPJPY","AUDJPY"]
con=prices.open_market_db()
KS=[0,1,2,3,5,8,10,12,15,18]
SPL=np.datetime64("2018-11-01").astype("datetime64[s]").astype(np.int64)
res={k:[] for k in KS}   # (gross pips, cost pips, exit time, pair)
for p in PAIRS:
    d=prices.read_rates(con,SOURCE,p,1440); h=prices.read_rates(con,SOURCE,p,60)
    pip=0.01 if p.endswith("JPY") else 0.0001; point=pip/10
    c=d["close"]; t=d["time"]; ma=sma(c,200); r=rsi(c,2)
    hmid=h["open"]+h["spread"]*point/2; ht=h["time"]; hs=h["spread"]*point/pip
    def px(tt):
        j=np.searchsorted(ht,tt)
        return (None,None) if j>=len(ht) or ht[j]-tt>4*3600 else (hmid[j],hs[j])
    cur=0; age=0; ent=None
    for i in range(200,len(c)-1):
        if t[i]<ht[0]: continue
        ex=False
        if cur!=0:
            age+=1
            if (r[i]>50 if cur>0 else r[i]<50) or age>=10: ex=True
        if ex:
            for k in KS:
                a,sa=px(t[i+1]+k*3600)
                if ent[k][0] is not None and a is not None:
                    res[k].append((cur*(a-ent[k][0])/pip, (sa+ent[k][1])/2+0.2, t[i+1], p))
            cur=0
        if cur==0 and np.isfinite(ma[i]):
            e=1 if (c[i]>ma[i] and r[i]<10) else -1 if (c[i]<ma[i] and r[i]>90) else 0
            if e: cur=e; age=0; ent={k:px(t[i+1]+k*3600) for k in KS}
print("約定をロールオーバー(日足の区切り)から k 時間遅らせたときの、押し目(RSI2)の 1 取引の損益(pip、10 ペア、2010〜2026)")
print("| k 時間後 | 取引 | 素の損益 | 費用(その時刻のスプレッド + 0.2) | 差し引き | t(月) | 前半 2010〜18 | 後半 2018〜26 |")
print("|---:|---:|---:|---:|---:|---:|---:|---:|")
for k in KS:
    a=np.array([(g,cst,tt) for g,cst,tt,_ in res[k]]); net=a[:,0]-a[:,1]; mon=(a[:,2]//(86400*30.44)).astype(int)
    ms=np.bincount(mon-mon.min(),weights=net); ms=ms[np.bincount(mon-mon.min())>0]
    print(f"| {k} | {len(a)} | {a[:,0].mean():+.2f} | {a[:,1].mean():.2f} | {net.mean():+.2f} | {ms.mean()/(ms.std(ddof=1)/np.sqrt(len(ms))):+.2f} | {net[a[:,2]<SPL].mean():+.2f} | {net[a[:,2]>=SPL].mean():+.2f} |")
