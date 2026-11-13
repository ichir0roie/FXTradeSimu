"""E048: テクニカルの手法の「今の実装」と「理論どおりの使い方」を、主要 10 ペアの日足の実データで並べる。

    python -m fxsim.research.ta_theory

単位・費用・期間の分け方は ta_scan と同じ(値動き − 費用、10 ペアの合計、建玉比の年率 %。A = 1996〜2013、B = 2014〜2026)。
RSI(売られすぎから戻るのを待つ・レンジのときだけ・トレンドの中の押し・ダイバージェンス)、ボリンジャー(横ばいのときの逆張り・
バンドウォーク)、一目(三役好転)、MACD(ゼロ線・トレンドの向き)、グランビルの法則、チャネル抜け(タートルの原典の逆指値)。
"""
# ruff: noqa: E501, E701, E702, E741, F403, F405, N806
import warnings

import numpy as np

from fxsim.research.ta_scan import *
from fxsim.strategies.indicators import ichimoku

warnings.simplefilter("ignore")
d=Data(); c,h,lo,o=d.c,d.h,d.l,d.o
false=np.zeros(c.shape,bool)
def stats(net):
    live=d.t>=START; res=[]
    for m in (live, live&(d.t<SPLIT), d.t>=SPLIT):
        tot=net[m].sum(1); mon=d.month[m]; ms=np.bincount(mon-mon.min(),weights=tot)[np.unique(mon-mon.min())]
        res.append((tot.sum()/(m.sum()/260)*100, ms.mean()/(ms.std(ddof=1)/np.sqrt(len(ms)))))
    return res, int((net[live].sum(0)>0).sum())
def line(name,pos=None,net=None,tr=None):
    if net is None:
        pos=np.nan_to_num(pos); pos[~np.isfinite(c)]=0
        r=np.nan_to_num(d.ret); net=np.zeros(pos.shape); net[1:]=pos[:-1]*r[1:]
        prev=np.vstack([np.zeros((1,10)),pos[:-1]]); net-=np.abs(pos-prev)*COST; tr=np.abs(pos-prev).sum()/2/32
    (a,b,cc),pp=stats(net)
    print(f"{name:52s} all {a[0]:+6.2f} t {a[1]:+5.2f} | A {b[0]:+6.2f} t {b[1]:+5.2f} | B {cc[0]:+6.2f} t {cc[1]:+5.2f} | tr/y {tr:4.0f} pairs {pp}")
a20=d.col(atr,h,lo,c,n=20); ax,pdi,mdi=d.col(adx,h,lo,c,n=14)
sma200=d.col(sma,c,n=200); tr200=np.sign(c-sma200)
def crossup(x,lv): return (shift(x)<=lv)&(x>lv)
def crossdn(x,lv): return (shift(x)>=lv)&(x<lv)
print("--- RSI(14): 今の実装(30 を割ったら即買い)と、理論(売られすぎから戻ったら買い・レンジのときだけ・トレンドの向きだけ)")
r=d.col(rsi,c,n=14)
naive=np.where(r<30,1,np.where(r>70,-1,0)); wil=np.where(crossup(r,30),1,np.where(crossdn(r,70),-1,0))
line("rsi14 naive <30 buy, exit 50", hold(naive,r>=50,r<=50,20))
line("rsi14 cross back above 30, exit 50", hold(wil,r>=50,r<=50,20))
line("rsi14 cross back, range only (adx<20)", hold(np.where(ax<20,wil,0),r>=50,r<=50,20))
line("rsi14 cross back, exit opposite zone 70", hold(wil,r>=70,r<=30,40))
# Cardwell / Brown: 上昇トレンドでは RSI は 40〜80 で動く。40 までの押しで買い、下降トレンドは 60 までの戻りで売り
for lv in (40,45):
    e=np.where((tr200>0)&crossup(r,lv),1,np.where((tr200<0)&crossdn(r,100-lv),-1,0))
    line(f"rsi14 range rule: uptrend cross up {lv}, exit 70/30 or 20d", hold(e,r>=70,r<=30,20))
    e2=np.where((tr200>0)&(r<lv),1,np.where((tr200<0)&(r>100-lv),-1,0))
    line(f"rsi14 range rule: uptrend rsi<{lv}, exit 60/40", hold(e2,r>=60,r<=40,20))
# ダイバージェンス: 20 日の安値を更新したのに RSI は前の安値のときより高い
l20=shift(d.col(lowest,lo,n=20)); h20=shift(d.col(highest,h,n=20))
rmin=shift(d.col(lowest,r,n=20)); rmax=shift(d.col(highest,r,n=20))
div=np.where((lo<l20)&(r>rmin+3)&(r<40),1,np.where((h>h20)&(r<rmax-3)&(r>60),-1,0))
for hd in (5,10,20): line(f"rsi14 divergence at 20d extreme hold{hd}", hold(div,false,false,hd))
print("--- ボリンジャー(20, 2): 今の実装(外に出たら即逆張り)と、理論(横ばいのときの逆張り・バンドウォークは順張り)")
mid=d.col(sma,c,n=20); sd=d.col(rolling_std,c,n=20); z=(c-mid)/sd
slope=np.abs(mid-shift(mid,5))/a20
nb=np.where(z<-2,1,np.where(z>2,-1,0))
line("bb naive fade, exit mid", hold(nb,z>=0,z<=0,40))
line("bb fade when mid flat (slope<0.3atr/5d)", hold(np.where(slope<0.3,nb,0),z>=0,z<=0,40))
line("bb fade when adx<20", hold(np.where(ax<20,nb,0),z>=0,z<=0,40))
back=np.where(crossup(z,-2),1,np.where(crossdn(z,2),-1,0))
line("bb re-entry inside band (cross back), exit mid", hold(back,z>=0,z<=0,40))
width=sd/mid; wexp=width>shift(width)
walk=np.where((z>2)&wexp&(ax>25),1,np.where((z<-2)&wexp&(ax>25),-1,0))
line("bb band walk follow (z>2, width expanding, adx>25), exit mid", hold(walk,z<=0,z>=0,60))
line("bb band walk follow, exit z<1", hold(walk,z<1,z>-1,60))
print("--- 一目: 今の実装(転換線の交差の足に雲の上)と、三役好転(転換>基準・雲の上・遅行>26 本前)")
I=d.col(lambda hh,ll: tuple(ichimoku(hh,ll)[k] for k in ("tenkan","kijun","span_a","span_b")),h,lo)
tk,kj,sa,sb=I; top=np.fmax(sa,sb); bot=np.fmin(sa,sb); dd=tk-kj
cur=np.where(crossup(dd,0)&(c>top),1,np.where(crossdn(dd,0)&(c<bot),-1,0))
line("ichimoku current impl", hold(cur,(dd<0)|(c<top),(dd>0)|(c>bot)))
up3=(dd>0)&(c>top)&(c>shift(c,26)); dn3=(dd<0)&(c<bot)&(c<shift(c,26))
line("ichimoku sanyaku state (hold while all three)", np.where(up3,1,np.where(dn3,-1,0)))
e3=np.where(up3&~(shift(up3.astype(float))>0),1,np.where(dn3&~(shift(dn3.astype(float))>0),-1,0))
line("ichimoku sanyaku entry, exit close<kijun", hold(e3,c<kj,c>kj))
line("ichimoku sanyaku entry, exit close into cloud", hold(e3,c<top,c>bot))
print("--- MACD: 今の実装(交差でドテン)と、理論(ゼロ線の下での買い交差・トレンドの向きだけ・ゼロ線の交差)")
m,sg=d.col(macd,c); df=m-sg
line("macd current (stop & reverse)", np.sign(df))
ez=np.where(crossup(df,0)&(m<0),1,np.where(crossdn(df,0)&(m>0),-1,0))
line("macd cross below zero buy / above zero sell, exit opp cross", hold(ez,df<0,df>0))
et=np.where(crossup(df,0)&(tr200>0),1,np.where(crossdn(df,0)&(tr200<0),-1,0))
line("macd cross with sma200 trend, exit opp cross", hold(et,df<0,df>0))
etz=np.where(crossup(df,0)&(tr200>0)&(m<0),1,np.where(crossdn(df,0)&(tr200<0)&(m>0),-1,0))
line("macd cross with trend AND pullback side of zero", hold(etz,df<0,df>0))
line("macd zero-line (sign of macd)", np.sign(m))
print("--- グランビル: 上向きの 200 日平均への押しで買い(法則 2・3)")
rising=sma200>shift(sma200,20); falling=sma200<shift(sma200,20); dist=(c-sma200)/a20
g2=np.where(rising&(dist<1)&(dist>-2)&(c>shift(c)),1,np.where(falling&(dist>-1)&(dist<2)&(c<shift(c)),-1,0))
for hd in (10,20,40): line(f"granville: rising ma200, within 1atr, up day, hold{hd}", hold(g2,dist<-2,dist>2,hd))
print("--- チャネル抜け: 今の実装(終値で確認、翌足)と、タートルの原典(その足の中で、抜けた水準で逆指値)")
def donchian_intrabar(n_in,n_out,stop_n=2.0):
    net=np.zeros(c.shape); ntr=0
    for j in range(10):
        ok=np.isfinite(c[:,j]); idx=np.flatnonzero(ok)
        O,H,L,C=o[ok,j],h[ok,j],lo[ok,j],c[ok,j]; N=atr(H,L,C,20)
        hi=np.r_[np.nan,highest(H,n_in)[:-1]]; lw=np.r_[np.nan,lowest(L,n_in)[:-1]]
        xl=np.r_[np.nan,lowest(L,n_out)[:-1]]; xh=np.r_[np.nan,highest(H,n_out)[:-1]]
        side=0; px=0; stop=0
        for i in range(60,len(C)):
            if side!=0:
                lvl=max(xl[i],stop) if side>0 else min(xh[i],stop)
                if (L[i]<=lvl if side>0 else H[i]>=lvl):
                    ex=min(lvl,O[i]) if side>0 else max(lvl,O[i])
                    net[idx[i],j]+=side*np.log(ex/px)-2*COST; ntr+=1; side=0
                    continue
            if side==0 and np.isfinite(hi[i]) and np.isfinite(N[i-1]):
                if H[i]>hi[i]: side=1; px=max(hi[i],O[i]); stop=px-stop_n*N[i-1]
                elif L[i]<lw[i]: side=-1; px=min(lw[i],O[i]); stop=px+stop_n*N[i-1]
    return net,ntr/32
for n_in,n_out in ((20,10),(55,20)):
    hi=shift(d.col(highest,h,n=n_in)); lw=shift(d.col(lowest,lo,n=n_in)); xl=shift(d.col(lowest,lo,n=n_out)); xs=shift(d.col(highest,h,n=n_out))
    line(f"donchian {n_in}/{n_out} close-confirmed (current)", hold(np.where(c>hi,1,np.where(c<lw,-1,0)),c<xl,c>xs))
    net,tr=donchian_intrabar(n_in,n_out); line(f"donchian {n_in}/{n_out} intrabar stop entry + 2N stop (turtle)", net=net,tr=tr)
