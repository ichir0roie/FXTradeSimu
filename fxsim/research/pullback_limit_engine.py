"""E050: 指値で待つ押し目(S049)を、本物のエンジンで 2010-11〜2026-11 を通しで回す(主要 10 ペア、入りの深さ 4 通り)。

    python -m fxsim.research.pullback_limit_engine

pair-eval は 4 年ずつに切るので、200 日の移動平均の助走で各期間の 2 割を失う。ここでは 1 ペア 1 本で通して回す。
量は S049 のまま(損切りで有効証拠金の 0.5% を失う量 = ボラが小さいときほど大きい)。bp は建値比で、量を掛けない。
"""
# ruff: noqa: E501, E701, E702, E741, I001
import fxsim.markets.models  # noqa: F401
import fxsim.strategies.builtin  # noqa: F401
import fxsim.strategies.main  # noqa: F401
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from fxsim.backtest.broker import BrokerConfig
from fxsim.backtest.engine import EngineConfig
from fxsim.backtest.portfolio import build_backtester
from fxsim.markets import MARKETS
from fxsim.strategies import STRATEGIES
PAIRS=["USDJPY","EURUSD","GBPUSD","AUDUSD","NZDUSD","USDCHF","USDCAD","EURJPY","GBPJPY","AUDJPY"]
def run(a):
    pair,lim,vr=a
    md=MARKETS.create("real_pair",{"source":{"path":f"db:{pair}/H1","start":"2010-11-01","end":"2026-11-10"}}).generate(seed=1)
    s=STRATEGIES.create("main_pullback_limit",{"limit_atr":lim,"vol_ratio_min":vr})
    bt=build_backtester(md,s,BrokerConfig.preset("oanda"),EngineConfig(initial_balance=1e6),seed=1); bt.run()
    pip=0.01 if pair.endswith("JPY") else 0.0001
    return [(t.exit_t/1e9,(t.exit_price-t.entry_price)*t.side/pip,t.pnl,t.swap,(t.exit_price-t.entry_price)*t.side/t.entry_price*1e4) for t in bt.trades]
if __name__=="__main__":
    CASES=[(0.0,0.0),(0.25,0.0),(0.5,0.0),(0.75,0.0),(0.0,1.2),(0.5,1.0),(0.5,1.2),(0.5,1.4)]
    jobs=[(p,l,v) for l,v in CASES for p in PAIRS]
    with ProcessPoolExecutor(10) as ex: outs=list(ex.map(run,jobs))
    SPL=np.datetime64("2018-11-01").astype("datetime64[s]").astype(int)
    print("| 入り | 取引 / 年(10 ペア) | 1 取引 bp(スプレッド込み) | 損益の合計(円、スワップ込み) | うちスワップ | t(月) | 前半 bp | 後半 bp | プラスのペア | プラスの年 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for l,v in CASES:
        per=[np.array(o).reshape(-1,5) for (p,ll,vv),o in zip(jobs,outs,strict=True) if (ll,vv)==(l,v)]; a=np.vstack(per)
        mon=(a[:,0]//(86400*30.4375)).astype(int); ms=np.bincount(mon-mon.min(),weights=a[:,2]); ms=ms[np.bincount(mon-mon.min())>0]
        yr=(a[:,0]//(86400*365.25)).astype(int); ys=np.bincount(yr-yr.min(),weights=a[:,2])
        print(f"| {'成行' if l==0 else f'指値 {l}'}{f'・ボラの比 {v} 以上' if v else ''} | {len(a)/15.2:.0f} | {a[:,4].mean():+.2f} | {a[:,2].sum():+,.0f} | {a[:,3].sum():+,.0f} | {ms.mean()/(ms.std(ddof=1)/np.sqrt(len(ms))):+.2f} | {a[a[:,0]<SPL,4].mean():+.2f} | {a[a[:,0]>=SPL,4].mean():+.2f} | {sum(x[:,2].sum()>0 for x in per)}/10 | {(ys>0).sum()}/{len(ys)} |")
