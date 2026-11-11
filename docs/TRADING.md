# MT5 の口座で戦略を常駐させる(fxsim trade)

動作ケース(`cases/behaviors/B###`)の戦略を、MetaTrader 5 の口座で動かし続ける。
戦略のコードはバックテストと同じものをそのまま使う。既定ではデモ口座でしか動かない。

これまでの検証(E011)では、合格した動作ケースはほぼ無い。ここでの目的は、シミュレーションと実口座のずれ
(約定・スプレッド・スワップ・ブローカーの振る舞い)を確かめることで、稼ぐことではない。

## 仕組み

足が 1 本確定するたびに、次の順で 1 回だけ判断する(`fxsim/trading/`)。

1. MT5 から、起点の足からいま作られている足までを取る(`feed.py` / `mt5.py`)。
   起点は初回に `history_bars` 本遡った足で、`data/trading/<name>/state.json` に残し、以後は動かさない。
   毎回同じ足からバックテストを回すので、戦略の中の状態(前回の取引の結果など)は回すたびに同じになる
2. その足で Backtester を回し、最後の確定足で戦略が出した注文と、その時点のシミュレーションの建玉を取り出す(`signals.py`)。
   シミュレーションのブローカーは既定で `fair`(滑りなし・スワップなし)。実口座で起きていない狩りや拒否で
   シミュレーションだけが手仕舞うのを避けるため
3. 実口座の建玉(`magic` が同じものだけ)と突き合わせて、出す注文を決める(`trader.py` の `plan`)
   - シミュレーションが手仕舞った(損切り・利確・ロスカット)/ 向きが逆 → 実口座も決済する
   - 実口座だけ損切りに掛かった / 起動したときにシミュレーションがすでに持っていた → 建て直さない。
     次にシミュレーションが注文を出すまで待つ
   - 注文(enter / close / add / reduce)は Backtester と同じ決まりで流す。損切り・利確は約定値からの距離
   - 同じ建玉のまま損切り・利確が動いたら(トレーリング)、実口座も同じ価格に動かす。すでに越えた価格ならその場で決済
4. 注文を出し、判断と結果を `data/trading/<name>/journal.jsonl` に 1 行ずつ残す。
   注文を出す前に「この足は処理済み」を書くので、途中で落ちても同じ足の注文を二重に出さない

数量は「動作ケースの数量(通貨)x `volume_scale`」をロットに直し、`volume_step` で切り捨てる
(10,000 通貨 = 0.1 ロット)。`max_lots` を超える注文は出さない。

## 口座の床(資産の 9 割を守る)

`protect`(既定 0.9)で、実口座の有効証拠金に床を置く(`trader.py`、決まりは `guards.EquityFloor` と同じ)。

- 床 = 月初の有効証拠金 x `protect`。月が変わって最初に見た有効証拠金で付け直す(UTC の月)。
  同じ口座の常駐どうしで `data/trading/account-<login>.json` を共有する(dry-run は `account-<login>.dry.json`)
- 足を待つ間も `floor_check_seconds`(既定 60 秒)ごとと、注文を出す直前に有効証拠金を比べる。
  割ったら、この常駐の建玉を全部決済し、その月は新規(open / add)を出さない。手動やほかの EA の建玉には触らない
- 新規の量は、口座の全建玉(銘柄・magic を問わない)とその新規が、今の値段からサーバーの逆指値まで飛んで、
  さらに窓で `gap_mult`(既定 1.5)倍滑っても床を割らない量まで減らす。損益は MT5 の `order_calc_profit` で口座の通貨に直す。
  床に近づくほど量が小さくなり、最小ロットに届かなければ出さない(journal に `floor_skip`)
- 逆指値の無い建玉が口座に 1 つでもあると、最悪の損失が分からないので新規を出さない。逆指値を置かない戦略や、
  逆指値なしの手動の建玉が同じ口座にあると、床のある常駐も新規を出せなくなる。そういう戦略は `protect: 0` の別の口座で動かす
- シミュレーションの中の床は切る(シミュレーションの有効証拠金は実口座と別物)。バックテスト(MT5 のテスター)では
  戦略が自分で床を持ち、足の確定でだけ比べる
- 窓が `gap_mult` を超えて飛べば、床を割ることはある(決済はその後になる)

## 処理の負荷

- 足を見に行く間隔は `poll_seconds: auto`(既定)で足の長さに合わせる: 足の長さ / 120 を 5〜60 秒に収めたもの
  (日足・4 時間足 60 秒、1 時間足 30 秒、15 分足 7.5 秒、5 分足 5 秒)。足が変わっていなければ、作りかけの足の時刻を 1 本取るだけ
- 足が確定したときだけ、起点からの足でシミュレーションを回す(日足の 3,000 本なら 1 秒かからない)
- 床は `floor_check_seconds` ごとに有効証拠金を 1 回取るだけ
- systemd では `Nice=10` と `CPUWeight=20` で、手元の作業より後回しにする(下のユニット)

## 用意するもの

MetaTrader5 の Python パッケージは Windows でしか動かない。

| 動かす場所 | backend | 用意 |
|---|---|---|
| Linux + Wine(既定) | `rpyc` | 下の「Wine の用意」。Wine の中の Windows 版 Python で `fxsim/trading/wine_server.py` を動かし、Linux 側から rpyc でつなぐ |
| Windows | `metatrader5` | MT5 の端末と `pip install MetaTrader5` |

### Wine の用意(Linux)

Ubuntu 26.04 で、WineHQ の wine-staging 11.21 と OANDA のデモ口座で確かめた手順(2026-11)。
専用の環境 `~/.wine-mt5` に入れる。

Ubuntu の Wine(10.0 repack)では、MT5 の端末が「A debugger has been found running in your system」と出して
起動しない(本物のデバッガーは無い。MT5 の検知が Wine の版によって誤って反応する。WineHQ の bug 57950 / 59352)。
WineHQ の wine-staging を `/opt/wine-staging` に入れて使う(Ubuntu の Wine は残してよい)。

```bash
sudo dpkg --add-architecture i386
sudo mkdir -pm755 /etc/apt/keyrings
wget -O - https://dl.winehq.org/wine-builds/winehq.key | sudo gpg --dearmor -o /etc/apt/keyrings/winehq-archive.key -
sudo wget -NP /etc/apt/sources.list.d/ https://dl.winehq.org/wine-builds/ubuntu/dists/resolute/winehq-resolute.sources
sudo apt update
sudo apt install --install-recommends wine-staging   # 26.04 では -amd64 / -i386 に分かれていない

export WINE=/opt/wine-staging/bin/wine WINEPREFIX=$HOME/.wine-mt5 WINEARCH=win64
/opt/wine-staging/bin/wineboot -i                    # Wine Mono を入れるか聞かれたら Cancel でよい(MT5 は使わない)
# Windows 版の Python(python.org の python-3.11.9-amd64.exe)
$WINE python-3.11.9-amd64.exe /quiet InstallAllUsers=0 PrependPath=1 Include_test=0 TargetDir='C:\Python311'
# numpy 2 系は Wine で落ちる(ucrtbase.dll.crealf が未実装)ので 1.26 にする。rpyc は Linux 側と同じ版
$WINE 'C:\Python311\python.exe' -m pip install MetaTrader5 "numpy==1.26.4" "rpyc==6.0.2"
# MT5 の端末(OANDA なら https://download.mql5.com/cdn/web/oanda.corporation/mt5/oanda5setup.exe)
$WINE oanda5setup.exe
# Linux 側
.venv/bin/pip install "rpyc==6.0.2"
```

常駐する前に、Wine の中で橋を動かしておく。端末は、設定の `mt5.path` に OANDA の terminal64.exe を書いておけば
つないだときに起動し、`.env.<account>` の口座でログインする(ログインの直後は履歴が届くまで数秒かかる。待ってやり直す)。

```bash
export WINEPREFIX=$HOME/.wine-mt5
/opt/wine-staging/bin/wine 'C:\Python311\python.exe' fxsim/trading/wine_server.py --port 18812 &
```

```yaml
mt5:
  backend: rpyc
  path: "C:\\Program Files\\OANDA MetaTrader 5\\terminal64.exe"
```

OANDA のデモ(OANDA-Japan MT5 Demo)は、銘柄名 `USDJPY`(接尾辞なし)、ヘッジ口座、レバレッジ 25 倍、
サーバー時刻 = UTC + 2 時間(冬時間)だった。

橋(rpyc の classic サーバー)は、つないだ相手に何でも実行させる。127.0.0.1 だけで待ち受けるようにしてあるので、
ポートを外に出さないこと。

MT5 の端末で「アルゴリズム取引」を有効にしておく(無効だと注文が 10027 で拒否される)。

## 口座(デモと本番を分ける)

口座の番号・パスワード・サーバーは設定に書かない。リポジトリ直下の `.env.demo` / `.env.real`(コミットしない)に置き、
設定の `account` でどちらを読むかを選ぶ。書式は `.env.example`(これだけコミットする)。

```
MT5_LOGIN=12345678
MT5_PASSWORD='...'
MT5_SERVER='OANDA-Japan MT5 Demo'
```

| 設定 | 読むファイル | 起動の条件 |
|---|---|---|
| `account: demo`(既定) | `.env.demo` | つないだ口座がデモであること |
| `account: real` | `.env.real` | つないだ口座が本番であること。さらに `allow_real: true` |

- つないだ後、端末がログインしている口座の番号がファイルの `MT5_LOGIN` と同じか確かめる。違えば起動しない
- 環境変数は見ない(シェルに残ったデモの値で本番の設定が動く、といった取り違えを避ける)。ファイルが無ければ、
  端末がいまログインしている口座を使う(種別の照合はする)
- 1 つの端末は 1 つの口座にしかつながらない。デモと本番を同時に動かすなら、端末を別の場所に入れ(`mt5.path`)、
  橋も別のポートで動かす(`wine_server.py --port 18813` と `mt5.port`)。同じ端末を指すと、後からつないだ方が
  口座を切り替え、先の常駐は口座番号の照合で止まる

## 設定(`configs/trading/*.yaml`)

```yaml
name: demo-usdjpy-b001       # 状態と記録の置き場所 data/trading/<name>/
behavior: B001               # 動作ケース(戦略・パラメータ・足・取引時間・数量)
symbol: USDJPY               # ブローカーの銘柄名(USDJPY.m などの接尾辞に合わせる)
magic: 20261110              # この常駐の注文の印。設定ごとに変える。ほかの建玉には触らない
account: demo                # .env.demo を読む(本番は real と .env.real、allow_real: true)
volume_scale: 1.0
max_lots: 0.5
history_bars: 3000           # 起点。戦略の warmup より十分長く
server_utc_offset_hours: auto
mt5: {backend: rpyc, host: localhost, port: 18812, path: "C:\\Program Files\\OANDA MetaTrader 5\\terminal64.exe"}
```

ほかのキー(`fxsim/trading/config.py`): `max_history_bars`(起点からの本数の上限。超えたら起点を進める)、
`poll_seconds`(既定 auto)、`protect`(既定 0.9)、`gap_mult`(既定 1.5)、`floor_check_seconds`(既定 60)、`deviation_points`、`pip`、`sim_broker`、`dry_run`、`allow_real`、`comment`。

### 複数ペアの戦略(B047 など、`symbols` を書く)

```yaml
name: demo-majors-b047
behavior: B047
symbol: USDJPY               # 足の確定と時刻の基準(symbols の中の 1 つ)
symbols: [USDJPY, EURUSD, GBPUSD, AUDUSD, NZDUSD, USDCHF, USDCAD, EURJPY, GBPJPY, AUDJPY]
magic: 20261112
fill_server_hour: 2          # 判断はサーバーの 0 時、注文は 2 時まで待つ(ロールオーバーの 1 時間はスプレッドが数倍)
```

`symbols` があると `PortfolioTrader`(`fxsim/trading/portfolio.py`)で動く。流れは 1 銘柄の常駐と同じで、全ペアでまとめて行う。

- 全ペアの足を取り、全ペアにある足だけを残す。作りかけの足がまだ来ていないペアがあれば、その回は待つ(処理済みにしない)
- 決済通貨が円でないペアは、今の気配から「決済通貨 1 単位の円の値」を求めて価格に掛け、バックテスト(real_pairs)と同じ
  円の値段でシミュレーションする。仮の口座(動作ケースの `engine.initial_balance`、B047 は 10 万円)で PortfolioBacktester を回す
- ペアごとに実口座の建玉(magic が同じもの)と突き合わせる(下の決まりは 1 銘柄と同じ。途中からは乗らない)。
  損切り・利確の幅は、円に換算した値段を銘柄の値段に戻して置く
- `fill_server_hour` があれば、サーバーの時刻がその時になるまで注文を `state.json` の `pending` に置いて待つ。
  待っている間に次の足の判断が出たら古い注文は捨てる(journal に `pending_dropped`)。出す直前に消すので二重には出ない
- 口座の床は口座全体の有効証拠金で見る。割ったら symbols の全ペアの、この常駐の建玉を決済する
- 対円でないペアの円への換算に、USDJPY と、決済通貨のペア(USDCHF・USDCAD など)を symbols に入れておく
- 金利差はペアの短期金利の差(`data/real/short_rates_monthly_fred.csv`)。月に 1 回 `python -m fxsim fetch-fred` で取り直す

2026-11-11 にデモ口座の端末へ `trade check` でつないで確かめた(注文は出していない): 10 ペアの日足 2,959 本で
シミュレーションし、仮の口座は 10 万円 → 21.2 万円(バックテストと合う)。その時点では USDCHF だけが金利差と
キャリー・トゥ・リスクの条件を満たし、20 日の高値抜けを待っていた。

## 使い方

```bash
python -m fxsim trade check  configs/trading/demo-usdjpy-b001.yaml   # 接続・口座・建玉と、いまの足での判断(注文は出さない)
python -m fxsim trade run    configs/trading/demo-usdjpy-b001.yaml --dry-run   # 注文を出さずに常駐(記録は <name>.dry/)
python -m fxsim trade run    configs/trading/demo-usdjpy-b001.yaml   # 常駐
python -m fxsim trade status configs/trading/demo-usdjpy-b001.yaml   # 状態と直近の判断
```

切れたらつなぎ直し(5 秒から倍々、最大 5 分)、想定していない失敗でも止まらずに次の足で判断し直す。
SIGTERM / SIGINT で止まる。1 時間ごとに有効証拠金と建玉の数をログに出す。

### 1 分足の再生で 24 時間運用を確かめる(trade-replay)

実口座で何週間も待たずに、常駐が 24 時間動き続けられるかを、相場の DB(`fetch-mt5` で溜めた 1 分足)の再生の上で確かめる
(`fxsim/trading/replay.py`)。常駐のコード(Service・LiveTrader・MT5Client)はそのまま動かし、MetaTrader5 のモジュールと
時計だけを差し替える。3 か月を 2 本の常駐で回して実時間 8 分ほど。

```bash
python -m fxsim fetch-mt5 configs/market_data/oanda-demo.yaml --timeframes M1 H1 D1   # 再生の前に溜める
python -m fxsim trade-replay configs/trading/demo-usdjpy-b045.yaml configs/trading/demo-usdjpy-b001.yaml
python -m fxsim trade-replay ... --faults none --start 2026-09-01 --end 2026-10-01 --seed 2
```

- 偽の端末は、1 分足 1 本を 4 つのティックにして再生する。上の時間足はサーバー時刻で 1 分足を区切って作り、それより前は
  DB に溜めた同じ時間足を端末の履歴として返す(B045 は日足 200 本、B001 は 1 時間足 3,000 本が要るので、H1・D1 も溜めておく)
- 銘柄と口座の決まりは OANDA Japan のデモの端末から読んだ値(成行は即約定、IOC、ストップレベル 5 ポイント、
  スワップ +10.16 / −25.06 ポイント、レバレッジ 25 倍、ストップアウト 100%)。サーバー側の逆指値・スワップ・ストップアウトも動く
- 同じ設定の常駐は既定で同じ口座(床を共有する)。`--separate-accounts` で分ける。`--manual` で手動の建玉を足せる
- `--faults default` は、端末が落ちる・橋が切れる・常駐の再起動・注文の送信中の切断・一時的な拒否・ログインの直後の履歴待ち・
  PC のスリープを、シードで決まる時刻に入れる
- 記録は `data/trading/replay/<時刻>/`(`report.md`・`result.json`・`replay.log`・常駐ごとの journal と state)。
  report は、判断した足 / 確定した足・判断しなかった足・判断の遅れ・シミュレーションと実口座の食い違い・注文の結果・
  判断の CPU・端末の関数の呼び出しの回数

### systemd(ユーザー単位)で常駐させる

橋と常駐を別のユニットにする。橋が先に立ち上がり、端末は常駐がつないだときに起動する。

`~/.config/systemd/user/fxsim-mt5-bridge.service`:

```ini
[Unit]
Description=MetaTrader5 bridge in Wine (fxsim)

[Service]
WorkingDirectory=%h/work/FXTradeSimu
Environment=WINEPREFIX=%h/.wine-mt5 WINEDEBUG=-all
ExecStart=/opt/wine-staging/bin/wine C:\\Python311\\python.exe fxsim/trading/wine_server.py --port 18812
Nice=10
CPUWeight=20
Restart=on-failure
RestartSec=30

[Install]
WantedBy=default.target
```

`~/.config/systemd/user/fxsim-trade@.service`:

```ini
[Unit]
Description=fxsim trade %i
Requires=fxsim-mt5-bridge.service
After=fxsim-mt5-bridge.service network-online.target

[Service]
WorkingDirectory=%h/work/FXTradeSimu
ExecStart=%h/work/FXTradeSimu/.venv/bin/python -m fxsim trade run configs/trading/%i.yaml
Nice=10
CPUWeight=20
Restart=on-failure
RestartSec=30

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now fxsim-mt5-bridge fxsim-trade@demo-usdjpy-b001
journalctl --user -u fxsim-trade@demo-usdjpy-b001 -f
loginctl enable-linger $USER     # ログアウトしても動かし続けるなら
```

## 気をつけること

- 対応する足は M1 / M5 / M15 / M30 / H1 / H4 / D1。複数ペアの戦略は `symbols` を書けば動く(上の節)。
  1 分足の再生(trade-replay)はまだ 1 銘柄だけ
- `swap_side_only` の動作ケース(全部、E022)では、シミュレーションのブローカーのスワップの決め方(金利差・業者の取り分)を
  動作ケースのブローカーから引き継ぐ(`trader.sim_broker_for`)。既定の `sim_broker` はスワップを固定の 0 にしているので、
  引き継がないと「受け取る向き」が無く、新規が全部止まる(2026-11-11 に直した。それより前に動かした常駐は新規を出していなかった)
- 本命の戦略(`strategies/main/`、MT5 のテスターで回るもの)も動かせる。テスターは成行をその場で約定させるので、
  作りかけの足の始値で戦略を呼んだ前後の建玉の差を注文に直す(`signals.py` の `_SnapshotMt5Tester`)。
  建玉は 1 つまで。シミュレーションの口座の決まり(ロット・証拠金・両建て)は動作ケースの `broker.overrides.mt5` を使い、
  約定は `sim_broker`(既定 fair)のまま。量は戦略がシミュレーションの有効証拠金から決めるので、実口座の残高とは関係しない
  (合わせるなら `volume_scale`)。終値の損切りは戦略が足の確定で決済し、サーバーには非常用の逆指値だけを置く
- 時刻: MT5 の足の時刻はサーバー時刻。`auto` は気配の時刻と手元の時計の差(30 分単位)から推定する。
  市場が閉じている間に初めて起動するときは推定できないので、数値(例: `2`)で書く。
  夏時間で差が変わると、それより前の足の時刻は 1 時間ずれたまま扱う(取引時間の判定と日足の区切りに効く)
- 足の中の経路が無いので、シミュレーションの損切り・利確は足の高安で判定する(両方届いたら損切りが先)。
  実口座は MT5 のサーバーで約定するので、同じ足で片方だけ手仕舞うことがある。食い違いは上の決まりで次の足に揃える
- 判断は足が確定してから最初の問い合わせ(`poll_seconds` ごと)で行うので、約定は足の始値から最大でその間隔だけ遅れる
  (auto なら日足で最大 60 秒)
- 起点からの本数が `max_history_bars` を超えるか、MT5 の履歴が起点に届かなくなると、起点を進める(ログに警告)。
  そのとき戦略の中の状態が変わることがある
- `account: real` と `allow_real: true` の両方が無い限り、本番の口座では起動しない
