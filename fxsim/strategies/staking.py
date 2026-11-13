"""賭け金の決め方(マネーマネジメント)。売買の判断とは別に、次の取引の数量を決める。

カジノの古典(マーチンゲールなど)は「負けを取り返す」「勝ちを伸ばす」ように数量を変えるだけで、
1 回ごとの期待値(コストの分だけマイナス)は変えない。
変わるのは損益の分布の形(小さな勝ちが多く、まれに大きく負ける、など)。
数量は基準の数量(1 単位)の倍数で返す。上限 max_mult で打ち止めにし、負けが続いたら 1 単位に戻す。

| 名前 | 負けたら | 勝ったら |
|---|---|---|
| flat | 同じ | 同じ |
| martingale | 2 倍 | 1 単位に戻す |
| paroli(逆マーチンゲール) | 1 単位に戻す | 2 倍(3 連勝で戻す) |
| dalembert | +1 単位 | −1 単位 |
| fibonacci | 数列を 1 つ進める | 2 つ戻す |
| labouchere | 数列の両端の和を賭け、負けたら末尾に足す | 両端を消す。数列が空になったら最初から |
| oscars_grind | 同じ | 1 単位足す(1 周の目標 +1 単位に届く分まで) |
| kelly | 過去の取引の勝率と損益比から f = p − (1 − p) / b を推定し、その half を賭ける(リスク基準) | |
"""
from __future__ import annotations

from dataclasses import dataclass, field

STAKING_SYSTEMS = ("flat", "martingale", "paroli", "dalembert", "fibonacci", "labouchere", "oscars_grind", "kelly")


@dataclass
class Staker:
    system: str = "flat"
    max_mult: float = 64.0          # 1 単位の何倍まで賭けるか(打ち止め)
    paroli_wins: int = 3            # paroli: この回数勝ったら 1 単位に戻す
    labouchere_seq: tuple = (1, 2, 3)
    kelly_fraction: float = 0.5     # kelly: 推定した f のこの割合だけ賭ける
    kelly_min_trades: int = 30
    kelly_window: int = 200
    mult: float = 1.0
    _fib: int = 0
    _seq: list = field(default_factory=list)
    _streak: int = 0
    _grind: float = 0.0             # oscars_grind: 今の周の損益(単位)
    _hist: list = field(default_factory=list)

    def __post_init__(self):
        if self.system not in STAKING_SYSTEMS:
            raise ValueError(f"unknown staking system {self.system!r}. known: {STAKING_SYSTEMS}")
        self._seq = list(self.labouchere_seq)
        self.mult = self._first()

    def _first(self) -> float:
        if self.system == "labouchere":
            return float(self._seq[0] + self._seq[-1]) if len(self._seq) > 1 else float(self._seq[0])
        return 1.0

    @staticmethod
    def _fib_n(k: int) -> int:
        a, b = 1, 1
        for _ in range(k):
            a, b = b, a + b
        return a

    def update(self, r: float) -> None:
        """直前の取引の結果を教える。r は 1 単位あたりの損益(リスク 1 単位で割った R など、符号が大事)。"""
        win = r > 0
        s = self.system
        self._hist.append(r)
        if s == "martingale":
            self.mult = 1.0 if win else self.mult * 2
        elif s == "paroli":
            self._streak = self._streak + 1 if win else 0
            if not win or self._streak >= self.paroli_wins:
                self._streak, self.mult = 0, 1.0
            else:
                self.mult *= 2
        elif s == "dalembert":
            self.mult = max(1.0, self.mult + (-1 if win else 1))
        elif s == "fibonacci":
            self._fib = max(0, self._fib - 2) if win else self._fib + 1
            self.mult = float(self._fib_n(self._fib))
        elif s == "labouchere":
            if win:
                self._seq = self._seq[1:-1]
            else:
                self._seq.append(int(round(self.mult)))
            if not self._seq:
                self._seq = list(self.labouchere_seq)
            self.mult = self._first()
        elif s == "oscars_grind":
            self._grind += self.mult * (1 if win else -1)
            if self._grind >= 1:
                self._grind, self.mult = 0.0, 1.0
            elif win:
                self.mult = min(self.mult + 1, 1 - self._grind)
        if self.mult > self.max_mult:          # 打ち止め: 取り返すのをあきらめて 1 単位からやり直す
            self.reset()

    def reset(self) -> None:
        self.mult = 1.0
        self._fib, self._streak, self._grind = 0, 0, 0.0
        self._seq = list(self.labouchere_seq)
        self.mult = self._first()

    def kelly_f(self) -> float:
        """過去の取引から推定したケリーの割合(1 回のリスクに対する賭け金の割合)。推定できなければ 0。"""
        h = self._hist[-self.kelly_window:]
        if len(h) < self.kelly_min_trades:
            return 0.0
        wins = [x for x in h if x > 0]
        losses = [-x for x in h if x < 0]
        if not wins or not losses:
            return 0.0
        p = len(wins) / len(h)
        b = (sum(wins) / len(wins)) / (sum(losses) / len(losses))
        return max(0.0, p - (1 - p) / b)
