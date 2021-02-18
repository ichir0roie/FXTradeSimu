import time
import TradeSystem
import iMa

import matplotlib.pyplot as plt



class simpleFollower(TradeSystem.tradeSystem):

    def __init__(self):
        super().__init__()
        self.Balance = 1000000

        self.tradeCount=0
        self.tradeLog=[]

    def onTick(self):
        super().onTick()
        time.sleep(self.delay)
        self.market.onTick()

        now=self.market.values[-1]
        before=iMa.simple(self.market.candle[-20:-10])
        after=iMa.simple(self.market.candle[-10:])

        acc=after-before

        if len(self.orders)==0:
            if acc<-1:
                self.buy()
                self.tradeCount+=1
        else:
            if acc>0:
                self.close(0)
                self.view()

    def view(self):
        if self.tradeCount%100!=0:
            return
        #self.market.candleView(100)
        elapse=self.market.candleIndex[-1] - self.market.dateBase
        print(str(elapse)+","+str(self.tradeCount)+","+str(int(self.Balance)))

        self.tradeLog.append(self.Balance)

        # if self.tradeCount%1000==0:
        #     plt.plot([i for i in range(len(self.tradeLog))],self.tradeLog)
        #     plt.show()

        # before = iMa.simple(self.market.candle[-20:-10])
        # after = iMa.simple(self.market.candle[-10:])
        # print(self.market.candle[-20:])
        # print(before)
        # print(after)
        # print("view")


if __name__ == "__main__":
    print("ts main")
    ts=simpleFollower()
    while True:
        ts.onTick()