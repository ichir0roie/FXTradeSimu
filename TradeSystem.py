import time


import randomWorkMarket


class tradeSystem:
    def __init__(self):
        print("ts init")
        self.market=randomWorkMarket.randomWorkMarket()

        self.Balance = 0
        self.orders = []
        self.delay = 0.0
        self.candleLen = 0

    def onTick(self):
        time.sleep(self.delay)
        self.market.onTick()

    def view(self):
        # if viewFlag:
        if len(self.market.candle)>self.candleLen:
            self.candleLen+=10
            self.market.candleView(100)

    def buy(self):
        o = self.__order()
        o.direction = 1
        o.openPrice = self.market.values[-1]
        o.stopLoss = o.openPrice - 0.05
        o.takeProfit = o.openPrice + 0.1
        self.orders.append(o)
    def sell(self):
        o=self.__order()
        o.direction=-1
        o.openPrice=self.market.values[-1]
        o.stopLoss=o.openPrice+0.05
        o.takeProfit=o.openPrice-0.1
        self.orders.append(o)

    def close(self,position):
        if position>=len(self.orders):
            print("close error")
            return
        o=self.orders.pop(position)
        p=self.market.values[-1]-o.openPrice
        p*=o.direction
        self.Balance+=p


    class __order:
        def __init__(self):
            self.direction=0
            self.openPrice=0
            self.takeProfit=0
            self.stopLoss=0




if __name__ == "__main__":
    print("ts main")
    ts=tradeSystem()
    # while True:
    #     ts.onTick()
    #     ts.view()
    ts.buy()
    ts.sell()
    ts.close(0)
    print(ts.orders)