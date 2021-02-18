import random
import time
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

import mplfinance as mpl

import datetime

class randomWorkMarket:
    def __init__(self):
        print("rw init")


        self.values=[]

        self.candleIndex=[]
        self.candle=[]
        
        self.values.append(100.0)
        
        self.range=60*5

        self.dateBase=datetime.datetime.now()

        for i in range(self.range*100):
            print("set up : "+str(int(i/self.range))+"%")
            self.onTick()


    def getVec(self):
        r=random.random()
        r-=0.5
        r*=2

        b=random.random()-0.5
        b*=1
        v=self.values[-1]+r*b

        self.values.append(v)

    def onTick(self):
        # print(len(rw.values))
        v=self.getVec()

        if len(self.values)%self.range==0:
            self.createCandle()
        return v
    def onCandle(self):
        for i in range(self.range):
            self.onTick()

    def createCandle(self):#サイズがちょっとずれてるバグ、影響軽微のだめ無視
        if len(self.candleIndex)==0:
            self.candleIndex.append(self.dateBase)
        else:
            nd=datetime.timedelta(seconds=self.range)
            self.candleIndex.append(self.candleIndex[-1]+nd)
        buf=[]
        buf.append(self.values[-self.range])
        sl=sorted(self.values[-self.range:-1])
        buf.append(sl[0])
        buf.append(sl[-1])
        buf.append(self.values[-1])
        self.candle.append(buf)


    def view(self):
        print(self.values[-1])

    def candleView(self,Length):
        df = pd.DataFrame(self.candle[-Length:],index=self.candleIndex[-Length:],columns=("Open","High","Low","Close"))
        #print(df)
        mpl.plot(df,type="candle")
        time.sleep(1)


    def genCounts(self,array):
        return [i for i in range(len(array))]


if __name__=="__main__":
    rw=randomWorkMarket()
    for i in range(100000):
        rw.onTick()



    plot.plot(rw.genCounts(rw.values),rw.values)
    plot.show()

    rw.candleView()
