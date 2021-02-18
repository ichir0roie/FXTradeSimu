import RandomWorkMarket
import random
import datetime

import glob

import PyPDF2 as ppdf

def someChartCreateAndSave(times):
    for i in range(times):
        market = RandomWorkMarket.RandomWorkMarket()
        for j in range(random.randint(100,10000)):
            market.onTick()
        name="CandlePdf/"+datetime.datetime.now().strftime("candle_%Y%m%d_%H%M%S")+"_"+str(i)+".pdf"
        market.candleSave(200,name)

def fileIntegration():
    files=glob.glob("CandlePdf/*.pdf")
    print(files)

    merger=ppdf.PdfFileMerger()
    for i in files:
        merger.append(i)
    name="MergedCandle/"+datetime.datetime.now().strftime("mergedCandle_%Y%m%d_%H%M%S")+".pdf"
    merger.write(name)


if __name__ == "__main__":
    # market = RandomWorkMarket()
    # for i in range(10000):
    #     market.onTick()

    # plot.plot(market.genCounts(market.values), market.values)
    # plot.show()

    # market.candleView(200)

    # someChartCreateAndSave(5)
    fileIntegration()