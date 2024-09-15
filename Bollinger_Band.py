import backtrader as bt
import matplotlib
import pandas as pd
from datetime import datetime
from backtrader.feeds import GenericCSVData
import numpy as np

tq07_corUp, tq07_corDown = ["#E1440F", "#B0F76D"]
tq_ksty07 = dict(
    volup=tq07_corUp, voldown=tq07_corDown, barup=tq07_corUp, bardown=tq07_corDown
)


# 定义自定义数据类
class MyCSVData(GenericCSVData):
    # 定义列名映射到Backtrader字段
    lines = ("datetime", "open", "close", "high", "low", "volume")
    params = (
        ("dtformat", "%Y-%m-%d"),  # 日期格式
        ("datetime", 0),  # 日期列索引
        ("open", 1),  # 开盘价列索引
        ("close", 2),  # 收盘价列索引
        ("high", 3),  # 最高价列索引
        ("low", 4),  # 最低价列索引
        ("volume", 5),  # 成交量列索引
        ("openinterest", -1),  # 开仓兴趣列索引，如果不存在可以设置为-1
    )


class Bollinger(bt.Strategy):
    buy_ = False

    def __init__(self):
        self.sma20 = bt.indicators.SimpleMovingAverage(self.data.close, period=20)
        self.stddev20 = bt.indicators.StandardDeviation(self.data.close, period=20)
        self.upper_track = self.sma20 + 2 * self.stddev20  # 布林带上轨
        self.lower_track = self.sma20 - 2 * self.stddev20  # 布林带下轨

    def next(self):
        size = int(self.broker.getcash() / self.data.close[0] * 0.9)

        # 检查是否需要开仓
        if not self.position:
            if self.data.close[0] <= self.lower_track[0]:
                self.buy_price = self.data.close[0]  # 初始化buy_price
                self.buy(size=size)
                self.buy_ = True

        # 检查是否需要平仓，增加对None的检查
        elif self.buy_:
            if self.data.close[0] >= self.upper_track[0]:
                self.sell(size=self.position.size)
                self.buy_ = False


def main(date1, date2, code):

    cerebro = bt.Cerebro()

    date_format = "%Y%m%d"
    from_date = datetime.strptime(date1, date_format)
    to_date = datetime.strptime(date2, date_format)

    # 加载数据
    data = MyCSVData(
        dataname="stock_data.csv",
        fromdate=from_date,
        todate=to_date,
    )

    # 添加数据到回测引擎
    cerebro.adddata(data)

    # 添加策略
    cerebro.addstrategy(Bollinger)

    # 设置初始资金
    start_cash = 100000.0
    cerebro.broker.setcash(start_cash)

    # 运行回测
    cerebro.run()

    # 获取回测结束后的总资金
    end_cash = cerebro.broker.getvalue()

    # 计算总收益率
    total_return_percentage = ((end_cash - start_cash) / start_cash) * 100

    # 假设回测周期是从date1到date2，计算总天数
    total_days = (to_date - from_date).days
    # 假设一年有365天，计算平均年化收益率
    annualized_return_percentage = (
        (1 + total_return_percentage / 100) ** (365 / total_days)
    ) - 1
    annualized_return_percentage *= 100  # 转换为百分比形式

    # 打印最终的资产价值、总收益率和平均年化收益率
    print(f"Final Portfolio Value: {end_cash:.2f}")
    print(f"Total Return Percentage: {total_return_percentage:.2f}%")
    print(f"Average Annualized Return Percentage: {annualized_return_percentage:.2f}%")

    matplotlib.use("agg")
    if abs(annualized_return_percentage) > 0.1:
        figs = cerebro.plot(style="candle", **tq_ksty07)
        fig = figs[0][0]
        fig.savefig("photo/" + code + ".png")
    return annualized_return_percentage


if __name__ == "__main__":
    date1 = "20230101"
    date2 = "20230630"
    main(date1, date2, "300750")
