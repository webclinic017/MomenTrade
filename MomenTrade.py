import backtrader as bt
import pandas as pd
from datetime import datetime
from backtrader.feeds import GenericCSVData
import numpy as np
import matplotlib


tq10_corUp, tq10_corDown = ["#7F7F7F", "#17BECF"]  # plotly
tq09_corUp, tq09_corDown = ["#B61000", "#0061B3"]
tq08_corUp, tq08_corDown = ["#FB3320", "#020AF0"]
tq07_corUp, tq07_corDown = ["#E1440F", "#B0F76D"]
tq06_corUp, tq06_corDown = ["#FF3333", "#47D8D8"]
tq05_corUp, tq05_corDown = ["#FB0200", "#007E00"]
tq04_corUp, tq04_corDown = ["#18DEF5", "#E38323"]
tq03_corUp, tq03_corDown = ["black", "blue"]
tq02_corUp, tq02_corDown = ["red", "blue"]
tq01_corUp, tq01_corDown = ["red", "lime"]
#
tq_ksty01 = dict(
    volup=tq01_corUp, voldown=tq01_corDown, barup=tq01_corUp, bardown=tq01_corDown
)
tq_ksty02 = dict(
    volup=tq02_corUp, voldown=tq02_corDown, barup=tq02_corUp, bardown=tq02_corDown
)
tq_ksty03 = dict(
    volup=tq03_corUp, voldown=tq03_corDown, barup=tq03_corUp, bardown=tq03_corDown
)
tq_ksty04 = dict(
    volup=tq04_corUp, voldown=tq04_corDown, barup=tq04_corUp, bardown=tq04_corDown
)
tq_ksty05 = dict(
    volup=tq05_corUp, voldown=tq05_corDown, barup=tq05_corUp, bardown=tq05_corDown
)
tq_ksty06 = dict(
    volup=tq06_corUp, voldown=tq06_corDown, barup=tq06_corUp, bardown=tq06_corDown
)
tq_ksty07 = dict(
    volup=tq07_corUp, voldown=tq07_corDown, barup=tq07_corUp, bardown=tq07_corDown
)
tq_ksty08 = dict(
    volup=tq08_corUp, voldown=tq08_corDown, barup=tq08_corUp, bardown=tq08_corDown
)
tq_ksty09 = dict(
    volup=tq09_corUp, voldown=tq09_corDown, barup=tq09_corUp, bardown=tq09_corDown
)
tq_ksty10 = dict(
    volup=tq10_corUp, voldown=tq10_corDown, barup=tq10_corUp, bardown=tq10_corDown
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


def main(date1, date2, code):

    # 自定义策略
    class MyStrategy(bt.Strategy):
        state = ""
        price = -1

        def __init__(self):
            self.sma20 = bt.indicators.SimpleMovingAverage(self.data.close, period=20)
            self.stddev20 = bt.indicators.StandardDeviation(self.data.close, period=20)
            self.upper_track = self.sma20 + self.stddev20  # 布林带上轨
            self.lower_track = self.sma20 - self.stddev20  # 布林带下轨

        def posein(self, pos, line):
            if -pos > len(line):
                return pos, 0, False
            tmp_pos = pos - 1
            sum = line[pos]
            while len(line) >= -tmp_pos - 1 and (
                line[pos] * line[tmp_pos] > 0
                or line[pos] * (line[tmp_pos] + line[tmp_pos - 1]) > 0
            ):
                sum += line[tmp_pos]
                tmp_pos -= 1
            return tmp_pos, sum, pos - tmp_pos

        def next(self):
            if self.state == "stop":
                if (
                    self.data.close[0] > self.data.open[0]
                    and self.data.close[-1] > self.data.open[-1]
                ):
                    cash = self.broker.getcash()
                    price = self.data.close[0]
                    self.buy(size=int(cash / price * 0.9))
                    self.state = "open"
                    return
            if self.state == "open":
                if (
                    self.data.close[0] < self.data.open[0]
                    and self.data.close[-1] < self.data.open[-1]
                ):
                    self.sell(size=self.position.size)
                    self.state = "stop"
                    return
                if self.data.close[0] < self.upper_track[0]:
                    return
                self.sell(size=self.position.size)
                self.state = "close"
                return

            if self.data.close[0] > self.lower_track[0]:
                return
            if len(self.data) > 60:
                line = []
                open = []
                close = []
                for i in range(59, -1, -1):
                    line.append(self.data.close[-i] - self.data.open[-i])
                    open.append(self.data.open[-i])
                    close.append(self.data.close[-i])
                if (
                    line[-1] > 0
                    and line[-2] <= 0
                    and line[-1] + line[-2] > 0
                    and line[-3] < line[-2]
                ):
                    price = self.data.close[0]
                    self.price = price
                    pos = -2
                    pos, neg_sum1, neg_len1 = self.posein(pos, line)
                    neg_min1 = min(close[pos + 1 : pos + neg_len1 + 1])
                    pos, pos_sum1, pos_len1 = self.posein(pos, line)
                    pos_max1 = max(close[pos + 1 : pos + pos_len1 + 1])
                    pos, neg_sum2, neg_len2 = self.posein(pos, line)
                    neg_min2 = min(close[pos + 1 : pos + neg_len2 + 1])
                    pos, pos_sum2, pos_len2 = self.posein(pos, line)
                    pos_max2 = max(close[pos + 1 : pos + pos_len2 + 1])
                    pos, neg_sum3, neg_len3 = self.posein(pos, line)
                    neg_max3 = max(open[pos + 1 : pos + neg_len3 + 1])
                    neg_min3 = min(close[pos + 1 : pos + neg_len3 + 1])
                    distinct = abs(neg_len2 - neg_len3)
                    min_ = max(min(neg_len2, neg_len3) - distinct, 2)
                    max_ = max(neg_len2, neg_len3) + distinct
                    if (
                        neg_sum3 < neg_sum2
                        and neg_sum2 < neg_sum1
                        and pos_sum1 > pos_sum2
                        and neg_max3 > pos_max2
                        and neg_min3 > pos_max1
                        and neg_min1 < neg_min2
                        and neg_min2 < neg_min3
                        and neg_len1 < max_
                        and neg_len1 > min_
                        and pos_len1 >= min_
                        and pos_len2 >= min_
                        and neg_len2 >= 2
                        and neg_len3 >= 2
                    ):
                        time = self.data.datetime.date(0)
                        cash = self.broker.getcash()
                        self.buy(size=int(cash / price * 0.9))
                        self.state = "open"

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
    cerebro.addstrategy(MyStrategy)

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
    date1 = "20230830"
    date2 = "20240830"
    main(date1, date2, "002790")
