# MomenTrade 动量交易

根据缠论的笔，对每日的K线进行识别，并输出买卖信号。

## 使用方法

运行 daily_predict ，有可能输出"已处理...总共...“，直接忽略。若结尾输出 "open … " 则第二天开盘买入，"stop ..." 或者"close ..."则第二天开盘卖出。
log.txt 内记录了当前的持股状态， open … 表示持股，为空表示未持股，stop... 表示虽然未持股但是后续会找机会重入，close ... 表示已经卖出。

如有需要，可以修改log.txt为当前的仓位。

## 特别鸣谢
借用了https://github.com/YuYuKunKun/chanlun.py 关于缠论的代码。
借用了https://github.com/myhhub/stock 中网络爬虫的代码。

好吧😂核心的都不是我写的，没有他们代码的贡献，不可能有这个项目。

python版本3.11.7
