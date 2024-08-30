from datetime import datetime, timedelta
import stock_hist_em as hist_em
import numpy as np


def posein(pos, line):
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


def main():
    now = datetime.now()
    days_ago = now - timedelta(days=120)
    strdate = now.strftime("%Y%m%d")
    strdate2 = days_ago.strftime("%Y%m%d")

    stock_list = []

    with open("stock_list.txt", "r") as f:
        for line in f.readlines():
            stock_list.append(line.strip())

    cnt = 0
    stdout = ""
    for i in stock_list:
        cnt += 1
        try:
            df = hist_em.stock_zh_a_hist(
                i,
                "daily",
                strdate2,
                strdate,
            )
            df.drop(
                columns=["成交额", "振幅", "涨跌幅", "涨跌额", "换手率"], inplace=True
            )
            df.rename(
                columns={
                    "日期": "Date",
                    "开盘": "Open",
                    "收盘": "Close",
                    "最高": "High",
                    "最低": "Low",
                    "成交量": "Volume",
                    # '成交额': 'Adj Close',  # 如果您想包含调整后的收盘价，取消注释并根据需要进行重命名
                },
                inplace=True,
            )
            close = df["Close"].values
            open_ = df["Open"].values
            line = [close[i] - open_[i] for i in range(len(close))]
            if (
                line[-1] > 0
                and line[-2] <= 0
                and line[-1] + line[-2] > 0
                and line[-3] < line[-2]
            ):
                pos = -2
                pos, neg_sum1, neg_len1 = posein(pos, line)
                neg_min1 = min(close[pos + 1 : pos + neg_len1 + 1])
                pos, pos_sum1, pos_len1 = posein(pos, line)
                pos_max1 = max(close[pos + 1 : pos + pos_len1 + 1])
                pos, neg_sum2, neg_len2 = posein(pos, line)
                neg_min2 = min(close[pos + 1 : pos + neg_len2 + 1])
                pos, pos_sum2, pos_len2 = posein(pos, line)
                pos_max2 = max(close[pos + 1 : pos + pos_len2 + 1])
                pos, neg_sum3, neg_len3 = posein(pos, line)
                neg_max3 = max(open_[pos + 1 : pos + neg_len3 + 1])
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
                    stdout += "open " + i
        except:
            pass
        print("已处理 " + str(cnt) + ", 总共 " + str(len(stock_list)))

    print(stdout)
    input("按任意键继续...")


if __name__ == "__main__":
    main()
