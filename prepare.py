import stock_hist_em as hist_em
import future_crawler as fc


def main(num, date1, date2):
    df = hist_em.stock_zh_a_hist(
        num,
        "daily",
        date1,
        date2,
    )
    df.drop(columns=["成交额", "振幅", "涨跌幅", "涨跌额", "换手率"], inplace=True)
    # 重命名列以匹配Backtrader的要求
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

    # 将DataFrame保存为CSV文件
    df.to_csv("stock_data.csv", index=False)


def main_future(num, date1, date2):
    df = fc.future_zh_a_hist(
        num,
        "daily",
        date1,
        date2,
    )
    df.drop(columns=["成交额", "振幅", "涨跌幅", "涨跌额", "换手率"], inplace=True)
    # 重命名列以匹配Backtrader的要求
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

    # 将DataFrame保存为CSV文件
    df.to_csv("stock_data.csv", index=False)


if __name__ == "__main__":
    main("301558", "20230830", "20240830")
