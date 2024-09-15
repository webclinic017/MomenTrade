import pandas as pd
import requests
import json


def future_zh_a_hist(
    symbol: str = "NI2508",
    period: str = "daily",
    start_date: str = "19700101",
    end_date: str = "20500101",
    adjust: str = "",
) -> pd.DataFrame:
    """
    东方财富网-行情首页-沪深京 A 股-每日行情
    https://quote.eastmoney.com/concept/sh603777.html?from=classic
    :param symbol: 股票代码
    :type symbol: str
    :param period: choice of {'daily', 'weekly', 'monthly'}
    :type period: str
    :param start_date: 开始日期
    :type start_date: str
    :param end_date: 结束日期
    :type end_date: str
    :param adjust: choice of {"qfq": "前复权", "hfq": "后复权", "": "不复权"}
    :type adjust: str
    :return: 每日行情
    :rtype: pandas.DataFrame
    """
    adjust_dict = {"qfq": "1", "hfq": "2", "": "0"}
    period_dict = {"daily": "101", "weekly": "102", "monthly": "103"}
    url = "http://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "klt": period_dict[period],
        "fqt": adjust_dict[adjust],
        "secid": f"113.{symbol}",
        "beg": start_date,
        "end": end_date,
        "_": "1623766962675",
    }
    r = requests.get(url, params=params)
    data_json = r.json()
    if not (data_json["data"] and data_json["data"]["klines"]):
        return pd.DataFrame()
    temp_df = pd.DataFrame([item.split(",") for item in data_json["data"]["klines"]])
    temp_df.columns = [
        "日期",
        "开盘",
        "收盘",
        "最高",
        "最低",
        "成交量",
        "成交额",
        "振幅",
        "涨跌幅",
        "涨跌额",
        "换手率",
    ]
    temp_df.index = pd.to_datetime(temp_df["日期"])
    temp_df.reset_index(inplace=True, drop=True)

    temp_df["开盘"] = pd.to_numeric(temp_df["开盘"])
    temp_df["收盘"] = pd.to_numeric(temp_df["收盘"])
    temp_df["最高"] = pd.to_numeric(temp_df["最高"])
    temp_df["最低"] = pd.to_numeric(temp_df["最低"])
    temp_df["成交量"] = pd.to_numeric(temp_df["成交量"])
    temp_df["成交额"] = pd.to_numeric(temp_df["成交额"])
    temp_df["振幅"] = pd.to_numeric(temp_df["振幅"])
    temp_df["涨跌幅"] = pd.to_numeric(temp_df["涨跌幅"])
    temp_df["涨跌额"] = pd.to_numeric(temp_df["涨跌额"])
    temp_df["换手率"] = pd.to_numeric(temp_df["换手率"])

    return temp_df


def return_future_list() -> list:
    # 设置请求的URL
    url = "https://futsseapi.eastmoney.com/list/113?callback=aaa_callback&orderBy=zdf&sort=desc&pageSize=1000&pageIndex=0&callbackName=aaa_callback&token=58b2fa8f54638b60b87d69b31969089c&field=dm%2Csc%2Cname%2Cp%2Czsjd%2Czde%2Czdf%2Cf152%2Co%2Ch%2Cl%2Czjsj%2Cvol%2Ccje%2Cwp%2Cnp%2Cccl&blockName=callback&_=1726302590044"

    # 发送HTTP请求
    response = requests.get(url)

    # 检查请求是否成功
    if response.status_code == 200:
        # 由于返回的是一个JSONP格式的数据，需要去除回调函数的包装
        callback_function = (
            "aaa_callback" + "(\u200b"
        )  # \u200b是零宽度空格，用于匹配JSONP的格式
        json_str = response.text[
            len(callback_function) : -2
        ]  # 去除JSONP的函数名和最后的括号

        json_str = "{" + json_str + "}"

        # 解析JSON数据
        data = json.loads(json_str)
        df = pd.DataFrame(data["list"])

        # # 重命名列名
        # df.rename(
        #     columns={
        #         "np": "卖盘",
        #         "h": "最高价",
        #         "dm": "代码",
        #         "zsjd": "",
        #         "l": "最低价",
        #         "ccl": "持仓量",
        #         "o": "开盘价",
        #         "p": "最新价",
        #         "sc": "",
        #         "vol": "成交量",
        #         "name": "名称",
        #         "wp": "买盘",
        #         "zde": "涨跌额",
        #         "zdf": "涨跌幅",
        #         "zjsj": "昨结算价",
        #         "cje": "成交额",
        #     },
        #     inplace=True,
        # )
        # df.drop(columns=[""], inplace=True)
        return df["dm"]
    else:
        print("请求失败，状态码:", response.status_code)
        return None
