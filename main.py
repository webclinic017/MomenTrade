import MomenTrade
import prepare
import pymysql
import csv

# import Bollinger_Band

config = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "passwd": "123456",
    "db": "instockdb",
    "charset": "utf8mb4",
}


# 输出到 CSV 文件
def write_to_csv(data, filename):
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Stock", "Result"])  # 写入表头
        writer.writerows(data)  # 写入数据


# 目标字段列表
stock_list = []

try:
    # 创建连接
    connection = pymysql.connect(**config)

    # 创建游标
    with connection.cursor() as cursor:
        # 构建SQL查询语句，使用DISTINCT去除重复行，选取感兴趣的字段
        select_query = f"SELECT DISTINCT code FROM cn_stock_spot;"

        # 执行查询
        cursor.execute(select_query)
        # 获取所有数据
        results = cursor.fetchall()

        for row in results:
            stock_list.append(row)

finally:
    # 关闭连接
    if connection:
        connection.close()
consequence = []
for i in stock_list:
    i = i[0]
    try:
        prepare.main(i, "20220619", "20230828")
        j = MomenTrade.main("20220619", "20230828", i)
        # j = Bollinger_Band.main("20230619", "20240828", i)
        if abs(j) > 0.1:
            consequence.append((i, j))
    except:
        pass
write_to_csv(consequence, "consequence.csv")
