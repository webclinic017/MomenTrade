import MomenTrade
import prepare
import csv
import os
import future_crawler as fc
import Bollinger_Band as bb

# # import Bollinger_Band
# config = {
#     "host": "127.0.0.1",
#     "port": 3306,
#     "user": "root",
#     "passwd": "123456",
#     "db": "instockdb",
#     "charset": "utf8mb4",
# }


# 输出到 CSV 文件
def write_to_csv(data, filename):
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Stock", "Result"])  # 写入表头
        writer.writerows(data)  # 写入数据


# 目标字段列表
stock_list = [i.upper() for i in fc.return_future_list()]

# try:
#     # 创建连接
#     connection = pymysql.connect(**config)

#     # 创建游标
#     with connection.cursor() as cursor:
#         # 构建SQL查询语句，使用DISTINCT去除重复行，选取感兴趣的字段
#         select_query = f"SELECT DISTINCT code FROM cn_stock_spot;"

#         # 执行查询
#         cursor.execute(select_query)
#         # 获取所有数据
#         results = cursor.fetchall()

#         for row in results:
#             stock_list.append(row)

# finally:
#     # 关闭连接
#     if connection:
#         connection.close()

# with open("stock_list.txt", "w") as f:
#     for stock in stock_list:
#         f.write(stock[0] + "\n")
if not os.path.exists("future"):
    os.makedirs("future")

os.chdir("future")

consequence = []

year = 2020

folder_path = str(year)

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

os.chdir(folder_path)

if not os.path.exists("photo"):
    os.makedirs("photo")
date1 = str(year) + "0830"
date2 = str(year + 30) + "0830"
for i in stock_list:
    try:
        prepare.main_future(i, date1, date2)
        j = MomenTrade.main(date1, date2, i)
        # j = Bollinger_Band.main("20230619", "20240828", i)
        if abs(j) > 0.1:
            consequence.append((i, j))
    except:
        pass
write_to_csv(consequence, "consequence.csv")
