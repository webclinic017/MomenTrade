from datetime import datetime, timedelta
import stock_hist_em as hist_em
import numpy as np
import chan
import prepare


def main():
    now = datetime(2024, 2, 29)
    days_ago = now - timedelta(days=360)
    tomorrow = now + timedelta(days=1)
    strdate = tomorrow.strftime("%Y%m%d")
    strdate2 = days_ago.strftime("%Y%m%d")

    stock_list = [i for i in hist_em.code_id_map_em().keys()]
    log_map = {}

    with open("log.txt", "a") as f:
        pass
    with open("log.txt", "r") as f:
        log = f.readlines()
        for code in log:
            code_list = code.strip().split()
            code = code_list[0]
            state = code_list[1]
            log_map[code] = state

    with open("log.txt", "w") as f:
        pass

    cnt = 0
    stdout = ""
    for i in stock_list:
        cnt += 1
        state = "close"
        if i in log_map:
            state = log_map[i]
        try:
            prepare.main(i, strdate2, strdate)
            new_state = chan.main2(i, strdate2, strdate)
            if new_state != state:
                stdout += new_state + " " + i + "\n"
            if new_state != "close":
                with open("log.txt", "a") as f:
                    f.write(" ".join([i, new_state]) + "\n")
        except:
            pass
        print("已处理 " + str(cnt) + ", 总共 " + str(len(stock_list)))

    print(stdout)
    input("按任意键继续...")


if __name__ == "__main__":
    main()
