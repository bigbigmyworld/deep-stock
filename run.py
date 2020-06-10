import os
import pickle
import pandas as pd
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from rlenv.StockTradingEnv0 import StockTradingEnv
from rlenv.FundTradingEnv0 import FundTradingEnv

from sqlalchemy import create_engine

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font = fm.FontProperties(fname='font/wqy-microhei.ttc')
# plt.rc('font', family='Source Han Sans CN')
plt.rcParams['axes.unicode_minus'] = False
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def stock_trade(stock_file):
    day_profits = []
    df = pd.read_csv(stock_file)
    df = df.sort_values('date')

    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([lambda: StockTradingEnv(df)])

    model = PPO2(MlpPolicy, env, verbose=0, tensorboard_log='./log')
    model.learn(total_timesteps=int(1e4))

    df_test = pd.read_csv(stock_file.replace('train', 'test'))

    env = DummyVecEnv([lambda: StockTradingEnv(df_test)])
    obs = env.reset()
    for i in range(len(df_test) - 1):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        profit = env.render()
        day_profits.append(profit)
        if done:
            break
    return day_profits


def fund_trade(ts_code):
    day_profits = []
    df = read_fund_nav(ts_code, 3000).head(-30)
    df = df.sort_values(by='end_date', ascending=True).reset_index(drop=True)
    # todo 最后一行包含了估值，导致报错，临时删除
    df.drop([len(df) - 1], inplace=True)

    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([lambda: FundTradingEnv(df)])
    model = PPO2(MlpPolicy, env, verbose=0, tensorboard_log='./log',seed=1)
    model.learn(total_timesteps=int(1e4))
    print("开始测试")
    df_test = df.tail(30).reset_index(drop=True)

    env = DummyVecEnv([lambda: FundTradingEnv(df_test)])
    obs = env.reset()
    for i in range(len(df_test) - 1):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        profit = env.render()
        day_profits.append(profit)
        if done:
            break
    return day_profits


def find_file(path, name):
    # print(path, name)
    for root, dirs, files in os.walk(path):
        for fname in files:
            if name in fname:
                return os.path.join(root, fname)


def test_a_fund_trade(stock_code):
    daily_profits = fund_trade(stock_code)
    fig, ax = plt.subplots()
    ax.plot(daily_profits, '-o', label=stock_code, marker='o', ms=10, alpha=0.7, mfc='orange')
    ax.grid()
    plt.xlabel('step')
    plt.ylabel('profit')
    ax.legend(prop=font)
    # plt.show()
    plt.savefig(f'./img/{stock_code}.png')


def test_a_stock_trade(stock_code):
    stock_file = find_file('./stockdata/train', str(stock_code))
    print(stock_file)
    daily_profits = stock_trade(stock_file)
    fig, ax = plt.subplots()
    ax.plot(daily_profits, '-o', label=stock_code, marker='o', ms=10, alpha=0.7, mfc='orange')
    ax.grid()
    plt.xlabel('step')
    plt.ylabel('profit')
    ax.legend(prop=font)
    # plt.show()
    plt.savefig(f'./img/{stock_code}.png')


def multi_stock_trade():
    start_code = 600000
    max_num = 3000

    group_result = []

    for code in range(start_code, start_code + max_num):
        stock_file = find_file('./stockdata/train', str(code))
        if stock_file:
            try:
                profits = stock_trade(stock_file)
                group_result.append(profits)
            except Exception as err:
                print(err)

    with open(f'code-{start_code}-{start_code + max_num}.pkl', 'wb') as f:
        pickle.dump(group_result, f)


database_string = "mysql+pymysql://root:@localhost:3306/funds"


def read_fund_nav(ts_code, head_num=90):
    engine = create_engine(database_string)
    table_name = 'FUND_NAV_' + ts_code[4:]
    print(table_name)
    try:
        # MySQL导入DataFrame
        sql_query = "select * from {table_name} where  ts_code like '{ts_code}%%' order by ann_date desc limit {limit};".format(
            table_name=table_name, ts_code=ts_code, limit=head_num)
        # print(sql_query)
        # 使用pandas的read_sql_query函数执行SQL语句，并存入DataFrame
        df_read = None
        if head_num == "all":
            df_read = pd.read_sql_query(sql_query, engine)
            diff = df_read.loc[:, "unit_nav"].diff(-1).shift(0).round(4)
            # 收益率百分比
            diff_pct = df_read.loc[:, "unit_nav"].pct_change(periods=-1).shift(0).round(4)
            df_read["diff"] = diff
            df_read["diff_pct"] = diff_pct
            # 以国债为参考，假设年化收益率为3.9%，每年252个交易日
            df_read['excess_daily_ret'] = df_read["diff_pct"] - 0.04 / 252
            df_read["diff_pct"] = df_read.apply(lambda x: cal_pct(x), axis=1).round(2)
        else:
            df_read = pd.read_sql_query(sql_query, engine)[:int(head_num)]
            df_predict = read_predict_nav(ts_code)
            df_read = pd.concat([df_read, df_predict], ignore_index=True)
            df_read = df_read.sort_values(by="end_date", ascending=False)
            diff = df_read.loc[:, "unit_nav"].diff(-1).shift(0).round(4)
            # 计算变化百分比
            diff_pct = df_read.loc[:, "unit_nav"].pct_change(periods=-1).shift(0).round(4)
            df_read["diff"] = diff  # shift(0)数据整体向上移动一行，第一行消失
            df_read["diff_pct"] = diff_pct
            df_read["diff_pct"] = df_read.apply(lambda x: cal_pct(x), axis=1).round(2)
        # diff_reduce = df_read[df_read["diff"] < 0].describe()
        # print(df_read)
        df_read = df_read.reset_index(drop=True)
        return df_read
    except Exception as e:
        print(e)
        return df_read


def read_predict_nav(ts_code):
    engine = create_engine(database_string)
    table_name = 'FUND_PREDICT_NAV'
    # MySQL导入DataFrame
    sql_query = "select * from {table_name} where  ts_code like '{ts_code}%%';".format(
        table_name=table_name, ts_code=ts_code)
    df_read = pd.read_sql_query(sql_query, engine)
    print(df_read)
    return df_read


def cal_pct(x):
    y = x.diff_pct * 100
    return y


if __name__ == '__main__':
    # multi_stock_trade()
    # test_a_stock_trade('sh.600036')
    # test_a_stock_trade('sz.002241')

    # ret = find_file('./stockdata/train', '600036')
    # print(ret)
    test_a_fund_trade("519005")
