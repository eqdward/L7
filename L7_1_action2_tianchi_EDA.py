# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 14:16:30 2020

@author: yy
"""

import pandas as pd

"""0. 加载数据"""
df = pd.read_csv(r'C:\Users\Admin\Desktop\L7\fresh_comp_offline\tianchi_fresh_comp_train_user.csv')

"""1. 数据预处理——采样分析"""
# 数据集过大，为方便分析，从中抽取部分数据

# 用户数为20000，远小于数据样本数，说明用户的每一次操作为一条记录，因此不能直接在数据集进行采样，否则会破坏用户的完整行为
users = df['user_id'].unique()
users = pd.DataFrame(users, columns=['user_id'])

# 使用无放回抽样，抽取10%的用户
sample_users = users.sample(frac=0.1, replace=False, random_state=666)

# 从原始数据集中筛选出样本用户的数据
data = pd.DataFrame(data=[], columns = df.columns)
i=1
for user in sample_users['user_id']:
    print(i)
    temp = df[df['user_id'] == user]
    data = data.append(temp)
    i += 1

#data.to_csv(r'C:\Users\Admin\Desktop\L7\fresh_comp_offline\sample.csv')

"""2. EDA"""
print(data.shape)   # 抽样数据集结构
print("%.2f%%" % (data.shape[0]/df.shape[0]*100))   #抽样数据集占原数据集比例
print(data.isnull().sum())   # 检查是否确实值

"""2.1 计算CVR"""
# CVR = 购买行为数 / 用户行为总数
behavior_types = {1:'浏览', 2:'收藏', 3:'加购', 4:'购买'}
behavior_count = data['behavior_type'].value_counts()
cvr = behavior_count[4] / len(data)
print("CVR={:.2f}%".format(cvr*100))

"""2.2 时间维度分析"""
data['time'] = pd.to_datetime(data['time'])   # 数据类型转化
data.set_index(['time'], drop=True, inplace=True)   # 时间数据设为index

from collections import defaultdict
from datetime import datetime, timedelta

day_count = defaultdict(int)
start_date = '2014-11-17'
temp_date = datetime.strptime(start_date, '%Y-%m-%d')
delta = timedelta(days=1)
for i in range(31):
    temp_date = temp_date + delta
    temp_str = temp_date.strftime('%Y-%m-%d')
    day_count[temp_str] += data[temp_str].shape[0]

df_day_count = pd.DataFrame.from_dict(data=day_count, orient='index', columns=['count'])

import matplotlib.pyplot as plt
plt.figure(figsize=(10,8))
df_day_count['count'].plot(kind='bar')
plt.legend(loc='best')
plt.show()


"""3.0 商品的时间维度分析"""
# 按日期分析
data_p = pd.read_csv(r'C:\Users\Admin\Desktop\L7\fresh_comp_offline\tianchi_fresh_comp_train_item.csv')
data = pd.merge(data.reset_index(drop=False), data_p, on=['item_id']).set_index('time')

data['date'] = data.index
data['date'] = data['date'].map(lambda x: x.strftime('%Y-%m-%d'))
data['date'] = data['date'].map(lambda x:x[:10])
df_day_count2 = data['date'].value_counts()
df_day_count2.sort_index(inplace=True)

plt.figure(figsize=(10,8))
df_day_count2.plot(kind='bar')
plt.legend(loc='best')
plt.show()

"""4.0 四种行为分小时比较"""
# 所有行为按小时分析
data['hour'] = data.index.hour
hour_count = data.groupby('hour')['hour'].count()

plt.figure(figsize=(10,8))
hour_count.plot(kind='bar')
plt.legend(loc='best')
plt.show()

# 四种行为比较
behavior_count = pd.pivot_table(data, values='user_id', index='hour', columns='behavior_type', aggfunc='count')

plt.figure(figsize=(10,8))
behavior_count.plot(kind='bar')
plt.legend(loc='best')
plt.show()
