---
layout:     post
title:      Pandas 数据分析实战
subtitle:   让你像写SQL一样做数据分析
date:       2017-07-31
author:     Stephen
header-img: img/post-bg-digital-native.jpg
catalog: true
tags:
    - Pandas
    - SQL
    - Python
    - 数据分析
---

# Pandas: 让你像写SQL一样做数据分析

## 1.引言

假设现在有一份简化版的设备统计数据：

|维度编号| 操作系统 |  设备型号  |    UV    |    PV     |
| ------ | -------- |  -------:  | :-----:  | :-------: |
|    0   |  android |    NLL     | 387546520| 2099457911|
|    0   |   ios    |    NLL     | 52877990 | 916421755 |
|    1   |  android |    魅族    | 8995958  | 120369597 |
|    1   |  android |    酷派    | 9915906  | 200818172 |
|    1   |  android |    三星    | 16500493 | 718969514 |
|    1   |  android |    小米    | 23933856 | 290787590 |
|    1   |  android |    华为    | 26706736 | 641907761 |
|    1   |   ios    |    苹果    | 52877990 | 916421755 |
|    2   |  android | 小米-小米4 | 2786675  | 55376581  |
|    2   |  android |魅族-m2-note| 4642112  | 130984205 |
|    2   |  android |  OPPO-A31  | 4893428  | 62976997  |
|    2   |   ios    | iPhone-6s  | 5728609  | 99948716  |

其中，第一列表示维度组合编号，第二列表示操作系统类型，第三列为维度值（NLL表示缺失，即第一行、第二行表示操作系统的统计，其余表示厂商或机型），第三列、第四列分别表示UV、PV；且字段之间为\t分隔。读取该文件为DataFrame：

```python
import pandas as pd

df = pd.read_csv(path, names=['id', 'os', 'dim', 'uv', 'pv'], sep='\t')
```

## 2.实战

> ADD

在原DataFrame上，增加一行数据；可通过DataFrame的append函数来追加：

```python
import numpy as np
row_df = pd.DataFrame(np.array([['2', 'ios', '苹果-iPad 4', 3287509, 32891811]]), columns=['id', 'os', 'dim', 'uv', 'pv'])
df = df.append(row_df, ignore_index=True)
```

增加一列数据，则比较简单：

```python
df['time'] = '2017-07-29'
```

> To Dict

关于android、ios的PV、UV的dict：

```python
def where(df, col_name, id_value):
    df = df[df[col_name] == id_value]
    return df
    
def to_dict(df):
"""
    {"pv" or "uv" -> {"os": os_value}}
    :return: dict
    """
    df = where(df, 'id', 0)
    df_dict = df.set_index('os')[['uv', 'pv']].to_dict()
    return df_dict
```

> Top

group某列后的top值，比如，android、ios的UV top 2的厂商：

```python
def group_top(df, group_col, sort_col, top_n):
    """
    get top(`sort_col`) after group by `group_col`
    :param df: dataframe
    :param group_col: string, column name
    :param sort_col: string, column name
    :param top_n: int
    :return: dataframe
    """
    return df.assign(rn=df.sort_values([sort_col], ascending=False)
                     .groupby(group_col)
                     .cumcount() + 1) \
        .query('rn < ' + str(top_n + 1)) \
        .sort_values([group_col, 'rn'])
```

全局top值加上group某列后的top值，并有去重：

```python
def top(df, group_col, sort_col, top_n):
    """overall top and group top"""
    all_top_df = df.nlargest(top_n, columns=sort_col)
    grouped_top_df = group_top(df, group_col, sort_col, top_n)
    grouped_top_df = grouped_top_df.ix[:, 0:-1]
    result_df = pd.concat([all_top_df, grouped_top_df]).drop_duplicates()
    return result_df
```

> 排序编号

对某列排序后并编号，相当于给出排序名次。比如，对UV的排序编号：

```python
df['rank'] = df['uv'].rank(method='first', ascending=False).apply(lambda x: int(x))
```

> Left Join

Pandas的left join对NULL的列没有指定默认值，下面给出简单的实现：

```python
def left_join(left, right, on, right_col, default_value):
    df = pd.merge(left, right, how='left', on=on)
    df[right_col] = df[right_col].map(lambda x: default_value if pd.isnull(x) else x)
    return df
```

> 自定义

对某一列做较为复杂的自定义操作，比如，厂商的UV占比：

```python
def percentage(part, whole):
    return round(100*float(part)/float(whole), 2)


os_dict = to_dict(df)
all_uv = sum(os_dict['uv'].values())
df = where(df, 'id', 1)
df['per'] = df.apply(lambda r: percentage(r['uv'], all_uv), axis=1)
```

> 重复值

某列的重复值的行：

```python
duplicate = df.duplicated(subset=columns, keep=False)
```

> 写MySQL

Pandas的to_sql函数支持Dataframe直接写MySQL数据库。在公司开发时，常常会有办公网与研发网是不通的，Python的sshtunnel模块提供ssh通道，便于入库debug。

```python
import MySQLdb
from sshtunnel import SSHTunnelForwarder


with SSHTunnelForwarder(('porxy host', port),
                        ssh_password='os passwd',
                        ssh_username='os user name',
                        remote_bind_address=('mysql host', 3306)) as server:
    conn = MySQLdb.connect(host="127.0.0.1", user="mysql user name", passwd="mysql passwd",
                           db="db name", port=server.local_bind_port, charset='utf8')
    df.to_sql(name='tb name', con=conn, flavor='mysql', if_exists='append', index=False)
```

> 转载自：http://www.cnblogs.com/en-heng/
