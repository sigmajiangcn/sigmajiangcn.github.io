---
layout: post
title:  "数据分析工具之二:Python常用库"
date:   2017-12-12 21:12:06 +0800
categories: [Tools]
---

* TOC
{:toc}

[TOC]
## 前言
Python是一门常用的编程语言，易于入门，方便灵活。尤其当掌握了一些常用的库，更是用的飞起。这里主要介绍介绍Pandas。

**持续更新**

## 文件交互
- read_csv
```python
df=pd.read_csv(train_file,sep=",",index_col=target_list,nrows=5)
```
- to_csv

## 常用算子
- query
- groupby
- merge
- date_range
- concat
- reset_index
- sort_values
```python
algo_filter_df=compare_df.groupby(target_list)[['algorithm','result']].apply(lambda x:pd.DataFrame.sort_values(x,by='result')[['algorithm']].head(2)).reset_index()
```
- lambda
```python
df.apply(lambda x:True if x.sum()>10 else 'false')
df['mean']=df.mean(axis=1)
df.query("mean>10").drop(['mean'],axis=1)
```

- fillna
- drop del
drop 删除列返回新的df，不影响原有数据，更安全

- loc、iloc、ix
```python
df.head(2).iloc[:,0:1]
```
## matplotlib绘图
- figure
- hist
- xlabel
- ylabel
- legend
- show

## 其他常用点
- glob
- os.makedirs
- os.sys.platform.startswith("win")

## 总结
"工欲善其事，必先利其器"。

## 参考文献
```python
```
