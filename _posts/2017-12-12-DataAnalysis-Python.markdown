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

2018年3月1日，微信公众号《机器学习研究会》推送的一篇《Pandas习题集》，其github链接提供了大量的**微操**（*Tiny but effective and strong operation*）,原文：
Since pandas is a large library with many different specialist features and functions, these excercises focus mainly on the fundamentals of manipulating data (indexing, grouping, aggregating, cleaning), making use of the core DataFrame and Series objects. [Many of the excerises](https://github.com/ajcr/100-pandas-puzzles) here are straightforward in that the solutions require no more than a few lines of code (in pandas or NumPy - don't go using pure Python!). Choosing the right methods and following best practices is the underlying goal.

另外，还有[《Numpy习题集》](https://github.com/rougier/numpy-100)。


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
- 坐标轴点数选取
横坐标如果太密集，可以[选取部分值出来](https://mail.python.org/pipermail/python-list/2006-March/389991.html)：
```Python
from matplotlib.ticker import MaxNLocator
from pylab import figure, show, nx

fig = figure()
ax = fig.add_subplot(111)
ax.plot(nx.mlab.rand(1000))
ax.xaxis.set_major_locator(MaxNLocator(4))
show()
```
- 坐标轴时间轴格式
可以调整时间轴的[格式](https://www.tuicool.com/articles/jmQzUzy)和旋转角度：
```Python
 ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))#设置时间标签显示格式
```
[mdate](https://matplotlib.org/devdocs/gallery/api/date.html)




## 其他常用点
- glob
- os.makedirs
- os.sys.platform.startswith("win")

## 总结
"工欲善其事，必先利其器"。

## 参考文献
```python
```
- 《Pandas习题集》 https://github.com/ajcr/100-pandas-puzzles
