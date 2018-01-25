---
layout: post
title:  "数据分析工具之五:Spark"
date:   2017-12-15 21:12:06 +0800
categories: [Tools]
---

* TOC
{:toc}

[TOC]
## 前言
Spark是一个高效内存迭代计算的模型训练框架。

**持续更新**

## 文件交互


## 并行计算
- PARALLEL

## 常用算子
- Transformation
Transformation 操作是延迟计算的，也就是说从一个RDD 转换生成另一个 RDD 的转换操作不是马上执行，需要等到有 Action 操作的时候才会真正触发运算。

- Action
Action 算子会触发 Spark 提交作业（Job），并将数据输出 Spark系统。

- map
- flatMap
- union
- groupBy
- filter
- join
- collect
- count
- reduce
- aggregate

## 其他常用点
- 自定义函数


## 总结
"工欲善其事，必先利其器"。

## 参考文献
```shell
```
