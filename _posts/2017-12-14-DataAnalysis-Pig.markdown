---
layout: post
title:  "数据分析工具之四:Pig常用库"
date:   2017-12-14 21:12:06 +0800
categories: [Tools]
---

* TOC
{:toc}

[TOC]
## 前言
Pig是一门脚本语言，易于入门，方便灵活，容易忘记。在进行模型样本准备时，比sql高效很多。

**持续更新**

## 文件交互
- load
```shell
xinwen_view = load '$INPUT_PATH' as (count:int, count_pv:int, count_kj:int)
```

- store
```shell
store b into '$OUTPUT_PATH';
```

## 并行计算
- PARALLEL

## 常用算子
- group
- flatten
- foreach
- rmf



## 其他常用点
- 自定义函数


## 总结
"工欲善其事，必先利其器"。

## 参考文献
```shell
```
