---
layout: post
title:  "Factorization Machines"
date:   2018-03-01 10:31:28 +0800
categories: [adRecommend]
---

* TOC
{:toc}



## 业务简介
特征交叉工作，通过人工发现成本较高，可以通过根据信息论GBDT或者矩阵分解FM两种方法来得到

实际上就FM因子分解而言，依赖于如下两个朴素的公式：
特征是相互交叉的二阶特征，
- 等式变换
$$ab+bc+ab=\dfrac{1}{2}[(a+b+c)^2-(a^2+b^2+c^2)]$$
- 矩阵分解
$$<x_i,x_j>,Cn2$$实对称矩阵

## 主要内容
### FMs
### wide&deep
### deepFM
### xdeepFM
### DeepCross
### DSSM
### NFM
### AFM
### DIN


## 致谢


## 参考文献
[1、[《图灵系统文集》](http://km.oa.com/knowledge/2074)
