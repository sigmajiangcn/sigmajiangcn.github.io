---
layout: post
title:  "6_ZigZagConversion"
date:   2018-04-13 23:22:28 +0800
categories: [lettcode]
---

* TOC
{:toc}

[TOC]

## 问题
[之字形路线](https://leetcode.com/problems/zigzag-conversion/description/)

The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this: (you may want to display this pattern in a fixed font for better legibility)
```python
P   A   H   N
A P L S I I G
Y   I   R
```
And then read line by line: "PAHNAPLSIIGYIR"

Write the code that will take a string and make this conversion given a number of rows:
Examples：
```python
Input: s = "PAYPALISHIRING", numRows = 3
Output: "PAHNAPLSIIGYIR"
```

```python
Input: s = "PAYPALISHIRING", numRows = 4
Output: "PINALSIGYAHRPI"
Explanation:

P     I    N
A   L S  I G
Y A   H R
P     I
```
## 思路分析
参考这个[实现方案](https://blog.csdn.net/qian2729/article/details/50507694)，采用模拟之字路线图的方法解决问题。

用一个数组来存储每一行的字符，用变量row来记录当前访问到的行，依次将字符串text中的字符放入不同的行，然后将所有行的字符串串联起来。
## 实际代码
Python
```python
class Solution(object):
    def convert(self, s, numRows):
        """
        :type s: str
        :type numRows: int
        :rtype: str
        """
        if numRows == 1:
            return s
        zigzag = ['' for _ in range(numRows)]
        row = 0
        step = 1
        for c in s:
            if row == 0:
                step = 1
            if row == numRows - 1:
                step = -1
            zigzag[row] += c
            row += step
        return ''.join(zigzag)
```

C++
```code

```

JAVA
```code

```

GO
```code

```


SCALA
```code

```
## 思路拓展

## 总结
