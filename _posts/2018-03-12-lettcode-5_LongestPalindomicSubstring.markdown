---
layout: post
title:  "5_Longest Palindomic Substring"
date:   2018-04-13 23:22:28 +0800
categories: [lettcode]
---

* TOC
{:toc}

[TOC]

## 问题
[最长回文子序列](Longest Palindromic Substring)

Given a string s, find the longest palindromic substring in s. You may assume that the maximum length of s is 1000.


Examples：
```python
Input: "babad"
Output: "bab"
Note: "aba" is also a valid answer.
```

```python
Input: "cbbd"
Output: "bb"
```
## 思路分析
参考这个[实现方案](https://blog.csdn.net/fuxuemingzhu/article/details/79573621)，采用动态规划的方法解决问题。

动态规划的两个特点：第一大问题拆解为小问题，第二重复利用之前的计算结果。

参考资料：

- [解题报告](http://fisherlei.blogspot.com/2012/12/leetcode-longest-palindromic-substring.html)
指出存在$O(n)$[解法](https://articles.leetcode.com/longest-palindromic-substring-part-ii/)。

- 需要注意下[Python中定义二维数组的方法](https://www.cnblogs.com/woshare/p/5823303.html)

## 实际代码
Python
```python
class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        if len(set(s)) == 1:
            return s
        n = len(s)
        start, end, maxL = 0, 0, 0
        dp = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i):
                dp[j][i] = (s[j] == s[i]) & ((i - j < 2) | dp[j + 1][i - 1])
                if dp[j][i] and maxL < i - j + 1:
                    maxL = i - j + 1
                    start = j
                    end = i
            dp[i][i] = 1
        return s[start : end + 1]
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
