---
layout: post
title: "3_LongestSubStringWithoutRepeatingCharacter"
date:   2018-04-13 23:22:28 +0800
categories: [lettcode]
---

* TOC
{:toc}

[TOC]

## 问题
[最长无重复字符的子串](https://leetcode.com/problems/longest-substring-without-repeating-characters/description/)

Given a string, find the length of the longest substring without repeating characters.

Examples：
```python
Given "abcabcbb", the answer is "abc", which the length is 3.

Given "bbbbb", the answer is "b", with the length of 1.

Given "pwwkew", the answer is "wke", with the length of 3. Note that the answer must be a substring, "pwke" is a subsequence and not a substring.
```


## 思路分析
参考使用[enumerate](https://leetcode.com/problems/longest-substring-without-repeating-characters/discuss/1731/A-Python-solution-85ms-O(n))迭代字符串，一次遍历字符串，如果字符已经存在，则更新起点位置；否则更新当前最大长度。

## 实际代码
Python
```python
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        start=0
        exist={}
        maxlen=0
        for i,c in enumerate(s):
            if(c in exist and exist[c]>=start):
                start=exist[c]+1
            else:
                maxlen=max(maxlen,i-start+1)
            exist[c]=i
        return maxlen
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
1.如果容许一个字符或者若干字符重复，则结果如何？
## 总结
一次遍历，充分利用了字典的查找效率。
