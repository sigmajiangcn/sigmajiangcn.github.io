---
layout: post
title:  "2_AddTwoNums"
date:   2018-04-13 23:22:28 +0800
categories: [lettcode]
---

* TOC
{:toc}

[TOC]

## 问题
[两个非负非空单链表表示的逆序整数之和](https://leetcode.com/problems/add-two-numbers/description//)

You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

Examples：
```python
Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 0 -> 8
Explanation: 342 + 465 = 807.
```


## 思路分析
两个逆序整数链表求和，逆序链表的head是整数的低位，而两个整数相加本身就应该从低位相加。需要注意的是最后有可能有进位1.并且在初始化链表的表头以及最后返回时需要去掉表头。
从lettcode的耗时看，当前解法python耗时很高。后面需要继续优化。

## 实际代码
Python
```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        if(not l1):
            return l2
        if(not l2):
            return l1
        res=ListNode(0)
        head=res
        carry=0
        while(l1 or l2):
            sum=(l1.val if l1 else 0) +(l2.val if l2 else 0)+carry
            node=ListNode(sum%10)
            carry=sum/10
            res.next=node
            res=res.next
            if l1:
                l1=l1.next
            if l2:
                l2=l2.next
        if carry:
            res.next=ListNode(1)

        head=head.next
        return head
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
1.如果是两个逆序整数相乘法？
## 总结
表面是逆序整数相加的问题，实际上就是正常的两个数从低位相加的过程。
