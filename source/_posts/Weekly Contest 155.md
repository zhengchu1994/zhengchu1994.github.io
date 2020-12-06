---
title: LeetCode-Weekly Contest 155
mathjax: true
date: 2019-09-20 21:53:33
tags: [Leetcode, Algorithm]
categories: 算法与数据结构
visible:
---

## Weekly Contest 155

#### 1200. Minimum Absolute Difference

> Given an array of **distinct** integers `arr`, find all pairs of elements with the minimum absolute difference of any two elements. 
>
> Return a list of pairs in ascending order(with respect to pairs), each pair `[a, b]` follows
>
> - `a, b` are from `arr`
> - `a < b`
> - `b - a` equals to the minimum absolute difference of any two elements in `arr`
>
> ```pseudocode
> Input: arr = [4,2,1,3]
> Output: [[1,2],[2,3],[3,4]]
> Explanation: The minimum absolute difference is 1. List all pairs with difference equal to 1 in ascending order.
> ```

```python
class Solution:
    def minimumAbsDifference(self, arr: List[int]) -> List[List[int]]:
        arr.sort()
        mini = 1e6
        table = []
        for i in range(1, len(arr)):
            res = arr[i] - arr[i-1]
            mini = min(res, mini)
        for i in range(1,len(arr)):
            res = arr[i] - arr[i-1]
            if res == mini:
                table.append([arr[i - 1], arr[i]])
        return table
        
```





####  1201. Ugly Number III

>Write a program to find the `n`-th ugly number.
>
>Ugly numbers are **positive integers** which are divisible by `a` **or** `b` **or** `c`.
>
>```pseudocode
>Input: n = 3, a = 2, b = 3, c = 5
>Output: 4
>Explanation: The ugly numbers are 2, 3, 4, 5, 6, 8, 9, 10... The 3rd is 4.
>```

* 解析：https://leetcode.com/problems/ugly-number-iii/discuss/388461/c%2B%2B-binary-search-solution-with-explanation

>There are x/a number can be divisible by a, x/b number can be divisible by b and x/c number can be divisible by c.
>However, in such case, if one number can be divisible by both (a,b), we count it twice. We have to subtract those cases from our total count.
>
>Lucky enough, if a number can be divisible by both (a,b), it must can be divisible by lcm(a,b). So we need to calucalte least common multiple between (a,b) (b,c) (a,c) and (a,b,c).
>
>Then we know, there are x/lcm(a,b) number can be divisible by both (a,b), x/lcm(b,c) number can be divisible by both (b,c, and x/lcm(a,c) number can be divisible by both (a,c).
>
>However, after subtracting them, we don't count number which can be divisible by (a,b,c) together, to fix it, we simply need to add x/lcm(a,lcm(b,c)).



* ![](https://assets.leetcode.com/users/hiepit/image_1569139496.png)

```c++
class Solution {
public:
    typedef long long ll;
    //Greatest Common Divisor(gcd)
    ll gcd(ll a, ll b)
    {
        if (b == 0) return a;
        return gcd(b, a % b);
    }
    //Least Common Multiple(lcm)
    ll lcm(ll a, ll b)
    {
        return (a * b) / gcd(a, b);
    }
    // lmc(a, b) * gcd(a, b) =  a * b
    ll numberOfDivision(ll num, ll a, ll b, ll c)
    {
        return  ((num/a) + (num/b) + (num/c) 
                   - (num / lcm(b,  a))
                   - (num / lcm(c, b))
                   - (num / lcm(c, a))
                   + (num / lcm(a, lcm(c, b)))); // lcm(a,b,c) = lcm(a,lcm(b,c))
    }
    
    int nthUglyNumber(int n, int a, int b, int c) {
        ll num;
        ll low = 1, high = INT_MAX;
        while(low < high)
        {
            ll mid = low  + (high - low) / 2;
            if (numberOfDivision(mid, a, b, c) >= n)
                high = mid;
            else
                low = mid + 1;
        }
        return high;
    }
};
```



#### 1202. Smallest String With Swaps

>You are given a string `s`, and an array of pairs of indices in the string `pairs` where `pairs[i] = [a, b]` indicates 2 indices(0-indexed) of the string.
>
>You can swap the characters at any pair of indices in the given `pairs` **any number of times**.
>
>Return the lexicographically smallest string that `s` can be changed to after using the swaps
>
>```pseudocode
>Input: s = "dcab", pairs = [[0,3],[1,2]]
>Output: "bacd"
>Explaination: 
>Swap s[0] and s[3], s = "bcad"
>Swap s[1] and s[2], s = "bacd"
>```
>
>- `1 <= s.length <= 10^5`
>- `0 <= pairs.length <= 10^5`
>- `0 <= pairs[i][0], pairs[i][1] < s.length`
>- `s` only contains lower case English letters.

* 解析：https://leetcode.com/problems/smallest-string-with-swaps/discuss/387524/Short-Python-Union-find-solution-w-Explanation

  他的意思是我们可以使用并查集来做，对所用同一集合内的元素我们直接排序便是。然后按次序从每个集合中拿出来排序最小字符，最终得到最小字符串。

  > The core of the idea is that if (0, 1) is an exchange pair and (0, 2) is an exchange pair, then any 2 in (0, 1, 2) can be exchanged.
  >
  > This implies, we can build connected components where each component is a list of indices that can be exchanged with any of them. In Union find terms, we simply iterate through each pair, and do a union on the indices in the pair.
  > At the end of the union of all the pairs, we have built connected component of indices that can be exchanged with each other.



```python
import collections
class Solution:
    def smallestStringWithSwaps(self, s: str, pairs: List[List[int]]) -> str:
        class UnionSet:
            def __init__(self, n):
                self.rank = [1] * n
                self.parent = list(range(n))
            def find(self, x):
                if x != self.parent[x]:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]
            def union(self, x, y):
                px, py = self.find(x) , self.find(y)
                if self.rank[px] >= self.rank[py]:
                    self.parent[py] = px
                    self.rank[px] += self.rank[py]
                else:
                    self.parent[px] = py
                    self.rank[py] += self.rank[px]
        
        res, us, dic = [], UnionSet(len(s)), collections.defaultdict(list)
        for x, y in pairs:
            us.union(x, y)
        for i in range(len(s)):
            dic[us.find(i)].append(s[i])
        for key in dic.keys():
            dic[key].sort(reverse = True)
        for i in range(len(s)):
            res.append(dic[us.find(i)].pop())
        return "".join(res)
```

