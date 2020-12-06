---
title: 头条2018年题
mathjax: true
date: 2019-10-20 21:53:32
tags: [Acwing, Algorithm]
categories: 算法与数据结构
visible:
---



#### 863. 国庆旅行 

https://www.acwing.com/problem/content/865/

```c++
#include<iostream>
using namespace std;
const int N = 100006;
int arr[N];
int main()
{
    int n; cin >> n;
    for(int i=1; i<= n; i++)
        cin >> arr[i];
    int Max = arr[1] + 1;
    int res = 0;
    for (int j = 2; j <= n; j++)
    {
        res = max(res, Max + arr[j] - j);
        Max = max(arr[j] + j, Max);
    }
    cout <<  res;
}
```

#### 864. 二维数组区块计数

题：https://www.acwing.com/problem/content/866/

心得：图的宽搜和深搜，其中用一维数组模拟二维数组的方式实现图的读取。坐标使用`tyepdef pair<int, int > PII`来完成。

```c++
#include<bits/stdc++.h>
using namespace std;
typedef pair<int, int> PII;
const int M = 1e6 + 3;
int table[M];
int n, m;
PII q[M];

void bfs(int x, int y)
{
    int hh = 0, tt = 0;
    q[0] = {x, y};
    while(hh <= tt)
    {
        PII t = q[hh++];
        x = t.first, y = t.second;   
        
        table[x *m + y] = 0;
        for (int i = -1; i < 2; i++)
            for (int j = -1; j < 2; j++)
                {
                    int a = x + i, b = y + j;
                    if (a >=0 && a< n && b >=0 && b < m && table[a * m + b])
                        q[++tt] = {a, b};
                }
    }
}
int main()
{
    cin >> n >> m;
    for (int i=0; i<n; i++)
        for(int j=0; j<m; j++)
            cin >> table[i * m + j];
    
    int res = 0;
    for (int i = 0; i< n; i++)
        for(int j = 0; j < m; j++)
        {
            if (table[i * m + j] == 1)
            {
                bfs(i, j);
                res++;
            }
        }
    cout << res;
}
```

```c++
#include<bits/stdc++.h>
using namespace std;
const int M = 1e6 + 3;
int table[M];
int n, m;

void dfs(int x, int y)
{
    table[x *m + y] = 0;
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++)
        {
            int a = x + i, b = y + j;
            if (a >= 0 && a < n && b >= 0 && b < m && table[a * m + b])
                    dfs(a, b);
        }
}
int main()
{
    cin >> n >> m;
    for (int i=0; i<n; i++)
        for(int j=0; j<m; j++)
            cin >> table[i * m + j];
    
    int res = 0;
    for (int i = 0; i< n; i++)
        for(int j = 0; j < m; j++)
        {
            if (table[i * m + j] == 1)
            {
                dfs(i, j);
                res++;
            }
        }
    cout << res;
}
```

#### 865. 字符串展开

题：https://www.acwing.com/problem/content/description/867/

解：递归的方式dfs。

```c++
/*
递归的方式来做这个题
*/
#include <bits/stdc++.h>
using namespace std;

string decode(int &u, string &s)
{
    string res;
    while(u < s.size())
    {
        char c = s[u];
        if (c == '#') return res;
        if (c >= '0' && c <= '9') 
        {
            int k = 0;
            while(s[u] != '%')
                k = k * 10 + s[u++] - '0';
            u++;
            string sub = decode(u, s);
            while(k--)
                res += sub;
            u++;
        }
        else
        {
            res += c;
            u++;
        }
    }
    return res;
}
int main()
{
    string s; cin >> s;
    int start = 0;
    cout << decode(start, s);
    
}
```

