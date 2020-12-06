---
title: JaccardDistance
mathjax: true
date: 2019-09-04 21:53:56
tags: Cooking
categories: C++
visible:
---

#### Jaccard距离

记录一下前几天面试遇到的一个题，计算两个字符串的`Jaccard`距离：

1.Jaccard系数的定义是：两串交集的长度比上并集的长度，当A、B串都是空串的时候，系数为1：

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/eaef5aa86949f49e7dc6b9c8c3dd8b233332c9e7)

Jaccard距离的定义是1减去Jaccard系数：

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/3d17a48a5fb6cea57b076200de6edbccbc1c38f9)



2. c++中对集合里的元素取交集和并集的操作分别是：`set_intersection`、`set_union`，注意两个输入串必须是按照相同排序规则排好序的串；插入的时候选择使用尾部插入迭代器`back_iterator`对空的vector做插入；

   

3. 在计算Jaccard距离之前，我先排除了串中的空白符，用到`::isspace`这个`c`中的全局变量；结合c++中string的方法`erase`删除掉空白的尾部。

   

4.  注意的是范围的时候对其中集合长度转换为double类型，避免整型相除结果为1或0.

```c++
// merge algorithm example
#include <iostream>     // std::cout
#include <algorithm>    // std::merge, std::sort
#include <set>
#include <iterator>  
#include <vector>       // std::vector
#include <algorithm>
using namespace std;

void trim(string &str)
{
    string::iterator it = remove_if(str.begin(), str.end(), ::isspace);
    str.erase(it, str.end());
}
double Jaccard(string str1, string str2)
{
    if (str1.empty() && str2.empty()) return 0;
    trim(str1); 
    trim(str2);
    sort(str1.begin(), str1.end());
    sort(str2.begin(), str2.end());
    set<char> s1(str1.begin(), str2.end());
    set<char> s2(str2.begin(), str2.end());
    vector<char> univ, interv;
	set_intersection(s1.begin(), s1.end(), s2.begin(), s2.end(), back_inserter(interv));
    for (auto it : interv)
        cout << it << " ";
    cout << endl;
    set_union(s1.begin(), s1.end(), s2.begin(), s2.end(), back_inserter(univ));
    for (auto it : univ)
        cout << it << " ";
    cout << endl;
    return 1 - double(interv.size()) / univ.size();
}
int main () {
    string A = "jojo's Adventure";
    string B = "jojo come from a cartoon";
    cout << Jaccard(A, B) << endl;
}

Output:
a c e f j m n o r t
 ' 1 A a c d e f j m n o r s t u v
0.444444
```

