---
title: C++---CHAPTER-11---ASSOCIATIVE-CONTAINER
mathjax: true
date: 2019-09-04 21:53:35
tags: PRIMER
categories: C++
visible:
---

#### 小结
关联容器通过关键字查找和提取元素。对关键字的使用将关联容器与顺序容器区分开来，顺序容器中是通过位置访问元素的。

标准库定义了8个关联容器，每个容器：
* 是一个`map`或者是一个`set`。`map`保存关键字-值对；`set`只保存关键字。
*  要求关键字唯一或不要求。
* 保持关键字有序或不保证有序。


2. 允许重复关键字的容器的名字都包含有`multi`，而使用哈希技术的容器的名字都以`unordered`开头。`set`是一个有序集合，其中每个关键字只可以出现一次；而`unordered_multiset`则是一个无序的关键字集合，其中关键字可以出现多次。


------------------

#### 使用关联容器
* 使用`map`：
定义一个`map`需要指定关键字和值的类型；
`map`中提取的`pair`是一个模板类型，保存两个名为`first`和`second`的成员。
```
int main() {
  map<string,size_t> word_count;
  string word;
  while(cin>>word && word!="stop")
    ++word_count[word];
  for(const auto &w: word_count)
    cout << w.first << " occurs " << w.second
    << ((w.second > 1) ? " times" : "time") << endl;
}
```


* 使用`set`:
`find`调用返回一个迭代器，如果为找到,`find`返回尾后迭代器。
```
  map<string,size_t> word_count;
  //给定类型；列表初始化关联容器
  set<string> exclude={"The", "but","And"};
  string word;
  while(cin>>word && word!="stop")
    if(exclude.find(word)==exclude.end())
        ++word_count[word];
```

#### 关联容器概述

