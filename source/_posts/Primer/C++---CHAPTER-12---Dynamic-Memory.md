---
title: C++---CHAPTER-12---Dynamic-Memory
mathjax: true
date: 2019-09-04 21:53:37
tags: PRIMER
categories: C++
visible:
---

### 静态内存、栈内存、动态内存
* 静态内存：保存局部`static`对象；类`static`对象、已经定义在任何函数之外的变量。
* 栈内存：保存定义在函数内的非`static`对象；

其中栈对象仅在其定义的程序块运行时才存在；`static`对象在使用之前分配，程序结束时销毁。

* 内存池（称为堆`heap`）：存储动态分配的对象，即在程序运行时分配的对象，动态内存不在使用时，代码必须显式地销毁它们。

提出智能指针：为了防止意外发生，针对`new`与`delete`处理失误的情况；

* 智能指针（`smart pointer`）：定于在`memory`头文件中，有`shared_ptr`、`unique_ptr`；伴随类`weak_ptr`表示的弱引用。

#### shared_ptr
```
shared_ptr<string> p1; //可以指向string
shared_ptr<list<int>> p2;
```

```
//如果p1不为空，检查它是否指向一个空string
if(p1 && p1->empty())
  *p1 = "hi"; //如果p1指向一个空string，解引用p1，将一个新值赋予string
```

* `make_shared`函数
使用时必须指定想要创建的对象的类型：
```
shared_ptr<int> p3 = make_shared<int>(42);
//p4指向一个值为"999999999"的string
shared_ptr<string> p4 = make_shared<string>(10,'9');
//p5指向值初始为0的int
shared_ptr<int> p5 = make_shared<string>();
```
通常使用`auto`：
```
auto p6 = make_shared<vector<string>>();
```

拷贝和赋值

```
auto p = make_shared<int>(42);  // p指向 的对象只有p一个引用
auto q(p);  //p和q指向相同的对象，此对象有两个引用者
```
可以认为被一个`shared_ptr`都关联一个**引用计数**,一旦计数器的值变为0，它就会自动释放自己所管理的对象：
```
auto r = make_shared<int>(42); //r指向的int只有一个引用者
r = q; //给r赋值，令它指向另一个地址；
          // 递增q指向的对象的引用计数；
    //   递减r原来指向的对象的引用计数；
  //r原来指向的对象已没有引用者，会自动释放；
```

下面的例子展示了使用动态内存的一个常见的原因是允许多个对象共享相同的状态：
```
Blob<string> b1; //空Blob
{
    Blob<string> b2 = {"a"};
    b1 = b2; //b1与b2共享相同的元素，b2被销毁了，但b2中的元素不能销毁；b1指向由最初b2创建的元素；
}
```


练习12-2:定义自己的`const`函数：
```
//
// Created by zc on 2019-08-21.
//

#include<iostream>
#include<vector>
#include<memory>
using namespace std;

class StrBlob
{
public:
    typedef vector<string>::size_type size_type;
    StrBlob();
    StrBlob(initializer_list<string> il);
    size_type size() const {return data->size();}
    bool empty() const { return data->empty();}
    // 添加、删除元素
    void push_back(const string &t){data->push_back(t);}
    void pop_back();
    // 访问元素
    string& front();
    string& back();
    string& front() const;
    string& back() const;
private:
    shared_ptr<vector<string>> data;
    void check(size_type i, const std::string &msg) const;

};

StrBlob::StrBlob() : data(make_shared<vector<string>>()) {}
StrBlob::StrBlob(initializer_list<string> il) : data(make_shared<vector<string>>(il)){}
//检查是否越界
void StrBlob::check(size_type i, const string &msg) const
{
    if (i >= data->size())
        throw out_of_range(msg);
}

string& StrBlob::front()
{
    check(0, "front on empty StrBlob");
    return data->front();
}

string& StrBlob::back()
{
    check(0,"back on empty StrBlob");
    return data->back();
}

//练习12-2
//目的：非常量函数不能够调用常量对象
string& StrBlob::front() const
{
    check(0, "front on empty StrBlob");
    return data->front();
}
string& StrBlob::back() const
{
    check(0, "front on empty StrBlob");
    return data->back();
}
void StrBlob::pop_back()
{
    check(0, "pop_back on empty StrBlob");
    data->pop_back();
}



//test
int main()
{
    const StrBlob csb{"hello", "world", "zhengchu"};
    StrBlob sb{"hello", "world", "jojo"};
    std::cout<< csb.front() << " " << csb.back() << endl;
    sb.back() = "dio";
    cout<< sb.front() << " " << sb.back() << endl;
    return 0;
}

hello zhengchu
hello dio
```

12-3:需要为上面的类添加`const`版本的`push_back`和`pop_back`吗
参考：[https://www.douban.com/group/topic/61573279/](https://www.douban.com/group/topic/61573279/)

Ans：
可以用的原因，因为修改的不是指针，而是指针指向的数据，因此完全可以用`const`指针。
不可以用的原因：虽然在类的具体实现中，数据成员是一个指向vector的智能指针；但由于类的封装，在类的使用者看来，数据成员是vector，他们并不知道具体的实现使用了智能指针。那么当类的使用者声明类的常量对象时，他们期待的结果是vector的内容不会被改变。所以我们在设计这个类的时候，要考虑到类的使用者的真实意图，对于像push_back和pop_back这样会改变智能指针所指向的vector内容的成员函数，我们不应该声明和定义成const版本。这样在类的使用者使用类的常对象时，就不能调用push_back和pop_back成员函数，不能改变智能指针所指向的vector的内容了，这正好与类的使用者的意图相符。 

