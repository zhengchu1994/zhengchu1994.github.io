---


title: 精通c++的拼写
mathjax: true
date: 2020-06-30 19:00:00
tags: C++
categories: C++
visible:

---



#### 左右值

> A useful heuristic to determine whether an expression is an lvalue is to ask if you can take its address. If you can, it typically is. If you can’t, it’s usually an rvalue. A nice feature of this heuristic is that it helps you remember that the type of an expression is independent of whether the expression is an lvalue or an rvalue.



另外还有两个有重要的准则:

1. 对象 (对象是最简单的表达式) 被用作左值时, 用的是对象的地址; 被用作右值时, 用的是对象的值
2. 在需要右值的地方, 可以用左值代替 (因为地址中存储着值), 这时左值被当成右值使用, 用的是左值里存储的值. 这条准则只有一个例外, 就是左值引用不能当做右值引用使用 (下面会讲到引用)



#### 对象移动

对象移动的目的是减少对象拷贝, 被移动的对象内容会被 "掏空", 赋予新对象中, 这个操作依赖于 "移动构造函数" 或者 "移动操作符", 然而如何定义这两个方法却成了一个问题, 我们知道拷贝构造函数和赋值操作符是以对象的引用 (左值引用) 为参数的, 而如果要定义移动构造函数或移动操作符, 其参数当然也得以代表对象的某种形态为参数, 对象的引用这种形态已经不能用了, 对象的非引用更不能用 (原因见 *c++ primer 5th, p422*), 那么只能新发明一中语义了, 这个新的语义就是右值引用, 写作 `T &&`. 于是移动构造函数就可以这么定义了:

```
Foo (Foo && other);
```

如此一来, 只要传递给构造函数的是右值引用, 就会调用移动构造函数, 免去拷贝所造成的开销.



https://steemit.com/cn-programming/@cifer/6c4dox-c

https://zhuanlan.zhihu.com/p/85668787





#### CONST笔记

* 列表初始化：
使用列表初始化且初始值存在丢失信息的风险，则编译器报错：
```c++
int unit_sold = {0};  // list initialization
long double ld =3.14159;
int a{id}, b = {id}; //错误
int c(id), d = ld; // 正确
```

* 默认初始化：
定义于任何函数体之外的内置类型变量被初始化为0，定义在函数体内部的内置类型变量不被初始化。

* 声明与定义：
变量只能定义一次，但能被多次声明；
关键字`extern`表示声明变量；
任何包含了显示初始化的声明即成为定义；
函数内部初始化一个由extern关键字标记的变量，会报错。
```c++
extern int i; //声明而非定义i
int j; //声明并定义j
extern double pi = 3.1416; // 定义
```
* 复合类型：指基于其他类型定义的类型，如引用，指针等。
声明语句的组成：基本数据类型 + 声明符，每一个声明符命名了一个变量并指定该变量为与基本数据类型有关的某种类型。
  - 引用类型：引用另一种类型,是已经存在的对象的别名，因此不是对象，所以必须初始化。
```c++
int ival = 1024;
int &refVal = ival; 
int &refVal2; // 报错
```

  - 指针：
```c++
int ival = 42;
int *p = &ival;  // &取地址符，p存放变量ival的地址。
cout << *p ; // *解引用符，得到p所指的对象。
```
  - 空指针：不指向任何对象.
```c++
int *p1 = nullptr;
int *p2 = 0;
int *p3 = NULL; // 需要include<cstdlib>
```

* 指向指针的指针
```c++
int ival = 1024;
int *pi = &ival; // pi是指向int的指针
int **ppi = &pi; // ppi是指向int指针的指针
```
* 指向指针的引用：
指针是对象，存在对指针的引用，但是引用不是对象，不存在指向引用的指针；
离变量最近的符号对变量的类型有最直接的影响，因此r是一个引用。
```c++
int i = 42;
int *p; 
int *&r = p ; //r是一个对指针p的引用
```
### const
`const`限定的变量，其值不能改变，因此必须初始化。
`const`变量初始化另一个对象不会改变这个变量：
```c++
int i = 42;
const int ci = i;
int j = ci; //正确；拷贝完成，新的对象和原来的对象就没有关系了；
```

* **默认状态下，const对象仅在文件内有效**，因此，当多个文件都要用这个`const`变量的时候，在每个文件都要定义该`const`变量，
只在一个文件中定义`const`,而在其他文件中声明且使用它的办法：在定义前加`extern`关键字：
```c++
//file.cc 定义并初始化了一个常量
extern const int bufSize = fcn();
// file_1.h头文件
extern const int bufSize; // 与file.cc中定义的bufSize是同一个
```

* 对常量的引用

```c++
const int ci = 1024;
const int &r1=ci; //正确
int &r2 =ci; //错误
```

对`const`的引用可能引用一个并非const的对象：

```c++
int i = 42;
int &r1 = i;
const int &r2 =i; 
r1 = 0; //正确
r2 = 0; //错误
```
* 指向常量的指针
```c++
const double pi = 3.1415;
double *ptr = &pi; //错误
const double *cptr = &pi; //正确


double dval = 3.14;
cptr = &dval; //正确
```

* `const`指针 ：形式为 `*const`，说明指针是一个常量：
```c++
int errNumb = 0;
int *const curErr = &errNumb; //只能指向errNumb 

const double pi = 3.14159;
const double *const pip = &pi; // pip是一个指向常量对象的常量指针

*pip = 2.72; // 错误，pip是一个指向常量的指针
* curEr = 0; // 正确，所指对象的值重置
```
* 顶层const
顶层`const` 表示指针本身是个常量，底层`const`表示指针所指的对象是一个常量。
指针类型即可以是顶层`const`也可以是低层`const`：
```c++
int i =0 ;
int *const p1 = &i; // 不能改变p1的值，是一个顶层const
const int ci = 42; //不能改变ci的值，是一个顶层const
const int *p2 = &ci; // 允许改变p2的值，这是一个低层const
const int *const p3 = p2; //靠右边的const是顶层const，靠左边的是低层const
const int &r =ci; // 用于声明引用的cosnt都是低层const
```



* constexpr常量表达式：变量声明为`constexpr`类型以便编译器验证变量是否是一个常量表达式：
```c++
constexpr int mf = 20;
cosntexpr int limit = mf +1;
constexpr int sz = size(); //只有size是一个constexpr函数时才是一条正确的声明语句
```

`constexpr`声明中如果定义指针，限定符`constexpr`仅对指针有效，与指针所指的对象无关：

```c++
const int *p = nullptr; // p是一个指向整型常量的指针
constexpr int *q = nullptr; //q是一个指向整型的常量指针 ，顶层const
```
与其他常量一样，`constexpr`能指向常量与非常量：
```c++
int j = 0;
constexpr int i = 42; // i的类型是整型常量

constexpr const int *p = &i ; // p是常数指针，指向整型常量i
constexpr int *p1 = &j; //p1是常量指针，指向整数j
```

* 类型别名
```c++
typedef double wages; //wages 是double的同义词
using SI = Sales_item; // SI是Sales_item的同义词
```


* auto类型说明符

一条语句声明多个变量，但是变量的初始数据类型必须一样：
```c++
auto i =0, *p = &i; //正确：i是整数、p是整型指针
auto sz=0, pi = 3.14; //错误，类型不一样
```
`auto`类型和初始值类型并不一定一样，如忽略顶层`const`，保留低层`const`：
```c++
int i =0, &r = i; 
auto a = r; // a是一个整数 

const int ci = i, &cr  = ci;
auto b = ci; // b是一个整数（ci的顶层const特性被忽略）
auto c = cr; // c是一个整数 （cr是ci的别名，ci本身是一个顶层const）
auto d = &i; // d是一个整型指针（整数的地址就是指向整数的指针）
auto e = &ci; // e是一个指向整数常量的指针（对常量对象去地址是一种低层const）
```
希望推断出的`auto`类型是一个顶层`const`,需要明确：
```c++
const auto f = ci; //  ci的推演类型是int，f是const int
```
引用的类型设为`auto`,上述规则依然适用：
```c++
auto &g = ci; //g是一个整型常量引用，绑定到ci
auto &h = 42; //错误，不能为非常量引用绑定字面值
const auto &j = 42; //正确：可以为常量引用绑定字面值
```

* `decltype`类型指示符：推断要定义的变量的类型
`decltype`使得变量，则返回该变量的类型：
```c++
const int ci = 0, &cj = ci;
decltype(ci)  x = 0; //x的类型是const int
decltype(cj) y = x; // y的类型是const int&, y绑定到变量x
```
`decltype((variable))`的结果永远是引用：

```c++
int i = 42;
decltype((i)) d; //错误，d是int&，必须初始化；
```

### 术语表

| 声明（declaration）    | 声称存在一个变量、函数或是别处定义的类型。名字必须在定义或者声明之后才能使用。 |
| ---------------------- | ------------------------------------------------------------ |
| **定义（defination）** | **为某一特定类型的变量申请存储空间，可以选择初始化该变量。  名字必须在定义或者声明之后才能使用。** |





#### 智能指针



```c++
#include<iostream>
#include<memory>
#include<string>
#include<vector>
#include <initializer_list>
using namespace std;

class StrBlob{
public:
    typedef std::vector<std::string>::size_type size_type;
    StrBlob();
    StrBlob(std::initializer_list<std::string> il);
    size_type size() const {return data->size();} //常量成员函数
    bool empty() const {return data->empty();}
    void push_back(const std::string &t){data->push_back(t);}
    void pop_back();
    std::string& front();
    std::string& back();
    const std::string& front() const{
        check(0, "front on empty StrBlob");
        return data->front();
    }
    const std::string& back() const{
        check(0, "back on empty StrBlob");
        return data->back();
    }

private:
    std::shared_ptr<std::vector<std::string> > data;
    void check(size_type i, const std::string &msg) const;
};

StrBlob::StrBlob(): data(make_shared<vector<string> >()){}
StrBlob::StrBlob(std::initializer_list<string> il): data(make_shared<vector<string> >(il)){}

void StrBlob::check(size_type i, const string &msg) const
{
    if (i >= data->size()) throw out_of_range(msg);
}

string& StrBlob::front()
{
    check(0,"front on empty StrBlob");
    return data->front();
}


string& StrBlob::back()
{
    check(0, "pop_back on empty StrBlob");
    return data->back();
}

void StrBlob::pop_back()
{
    check(0, "pop-back on empty StrBlob");
    data->pop_back();
}


int main()
{
    const StrBlob csb{"see you in"};
    StrBlob sb{"2077"};
    cout << csb.front() << endl;
    cout << sb.back() << endl;
    return 0;
}
```





#### lambda表达式



```
[外部变量访问方式说明符] (参数) mutable noexcept/throw() -> 返回值类型
{
   函数体;
};

```

| 外部变量格式      | 功能                                                         |
| ----------------- | ------------------------------------------------------------ |
| []                | 空方括号表示当前 lambda 匿名函数中不导入任何外部变量。       |
| [=]               | 只有一个 = 等号，表示以值传递的方式导入所有外部变量；        |
| [&]               | 只有一个 & 符号，表示以引用传递的方式导入所有外部变量；      |
| [val1,val2,...]   | 表示以值传递的方式导入 val1、val2 等指定的外部变量，同时多个变量之间没有先后次序； |
| [&val1,&val2,...] | 表示以引用传递的方式导入 val1、val2等指定的外部变量，多个变量之间没有前后次序； |
| [val,&val2,...]   | 以上 2 种方式还可以混合使用，变量之间没有前后次序。          |
| [=,&val1,...]     | 表示除 val1 以引用传递的方式导入外，其它外部变量都以值传递的方式导入。 |
| [this]            | 表示以值传递的方式导入当前的 this 指针。                     |



```c++
#include<iostream>
#include<vector>
#include<algorithm>
using namespace std;

int main()
{
    auto print = [](vector<int> vec) -> void
    {
        for(auto v: vec) cout << v << " ";
        cout << endl;
    };
    vector<int> vec{1, 2, 3, 4, 222, 10};
    print(vec);

    sort(vec.begin(), vec.end(), [&](const auto a, const auto b){; return a < b;});
    print(vec);
    return 0;
}



-----
  
 [Running] cd "/Users/zhengchu/Desktop/Mxnet-snippet/" && g++ -std=c++17 lambda.cpp -o lambda && "/Users/zhengchu/Desktop/Mxnet-snippet/"lambda
1 2 3 4 222 10 
1 2 3 4 10 222 
```

* 计算偶数：

```c++
int main()
{
		vector<int> p = {1, 2, 3, 4, 5 , 6};
    int even_count = 0;
    for_each(p.begin(), p.end(), [&even_count](const int& val)
    {
        if(!(val&1)) even_count++;
    });
    cout << "The number of even is " << even_count << endl;
    return 0;
}
----

The number of even is 3
```



#### C++强制类型转换运算符

## static_cast

static_cast 用于进行比较“自然”和低风险的转换，如**整型和浮点型、字符型之间的互相转换**。另外，如果对象所属的类重载了强制类型转换运算符 T（如 T 是 int、int* 或其他类型名），则 static_cast 也能用来进行对象到 T 类型的转换。

static_cast 不能用于在不同类型的**指针之间互相转换**，也不能用于**整型和指针之间的互相转换，当然也不能用于不同类型的引用之间的转换**。因为这些属于风险比较高的转换。



## reinterpret_cast

reinterpret_cast 用于进行各种不同类型的指针之间、不同类型的引用之间以及指针和能容纳指针的整数类型之间的转换。转换时，执行的是逐个比特复制的操作。

这种转换提供了很强的灵活性，但转换的安全性只能由程序员的细心来保证了。

**一般别用就是了**

reinterpret_cast体现了 C++ 语言的设计思想：用户可以做任何操作，但要为自己的行为负责。(天花乱坠术)



## const_cast

const_cast 运算符仅用于进行去除 const 属性的转换，它也是四个强制类型转换运算符中唯一能够去除 const 属性的运算符。

将 const 引用转换为同类型的非 const 引用，将 const 指针转换为同类型的非 const 指针时可以使用 const_cast 运算符。例如：

```c++
const string s = "Inception";
string& p = const_cast <string&> (s);
string* ps = const_cast <string*> (&s);  // &s 的类型是 const string*
```



## dynamic_cast

dynamic_cast专门用于将多态基类的指针或引用强制转换为派生类的指针或引用，而且能够检查转换的安全性。对于不安全的指针转换，转换结果返回 NULL 指针。

```c++
#include<iostream>
using namespace std;

class A 
{
public:
    virtual ~A(){}
};
class B: public A
{
    public:
        operator uint64_t() {return 1;}
};

int main()
{
    //static_cast
    B a;
    uint64_t n = static_cast<uint64_t> (-1); 
    cout << n << endl;  // n = 18446744073709551615
    n = static_cast<uint64_t>(a); //调用 a.operator int，n 的值变为 1
    cout << n << endl;

    //const_cast
    const string s = "Inception";
    string& p = const_cast<string&> (s);
    cout << p << endl;
    string* ps = const_cast<string*> (&s); // &s 的类型是 const string*
    
    //dynamic_cast
    A b;
    B d;
    B *pd;
    pd = dynamic_cast<B*>(&b);
    if(pd == NULL) cout << "unsafe dynamic_cast1" << endl;
    pd = dynamic_cast<B*>(&d);
    if(pd != NULL) cout << "safe dynamic_cast2" << endl;
    return 0;
}

-----
[Running] cd "/Users/zhengchu/Desktop/Mxnet-snippet/" && g++ -std=c++17 cast.cpp -o cast && "/Users/zhengchu/Desktop/Mxnet-snippet/"cast
18446744073709551615
1
Inception
unsafe dynamic_cast1
safe dynamic_cast2

```





#### uint8_t / uint16_t / uint32_t /uint64_t

：具体定义在：`/usr/include/stdint.h `

```c++

#ifndef __int8_t_defined  
# define __int8_t_defined  
typedef signed char             int8_t;   
typedef short int               int16_t;  
typedef int                     int32_t;  
# if __WORDSIZE == 64  
typedef long int                int64_t;  
# else  
__extension__  
typedef long long int           int64_t;  
# endif  
#endif  
  
  
typedef unsigned char           uint8_t;  
typedef unsigned short int      uint16_t;  
#ifndef __uint32_t_defined  
typedef unsigned int            uint32_t;  
# define __uint32_t_defined  
#endif  
#if __WORDSIZE == 64  
typedef unsigned long int       uint64_t;  
#else  
__extension__  
typedef unsigned long long int  uint64_t;  
#endif 
————————————————
版权声明：本文为CSDN博主「海阔天空sky1992」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/Mary19920410/article/details/71518130
```

