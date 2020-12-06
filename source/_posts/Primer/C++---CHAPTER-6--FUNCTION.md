---
title: C++---CHAPTER-6---FUNCTION
mathjax: true
date: 2019-09-04 21:53:17
tags: PRIMER
categories: C++
visible:
---

#### 参数传递
1. 传值调用
* 指针形参：
```
void reset(int *p)
{
  *ip = 0; // 改变指针ip所指对象的值
   ip = 0; // 只改变了ip的局部拷贝，实参未被改变
{
```
2.传引用调用
* 如果函数无需改变引用形参的值，最好将其声明为常量引用。
* `const`形参和实参，当使用实参初始化形参，会忽略掉顶层`const`：也就是说，当形参有顶层`const`,传给它常量对象或者非常量对象都是可以的：
```
void fn(const int i) {/* fn能读取i，但是不能向i写入值*/}
```
* 尽量使用常量引用如 `const i&`，而不是普通引用`i &`,因为我们不能把const对象、字面值、或者需要类型转换的对象传递给普通的引用：
```
string is_sentence(const string &s)
{
  string::size_type ctr = 0;
  return find_char(s, '.',ctr) == s.size() - 1 && ctr == 1;
}
```
其中如果`find_char`的第一个形参是普通引用`string&`，程序会失败，因为`s`是常量引用。


#### 返回类型和`return`语句

* 引用返回左值：调用一个返回引用的函数返回`左值`，其他返回类型得到右值,和其他左值一样它也能出现在赋值运算符的左边：
```
char &get_val(string &*str, string::size_type ix)
{
  return str[ix];
}
int main()
{
  string s("a value");
  cout << s << endl;
  get_val(s, 0) = 'A'; // 将s[0]的值改为A
cout<< s << endl;
}
```
如果返回类型是常量引用，则不能给结果赋值：
```
shorterString("hi", "bye") = "X"; //错误，返回类型是一个常量
```

* 列表初始化返回值：函数可以返回花括号`{}`包围的值：
```
vector<string> process()
{
  if (expected.empty())
  return {}; // 返回一个空的vector对象
else if (expected == actual)
  return {"functionX", "okey");  //返回列表初始化的vector
else
  return {"functionX", expected, actual};
}
```

* 返回数组指针
定义一个返回数组指针的函数，则数组的维度必须跟在函数名字之后。

```
int arr[10];
int (*p2)[10] = &arr; //p2是一个指针，指向含有10个整数的数组

int (* func(int i))[10];  //解引用func的调用得到一个大小为10的数组
```

* 使用尾置返回类型，进一步简化返回类型复杂的函数
```
auto func(int i)->int(*)[10];  //func返回的类型是一个指针，该指针指向含有10个整数的数组。
```
* 使用`decltype`的情况，我们知道函数返回的指针指向哪个数组：
```
int odd[] = {1,2,3,4,5};
int even[] = {3,4,52,,5,1};
// 已知返回一个指针指向5个整数的数组
decltype(odd) *arrPtr(int i)
{
  return (i % 2) ? &odd : &even; //返回一个指向数组的指针
}
 ```
注：`decltype`表示他的返回类型是个指针，并且该指针所指的对象与odd的类型一致。但是`decltype`不会把数组类型转换成对应的指针，所以`decltype`的结果是一个数组，要表示`arrPtr`是一个指针，必须在函数声明的时候加上一个`*`。


* 重载和`const`形参
顶层`const`不影响传入函数的对象：
```
Record  lookup(Phone);
Record lookup(const Phone); //重复声明了上面的函数
```
当形参是某种类型的指针或引用，则区分常量对象和非常量对象可以实现函数重载：
```
Record lookup(Account&);  //作用于Account的引用
Record lookup(const Account&); // 新函数，作用于常量引用
```
* `const_cast`和重载 ：重载函数时，对实参做强制转换成对`const`的引用。
```
#include <iostream>
#include <string>
using namespace std;

const string &shorterString(const string &s1, const string &s2)
{
    return s1.size() <= s2.size() ? s1 : s2;
}
string &shorterString(string &s1, string &s2)
{
  auto &r = shorterString(const_cast<const string&>(s1),const_cast<const string&>(s2));
  return const_cast<string&>(r);
}
int main() {

  string s1{"pig"};
  string s2{"dunk"};
  string &ans1 = shorterString(s1,s1);
  cout <<" ans1 is " << ans1 << endl;
  const string s3{"moster"};
  const string s4{"tigger"};
  const string &ans2 = shorterString(s3,s4);
  cout <<" ans2 is " << ans2 << endl;
}

```

* `constexpr`函数：是指能用于常量表达式的函数。
函数的返回类型及所有形参的类型都得是字面值类型；
函数体中必须有且只有一条`return`语句。
为了在编译过程随时展开，`constexpr`函数被隐式地指定为内联函数。
```
constexpr  int new_sz() {return 42;}
constexpr int foo = new_sz(); //正确 foo是一个常量表达式

constexpr size_t scale(size_t cnt){
    return new_sz() * cnt;
 }
int arr[scale(2)]; // 正确：scale(2)是常量表达式
int i = 2; //i不是常量表达式
int a2[scale(i)]; //错误
```

注： 对于某个给定的内联函数或者`constexpr`函数来说，它的多个定义必须完全一致。因此，内联函数和`constexpr`函数通常定义在头文件中。

* `assert`预处理宏，预处理变量，有预处理器而非编译器管理，因此可以直接使用处理名字无须使用`using`声明。
* 编译器为每一个函数定义了：
```
__FILE__   
__LINE__
__TIME__
__DATE__
__FUNC__
```
* 函数指针：指向的是函数而非对象，函数的类型有它的返回值还有它的形参共同决定，与函数名无关。

```
bool lengthCompare(const string&, const string &);
bool (*pf) (const string &, const string &); // 未初始化，pf指向一个函数，该函数的参数是两个const string的引用，返回值是bool类型
```

* 使用函数指针：函数名作为一个值使用时，该函数自动地转换成指针：
```
pf = lengthCompare;  // pf 指向名为lengthcompare的函数
pf = &lengthCompare;  //  等价上式


bool b1 = pf("hello", "goodbye"); // 调用lengthcompare函数
bool b2 = (*pf)("hello", "goodbye"); //等价的调用
```

* 函数指针形参：形参定义为指向函数的指针，所以看着像函数类型，实际被当做指针使用：
```
// 第三个形参是函数类型，自动转换为指向函数的指针；
void useBigger(const string &s1, const string &s2, bool pf(const string &, const tring &));
//等价的声明
void useBigger(const string &s1, const string &s2, bool (*pf) (const string &, const tring &));
```
这就能直接把函数作为实参使用：
```
useBigger(s1, s2, lengthCompare);
```

* 考虑使用类型别名和`decltype`简化声明：
```
// Func与Func2是函数类型
typedef bool Func(const string &， const string &);
typedef bool decltype(lengthCompare) Func2; //等价的类型

// 下面两个是指向函数的指针
typedef bool(*FuncP) (const string&, const string&);
typedef decltype(lengthCompare) *FuncP2; // 等价的类型
```
注：
1. 含有`typedef`的声明语句定义的不在是变量而是类型别名。
2. `decltype`返回函数类型，不会将函数类型自动转换为指针类型，所以结果的前面加上`*`，得到指针。

使用的时候：
```
void useBigger(const string &s1, const string &s2, Func);
void useBigger(const string &s1, const string &s2, FuncP2);
```

* 返回指向函数类型的指针：必须把返回类型写成指针形式，编译器不会自动将函数返回的类型当成对应的指针类型处理。

使用类型别名：
```
using F = int(int* , int); // F是函数类型不是指针
using PF = int(*)(int *, int); // PF是之真理类型
PF f1(int);  // PF是指向函数的指针，f1返回指向函数的指针
```
尾置返回类型的方式声明一个返回函数指针的函数：
```
auto f1(int) -> int(*) (int*, int);
```
