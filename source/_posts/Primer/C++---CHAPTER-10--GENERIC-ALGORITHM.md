---
title: C++---CHAPTER-10---ALGORITHM
mathjax: true
date: 2019-09-04 21:53:34
tags: PRIMER
categories: C++
visible:
---

* 泛型算法：经典算法的公共接口。
* 泛型的含义：用于不同类型的元素和多种容器类型，以及其他类型的序列。

### 初识
* 例子：
泛型算法不直接操作容器，而是遍历由两个迭代器指定的一个元素范围，如`find`:
```
int val = 42;
auto result = find(vec.cbegin(), vec.cend(), val)
```
可以看到，`find`内部使用迭代器进行，这使得迭代器令算法不依赖与容器。
但是算法依赖于元素类型的操作：比如`find`要求元素类型支持`<`云算符。
并且多数算法支持我们自定义的操作代替默认的比较运算符。


* 结论：算法运行于迭代器上，执行迭代器的操作，而不会执行容器的操作；算法永远不会改变低层容器的大小。


#####  只读算法：只会读取范围内的元素，不会改变元素，如`find`、`count`。
* `numeric`里面的`accumulate`,第三个参数的类型决定了函数使用哪个加法运算符和函数值返回类型：
```
对vec中的元素求和，和的初值是0 （第三个参数）
int sum = accumulate(vec.cbegin(),vec.cend(), 0);
```
编程假定：元素类型加到和的类型上的操作必须是可行的。
```
连接所有的string元素。
string sum = accumulate(v.cbegin(), v.cend(), string(""));

错误的，因为const char*上没有定义+运算符
string sum = accumulate(v.cbegin(), v.cend(), "");
```
* `equal`比较两个序列是否保存了一样的值。
```
roster2中的元素数目应该至少与roster1一样多
equal(roster1.cbegin(), roster1.cend(), roster2.cbegin());
```
上面的`equal`接受一个单一迭代器表示第二个序列，一般这样的算法都假定了第二个序列至少与第一个序列一样长。

##### 写容器算法：这类算法要求确保序列大小至少不小于我们要求算法写入的元素数目。
* `fill`算法：接受一对迭代器、一个值：
```
将每个元素重置为0
fill(vec.begin(), vec.end(), 0); 

将容器的子序列设置为10
fill(vec.begin(), vec.begin() + vec.size() / 2, 10);
```
* `fill_n` 接受一个单迭代器、一个计数值、和一个值：
```
vector<int> vec;

将所有元素重置为0
fill_n(vec.begin(), vec.size(), 0); 

fill_n(dest, n, val);
```
上面的`fill_n`假设了`dest`开始的序列至少有包含了`n`个元素，可能犯的错误：
```
vector<int> vec; // 空容器
fill_n(vec, 10, 0);  // 错误， 该10个位置都不存在
```

* 插入迭代器 `back_inserter`：一种可以向容器添加元素的迭代器。定义在头文件`iterator`中。接受一个指向容器的引用， 返回一个与该容器绑定的插入迭代器：
```
vector<int> vec;
auto it = back_iterator(vec);
* it = 42; // vec中现在多了一个元素42
```
可以使用`back_iterator`作为算法的目的位置使用：
```
vector<int> vec; 
fill_n(back_inserter(vec), 10, 0); //现在添加了10个元素到vec
```
* `copy`算法：要求目的序列至少要与输入序列一样多的元素：
```
int a1[] = {0,1,2};
int a2[ sizeof(a1) / sizeof(*a1)]; // 与a1大小一样
auto ret = copy(begin(a1), end(a1), a2);  //把a1的内容复制给a2；
```
`copy`返回目的的位置迭代器，即`ret`恰好指向`a2`的尾元素之后的位置。

* `replace`算法：接受一对迭代器表示输入序列，一个要搜索的值、一个新值。把搜索的值变为新值：
```
把所有0元素变为42
replace(ilst.begin(), ilst.end(), 0, 42); 
```

* `replace_copy`算法：接受额外第三个迭代器参数，指出调整序列的保存位置：
```
使得原来的ilst序列不变，ivec保存了replace操作的序列
replace_copy(ilst.cbegin(), ilst.end(), back_inserter(ivec), 0, 42);
```

##### 重排容器元素的算法
* `unique`算法：使得不重复的元素出现在输入序列的开始部分，但是并非删除了重复的元素，只是覆盖了相邻的重复元素。`unique`返回的迭代器指向最后一个不重复元素之后的位置。


一个消除重复单词的例子：

```
void elimDups(vector<string> &words)
{
  sort(words.begin(), words.end());
  // unique重拍输入范围,使得每一个单词只出现一次
  auto end_unique = unique(words.begin(), words.end());
  words.erase(end_unique, words.end());
}
int main() {
  vector<string> words = {"the", "quick", "red", "fox", "jumps",
  "over", "the", "slow","red","turtle"};
  elimDups(words);
  for (auto i: words)
    cout << i << " ";
}

fox jumps over quick red slow the turtle
```


### 定制操作
比如`sort`算法默认使用`<`运算符，但我们希望排序顺序与`<`所定义的顺序不同，或者保存的序列元素未定义`<`运算符，因此都需要重载`sort`的默认行为，为此需要定制我们自己的操作。

##### 传递函数给算法

* 谓词（predicate）参数：谓词是一个可调用的表达式，其返回结果是一个能用做条件的值。标准库算法使用的两类谓词：
  1.一元谓词：只接受单一参数。
  2. 二元谓词：有两个参数。

接受谓词参数的算法对输入序列中的元素调用谓词。因此，元素类型必须能够转换为谓词的参数类型。

* `stable_sort`：保持等长元素间的字典序。
一个新`sort`例子：
```
bool isShorter(const string &s1, const string &s2)
{
  return s1.size() < s2.size();
}

elimDups(words);

这使得短的单词排在长的单词的前面；
stable_sort(words.begin(), words.end(), isShorter);
  for (const auto &s: words) //无须拷贝字符串
    cout << s << " ";


fox red the over slow jumps quick turtle
```


#### lambda表达式
* motivation：希望进行的操作需要更多的参数怎么办？
* `lambda`介绍：我们可以向一个算法传递任何类别的可调用对象（`callable object`）。
对于一个对象或一个表达式，可以对其使用调用运算符（圆括号`()`），则称它为可调用的。
比如`e`是可调用表达式，可以使用`e(args)`.
* 几种可调用对象：函数、函数指针、重载了函数调用运算符的类、`lambda`表达式。

一个`lambda`表达式可理解为一个未命名的内联函数，其形式如下：
```
[capture list] (parameter list) -> return type{function body}
```
`capture list`是一个`lambda`所在函数中定义的局部变量的列表（通常为空），其他参数与普通函数类似。
与其他函数的不同，`lambda`必须使用尾置返回来指定返回类型。
* 复习尾置返回：尾置返回类型跟在形参列表后面用一个`->`符号开头，例子：

```
func接受一个int类型的实参，返回一个指针，该指针指向含有是个整数的数组。
auto func(int i) -> int(*)[10];
可以看到，函数的返回类型放在了形式参数列表的后面。
```

`lambda`表达式必须永远包含`捕获列表`和`函数体`：
```
auto f= [] {return 42;};

调用lambda表达式：
cout << f() << endl;
```

注：如果`lambda`函数体包含任何单一return语句之外的内容，且未指定返回类型，则返回`void`。


* 向`lambda`传递参数：不能有默认参数，因此调用`lambda`的实参数目永远与形参数目相等。
与`isShorter`同样功能的`lambda`表达式：
```
 [](const string &a, const string &b){return a.size() < b.size();}
```
* 使用捕获列表：
Note：一个`lambda`只有在其捕获列表中捕获一个它所在函数中的局部变量时，才能在函数体中使用该变量。

`lambda`表达式通过将局部变量包含在捕获列表中来指出将会使用这些变量。

  一个例子，上述的`sort`单词，要求出大于等于一个给定长度的单词有多少，修改输出，打印这些单词；
```
定义一个给定长度`sz`，然后把`sz`加入捕获列表,在函数体内使用。
[sz](const string &a){return a.size() >= sz;};

错误，sz未捕获
[](const string &a){return a.size() >= sz;};
```
* `find_if`算法：接受一对迭代器，第三个参数是一个谓词。对输入序列中的每个元素调用这个谓词。返回第一个使得谓词返回非0值的元素，不存在这样的元素则返回尾迭代器。

* `for_each`算法：接受一个可调用对象，并对输入序列中每个元素调用此对象。
```
for_each(wc, words.end(), [](const string &s){cout << s << " ";});
```
上面的`lambda`空捕获列表,但是函数体使用了`s`、`cout`，因为：
Note: 捕获列表只用于局部非`static`变量，`lambda`可以之间直接使用局部`static`变量还有它所在函数之外声明的名字。

代码：
```
string make_plural(size_t ctr, const string &word, const string &ending)
{
  return (ctr > 1) ? word + ending: word;
}

  //vector<string> words = {"the", "quick", "red", "fox", "jumps","over", "the", "slow", "red", "turtle"};

void biggies(vector<string> &words, vector<string>::size_type sz) {
  elimDups(words);
  stable_sort(words.begin(), words.end(), 
  [](const string &a, const string &b){return a.size() < b.size();});
  for (const auto &s: words) //无须拷贝字符串
    cout << s << " ";
  cout << endl;
  auto wc = find_if(words.begin(),words.end(),
 
 //sz不在函数体内
  [sz](const string &a){return a.size() >= sz;});
  // wc指向words中第一个string长度大于等于5的位置，用end减去便是剩下都大于等于5的string数目
  auto count = words.end() - wc;
  cout << count << " " << make_plural(count, "word", "s")
    << " of length " << sz << " or longer " << endl;
  for_each(wc, words.end(), [](const string &s){cout << s << " ";});
}

fox red the over slow jumps quick turtle
3 words of length 5 or longer
jumps quick turtle
```

* lambda的捕获和返回

##### 参数绑定


### 再探迭代器
* 插入迭代器
插入器是一种迭代器适配器：接受一个容器，生成一个迭代器。

* 插入迭代器操作：
    * `it=t`：在`it`指定的当前位置插入值`t`.
    * `*it, ++it, it++`：不会对`it`做任何事情，都返回`it`.

* 插入迭代器有三种：
     * `back_inserter`：创建一个使用`push_back`的迭代器。
     * `front_inserter`:   创建一个使用`push_front`的迭代器。
     * `inserter`: 创建一个使用`inserter`的迭代器，接受第二个参数，该参数指定一个给定容器的迭代器。元素被插入到给定迭代器所表示的元素之前。（`inserter(c, iter)`）

1. `inserter`的例子：设`it`是由`inserter`生成的迭代器，则：
```
*it = val;

等价于：
it = c.insert(it, val); //it指向新加入的元素
++it; // 递增it使它指向原来的元素 
```

2. `front_inserter`的例子：

```
list<int> lst = {1, 2, 3, 4, 5};
list<int> lst2, lst3; 

// 拷贝完成后， lst2包含4，3，2，1
copy(lst.cbegin(), lst.cend(), front_inserter(lst2));
// lst3包含1，2，3，4
copy(lst.cbegin(), lst.cend(), inserter(lst3, lst3.begin()));
```



* 流迭代器（`iostream`迭代器）

* 反向迭代器
* 移动迭代器

### 泛型算法结构

### 特定容器算法
