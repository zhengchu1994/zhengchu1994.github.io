---
title: 从map中找到value或者key最大的iterator
mathjax: true
date: 2019-09-04 21:53:32
tags: Cooking
categories: C++
visible:
---

#### 从map中找到value或者key最大的iterator

预备点：

1. `delctype` 的使用：

c++里的两个类型推理符号`auto` 和`delctype`：

* `auto` ：auto关键字指定正在声明的变量的类型将自动从其初始化器中推得。函数的返回类型是auto，那么将在运行时通过返回类型表达式对其进行计算

```c++
int main() 
{ 
    auto x = 4; 
    auto y = 3.37; 
    auto ptr = &x; 
    cout << typeid(x).name() << endl 
         << typeid(y).name() << endl 
         << typeid(ptr).name() << endl; 
  
    return 0; 
} 

i //int整型
d //double类型
Pi // pointer to int 类型
```

* `delctype`：不经常用，所以关键是理解这个玩意。它可以检查实体的声明类型或表达式的类型，`delctype`可以从variable中抽取它的类型，所以`delctype`类似于一个操作符，它可以抽取出表达式的类型。

  ```c++
  int fun1() { return 10; } 
  char fun2() { return 'g'; } 
    
  int main() 
  { 
      // Data type of x is same as return type of fun1() 
      // and type of y is same as return type of fun2() 
      decltype(fun1()) x; 
      decltype(fun2()) y; 
      cout << typeid(x).name() << endl; 
      cout << typeid(y).name() << endl; 
      return 0; 
  } 
  output:
  i //x的类型与函数的返回类型一致
  c //y的类型与函数的返回类型一致
  ```

  一个泛型函数的例子，它的函数返回类型用`auto`声明，又用`delctype` 使得返回类型==求得的最小值的类型

  ```c++
  // A generic function which finds minimum of two values 
  // return type is type of variable which is minimum 
  template <class A, class B> 
  auto findMin(A a, B b) -> decltype(a < b ? a : b) 
  { 
      return (a < b) ? a : b; 
  } 
    
  // driver function to test various inference 
  int main() 
  { 
      // This call returns 3.44 of doubale type 
      cout << findMin(4, 3.44) << endl; 
    
      // This call returns 3 of double type 
      cout << findMin(5.4, 3) << endl; 
    
      return 0; 
  }
  Output:
  i3.44
  3
  ```

  

2. `max_element`的使用：

   * Returns an iterator pointing to the element with the largest value in the range `[first,last)`.；
   * If more than one element fulfills this condition, the iterator returned points to the first of such elements.

   函数行为等于：

   ```c++
   template <class ForwardIterator>
     ForwardIterator max_element ( ForwardIterator first, ForwardIterator last )
   {
     if (first==last) return last;
     ForwardIterator largest = first;
   
     while (++first!=last)
       if (*largest<*first)    // or: if (comp(*largest,*first)) for version (2)
         largest=first;
     return largest;
   }
   ```

   

3. 从map中找到value或者key最大的iterator：

   * `map`的`member type`之一：`value_type`定义为：`pair<const key_type,mapped_type>`。

   * `find`方法：

     ```c++
           iterator find (const key_type& k);
     const_iterator find (const key_type& k) const;
     ```

     1. Searches the container for an element with a *key* equivalent to *k* and returns an iterator to it if found, otherwise it returns an iterator to [map::end](http://www.cplusplus.com/map::end).

     2. Two *keys* are considered equivalent if the container's [comparison object](http://www.cplusplus.com/map::key_comp) returns `false` reflexively (i.e., no matter the order in which the elements are passed as arguments).

     下面的方法其实可以用`count`来判断一个`keys`在不在`map`中：

     * `count`方法：

       ```c++
       size_type count (const key_type& k) const;
       ```

       1. Searches the container for elements with a key equivalent to *k* and returns the number of matches.
       2. Two *keys* are considered equivalent if the container's [comparison object](http://www.cplusplus.com/map::key_comp) returns `false` reflexively (i.e., no matter the order in which the keys are passed as arguments).
       3. 返回值：`1` if the container contains an element whose key is equivalent to *k*, or zero otherwise.

       

```c++
int main ()
{
    map <int, int> table;
    using pair_type = decltype(table)::value_type;
    vector<int> arr = {5,10};
    for(auto i : arr)
        table[i] = i * 10;
    table[1] = 100;
    for (auto it = table.begin(); it != table.end(); it++)
        cout << it->first << " " << it->second << endl;
    cout << "------" << endl;
    // 内部传递了一个对value做比较的匿名函数
    auto pr = max_element(begin(table), end(table),
                          [](const pair_type &p1, const pair_type &p2){return p1.second < p2.second;});
    cout << pr->first << " " << pr->second << endl;
    cout << "------" << endl;
    auto it = table.find(5);
    cout << "find it : key is "<< it->first << " value is : " << it->second << endl;
    auto if_find = table.count(10);
    cout <<" whether a keys in map : 1 means yes, 0 means no : " << if_find << endl;
}

```

output：

```c++
1 100
5 50
10 100
------
1 100
------
find it : key is 5 value is : 50
 whether a keys in map : 1 means yes, 0 means no : 1
```

