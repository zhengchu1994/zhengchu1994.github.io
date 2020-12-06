---
title: CookingC++
mathjax: true
date: 2019-09-15 21:53:32
tags: Cooking
categories: C++
visible:
---

#### 把一个array拷贝到vector

其中的copy方法里，`copy(&dataArray[0], &dataArray[dataArraySize], back_inserter(dataVec));` 不会拷贝`dataArray[dataArraySize]` 元素；例子如下：

```c++
int arr[] = {1,4,2,4,56,5,43,2,2,5};
int main()
{
    int size = sizeof(arr) / sizeof(int);
    vector<int> table;
    copy(&arr[0], &arr[3], back_inserter(table));
    for (auto  i : table)
        cout << i << " ";
    cout << endl;
}
//打印到arr[2]处为止；
$main
1 4 2 
```



```c++
vector<int> dataVec;

int dataArray[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
unsigned dataArraySize = sizeof(dataArray) / sizeof(int);

// Method 1: Copy the array to the vector using back_inserter.
{
    copy(&dataArray[0], &dataArray[dataArraySize], back_inserter(dataVec));
}

// Method 2: Same as 1 but pre-extend the vector by the size of the array using reserve
{
    dataVec.reserve(dataVec.size() + dataArraySize);
    copy(&dataArray[0], &dataArray[dataArraySize], back_inserter(dataVec));
}

// Method 3: Memcpy
{
    dataVec.resize(dataVec.size() + dataArraySize);
    memcpy(&dataVec[dataVec.size() - dataArraySize], &dataArray[0], dataArraySize * sizeof(int));
}

// Method 4: vector::insert
{
    dataVec.insert(dataVec.end(), &dataArray[0], &dataArray[dataArraySize]);
}

// Method 5: vector + vector
{
    vector<int> dataVec2(&dataArray[0], &dataArray[dataArraySize]);
    dataVec.insert(dataVec.end(), dataVec2.begin(), dataVec2.end());
}
```



#### 按符号切割字符串



```c++
std::string s = "scott>=tiger";
std::string delimiter = ">=";
std::string token = s.substr(0, s.find(delimiter)); // token is "scott"
```

* The `find(const string& str, size_t pos = 0)` function returns the position of the first occurrence of `str` in the string, or [`npos`](http://en.cppreference.com/w/cpp/string/basic_string/npos) if the string is not found.
* The `substr(size_t pos = 0, size_t n = npos)` function returns a substring of the object, starting at position `pos` and of length `npos`.

#### 删除字符串中的空白符号

```c++
s.erase(remove_if(s.begin(), s.end(),::isspace), s.end());
```

