---
title: 剑指offer--c++
mathjax: true
date: 2019-09-04 21:53:32
tags: Cooking
categories: C++
visible:
---

#### 队列和栈

1. 用两个栈实现队列的头部出队，尾部入队操作：
```c++
class Solution
{
public:
    void push(int node) {
        while(!stack2.empty())
        {
            stack1.push(stack2.top());
            stack2.pop();
        }
        stack1.push(node);
    }
    int pop() {
        while(!stack1.empty())
        {
            stack2.push(stack1.top());
            stack1.pop();
        }
        int ans = stack2.top();
        stack2.pop();
        return ans;
    }

private:
    stack<int> stack1;
    stack<int> stack2;
};
```

30、包含min函数的栈

>定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数（时间复杂度应为$O(1)$）。
* 解题思路：复杂度是$O(1)$，则不是重装栈，trick是辅助栈进展方式为遇到最小就装它，否则装入当前栈顶元素。
```c++
class Solution {
public:
    void push(int value) {
        st1.push(value);
        if(st2.empty())
            st2.push(value);
        else
        {
            int top = st2.top();
            if(top>value)
                st2.push(value);
            else
                st2.push(top);
        }
    }
    void pop() {
        st1.pop();
        st2.pop();
    }
    int top() {
        return st1.top();
    }
    int min() {
        return st2.top();
    }
private:
    stack<int> st1;
    stack<int> st2;
};
```

31、栈的压入和弹出
>输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如序列1,2,3,4,5是某栈的压入顺序，序列4,5,3,2,1是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的弹出序列。（注意：这两个序列的长度是相等的）

* 解：比较进栈数组中对每一个进栈的数与出栈数组的第一个元素，相等则对该元素出栈，继续比较进栈数组出栈后的栈顶与出栈数组第二个元素，相等则弹出，否则，继续进栈。若是最后进栈数组为空，则结果匹配正确。
```c++
class Solution {
public:
    bool IsPopOrder(vector<int> pushV,vector<int> popV) {
        if(pushV.empty() && popV.empty()) return true;
        if(pushV.empty() || popV.empty() || pushV.size()!=popV.size()) return false;
        
        stack<int> st; 
        for(int i=0,j=0; i<pushV.size();++i)
        {
            st.push(pushV[i]);
            //未越界且相等
            while(j<popV.size() && popV[j]==st.top())
            {
                st.pop();
                j++;
            }
        }
        if(st.empty()) 
            return true;
        else 
            return false;
    }
};
```

59、队列最大的值
>给定一个数组和滑动窗口的大小，找出所有滑动窗口里数值的最大值。例如，如果输入数组{2,3,4,2,6,2,5,1}及滑动窗口的大小3，那么一共存在6个滑动窗口，他们的最大值分别为{4,4,6,6,6,5}； 针对数组{2,3,4,2,6,2,5,1}的滑动窗口有以下6个： {[2,3,4],2,6,2,5,1}， {2,[3,4,2],6,2,5,1}， {2,3,[4,2,6],2,5,1}， {2,3,4,[2,6,2],5,1}， {2,3,4,2,[6,2,5],1}， {2,3,4,2,6,[2,5,1]}

1.直接算最大再排除第一个：这个算法总的时间复杂度是O(nk)
```c++
class Solution {
public:
    vector<int> maxInWindows(const vector<int>& num, unsigned int size)
    {
        vector<int> ans;
        deque<int> index;
        if(num.empty() || size == 1) return num;
        if(size<=0) return vector<int>();  
        int count = 0; //记录滑动窗口大小是否达到
        for(int i=0; i<num.size();++i)
        {
            index.push_back(num[i]);
            count++;
            if(count==size)
            {
                count -= 1;
                ans.push_back(*max_element(index.begin(), index.end()));
                index.pop_front();
            }
        }
        return ans;
    }
};
```

2.
 用双向队列实现，主要是理解思路。
从后删除的情况：只有当 当前数字比队列的后面数字大时。
从前删除的情况： 只有 当 队列前面的数字的序号不在滑动窗口内。
```c++
class Solution {
public:
    vector<int> maxInWindows(const vector<int>& num, unsigned int size)
    {
        vector<int> res;
        deque<int> s;
        for(unsigned int i=0;i<num.size();++i){
            while(s.size() && num[s.back()]<=num[i])//从后面依次弹出队列中比当前num值小的元素，同时也能保证队列首元素为当前窗口最大值下标,同时排出多个
                s.pop_back();
            //这里的while改为if也是可行的，因为只可能超出一个
            while(s.size() && i-s.front()+1>size)//当当前窗口移出队首元素所在的位置，即队首元素坐标对应的num不在窗口中，需要弹出
                s.pop_front();
            s.push_back(i);//把每次滑动的num下标加入队列
            if(size&&i+1>=size)//当滑动窗口首地址i大于等于size时才开始写入窗口最大值
                res.push_back(num[s.front()]);
        }
        return res;
    }
};
```

#### 数组和数值



1. 下面代码的输出是什么

```c++
int GetSize(int data[])
{
    return sizeof(data);
}
int main()
{

    int data1[]= {1,2,3,4,5};
    int size1 = sizeof(data1);
    int *data2 = data1;
    int size2 = sizeof(data2);
    int size3 = GetSize(data1);
    cout <<size1 << " " << size2 << "  " << size3 << endl;
}

20 8  8
```

因为`sizeof(data1)`是求数组的大小，每个整数占4个字节；
第二个是因为指针占8个字节；
第三个是因为数组作为函数参数进行传递，数组退化为指针，也为8个字节。

2. 面试题：找出数组中任意一个重复的数字，数组满足：长度为n的数组里所有数字都在0~n-1的范围内。

- 思路：若不重复，每个下标对应一个等于下标值的数。对下标`i`与`m=arr[i]`作比较，不相等交换`arr[i]`与`arr[m]`
- 时间复杂度$O(n)$，空间复杂度$O(1)$。

```c++
#include <iostream>
#include<vector>
#include<algorithm>
using namespace std;

bool duplicate(vector<int> &arr, int &answer)
{
  for(auto i:arr)
    if(i<0 ||i>arr.size()-1)
      return false;
  for(auto i=0; i<arr.size();++i)
  {
    while(arr[i]!=i)
    {
      if(arr[i]==arr[arr[i]])
      {
        answer = arr[i];
        return true;
      }
      else
      {
        swap(arr[i], arr[arr[i]]);
      }
    }
  }
  return false;
}
int main() {
  int answer;
  vector<int> t={2,3,1,0,2,5,3};
  if(duplicate(t, answer))
    cout << "one duplicate number is: " << answer << endl;
  else
    cout << "no duplicate number ." << endl;
}

one duplicate number is: 2
```







1. 不修改数组找出重复的数字，数组满足：长度为n+1的数组里所有的数字都在1~n范围内，因此至少有一个数字是重复的。找出任意一个数字。

- 思路：辅助数组的方法$O(n)$时间复杂度，$O(n)$空间复杂度，空间换时间；折半查找的方法$O(n\lg n)$时间复杂度,$O(1)$空间复杂度，时间换空间。
- 排查数组哪一半的数字有重复，遍历数组所有数，计数每个数是否在下标范围，总计数值超过的这一半的大小的话，有重复元素；

```c++
#include <iostream>
#include<vector>
#include<algorithm>
using namespace std;


int countRage(const vector<int> &arr,const int &start,const int &end)
{
  int count = 0;
  for(auto iter:arr)
  {
    if(start<=iter && iter<=end)
      ++count;
  }
  return count;
}
int getDuplication(const vector<int> &arr)
{
  int start =1;
  int end = arr.size()-1;
  //直到二分数组大小为1为止
  while(end>=start)
  {
    int middle=((end-start)>>1) + start;
    int count=countRage(arr, start, middle);
    //大小为1
    if (end==start)
    {
      //单个数字的计数是否重复
      if(count>1)
        return start;
      else
        break;
    }
    
    if(count > middle-start+1) //是否在左侧
      end = middle;
    else
      start = middle+1;
  }
  return -1;
}

int main() {
 
  vector<int> t={2,3,5,4,3,2,6,7};
  int answer=getDuplication(t);
  if(answer>=0)
    cout << "one duplicate number is: " << answer << endl;
  else
    cout << "no duplicate number ." << endl;
}
```

方法2：

```c++
class Solution {
public:
    // Parameters:
    //        numbers:     an array of integers
    //        length:      the length of array numbers
    //        duplication: (Output) the duplicated number in the array number
    // Return value:       true if the input is valid, and there are some duplications in the array number
    //                     otherwise false
bool duplicate(int numbers[], int length, int* duplication) {
    for(int i=0;i!=length;++i){
        int index=numbers[i]%length;
        if(numbers[index]>=length){
            *duplication=index;
            return 1;
        }              
        numbers[index]+=length;  
    }
    return 0;
}
};
```

4. 有规律的二维数组中找出某一个数：数组满足每行数大小递增，每一列大小递增。

- 从第一行最后一列排除：

```c++
//number 是要找的数
bool Find(vector<vector<int>> &mat,const int &rows, const int &columns, const int &number)
{
  bool find = false;
  if (rows>0 && columns>0)
  {
    int row=0, column=columns-1;
    while(row<rows && column >=0)
    {
      if(mat[row][column] == number)
      {
        find = true;
        break;
      }
      else if(mat[row][column] > number)
        --column;
      else
        ++row;
    }
  }
  return find;
}

int main() {
 
  vector<vector<int>> test={{1,2,8,9},{2,4,9,12},{4,7,10,13},{6,8,11,15}};
  int ans=7;
  int answer=Find(test,0, 4, ans);
  if(answer>=0)
    cout << "find answer: " << ans << endl;
  else
    cout << "no answer ." << endl;
}
```

16. 数值的整数次方：给定一个`double`类型的浮点数`base`和`int`类型的整数`exponent`。求`base`的`exponent`次方。

- 解：
  利用了公式：

$$
a^n =
\begin{cases}
a^{n/2} * a^{n/2}, & n 为偶数\\
a^{(n-1)/2} * a^{(n-1)/2}*a, & n 为奇数
\end{cases}
$$

实现的时候先对目标$target$的一半运算出来，然后判断是否是奇数。

```c++
class Solution {
public:
    double Power(double base, int exponent) {
        // 直接返回
        if(base==0)
            return 0;
        //对数的绝对值
        unsigned int absExponent=(unsigned int)(abs(exponent));
        double ans = _power(base, absExponent);
        if(exponent<0)
            ans = 1.0/ans;
        return ans;
    }
private:
    //避免复杂度高，利用公式递归求解；
    double _power(int base, unsigned int absExponent)
    {
        if(absExponent==0)
            return 1;
        if(absExponent==1)
            return base;
        //递归公式
        double result=_power(base, absExponent>>1);
        //还剩一半的部分
        result *= result;
        //若为奇数值，乘上少乘的一个基底
        if(absExponent&0x1 ==1)
            result*=base;
        return result;
    }
};
```

21. 调整数组顺序使得奇数位于偶数前面

> 输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。

- 解：
  **相对位置不变--->保持稳定性；奇数位于前面，偶数位于后面 --->存在判断，挪动元素位置；** 这些都和内部排序算法相似，考虑到具有稳定性的排序算法不多，例如插入排序，归并排序等；

```c++
class Solution {
public:
    static bool isOk(int n){return (n&0x1)==1;}
    //stable_partition 这个函数函数功能是将数组中　isOk为真的放在数组前，假的放在数组后，和题意相符
    //stable_partition函数源码其实是开辟了新的空间，然后把符合条件的放在新空间的前面，其他的放在后面。
    void reOrderArray(vector<int> &array) {
        stable_partition(array.begin(),array.end(),isOk);
    }

};
```



29、顺时针打印矩阵

> 输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，例如，如果输入如下4 X 4矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.

- 解：需要分析它的边界：
  打印四个边，什么时候退出：行、列数目大于2倍起点的时候；起点是每次的左上角位置；

打印第一步：因为打印一圈至少有一步，所以直接打印；
打印第二步：行号大于起点行号才能多处行往下打印；
打印第三步：考虑这时候不能为一行一列的打印，至少是两行两列；
打印第四步：考虑已经少了一行数目；

```c++
class Solution {
public:
    vector<int> printMatrix(vector<vector<int> > matrix) {
        vector<int> ans;
        if(matrix.empty()) return ans;
        int rows=matrix.size();
        int cols=matrix[0].size();
        int position = 0;
        //退出条件是两倍位置大小
        while(rows> 2*position && cols > 2*position)
        {
            print(matrix,position,rows,cols, ans);
            ++position;
        }
        return ans;
    }
private:
    void print(const vector<vector<int>> &mat,int position, int rows, int cols,vector<int>& ans)
    {
        rows = rows-position-1;
        cols = cols-position-1;
        
        //case1:多余的列存在
        for(int i=position; i<=cols; ++i)
        {
            ans.push_back(mat[position][i]);
        }
        //case2:多余的行存在
        if(position<rows)
        {
            for(int i=position+1;i<=rows;++i)
            {
                ans.push_back(mat[i][cols]);
            }
        }
        //case3:从右到左打印;排除一行一列的打印
        if(position<rows && position<cols)
        {
            for(int i=cols-1;i>=position;--i)
            {
                ans.push_back(mat[rows][i]);
            }
        }
        //case4:从下到上打印
        if(position+1<rows && position<cols)
        {
            for(int i=rows-1;i>=position+1;--i)
                ans.push_back(mat[i][position]);
        }     
    }
};
```

39、数组中出现次数超过一半的数字

> 数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。

- 解：    两种方法实现：1,快排分割，元素被放置在中间位置，则找到结果；2.设置每次遇到的该数为result，并计数1，若下次遇到的
  是该数，计数增加，若不是，计数减少，当计数为0，设result为新的该数字，最后被保存的result肯定是超过一半的数字

```c++
class Solution {
public:
    int MoreThanHalfNum_Solution(vector<int> numbers) {
        if(numbers.empty()) return 0;
        int result = numbers[0];
        int count = 1;
        for(int i=1; i<numbers.size();++i)
        {
            if(numbers[i] == result)
            {
                count++;
            }
            else
            {
                count--;
                if (count==0)
                {
                    result = numbers[i];
                    count=1;
                }
            }
        }
        //检查该数组是否是满足题意的数组；
        count=0;
        for(int i=0; i<numbers.size(); ++i)
        {
            if(numbers[i]==result)
                count++;
        }
        if(count*2<=numbers.size())
            return false;
        return result;
    }
};

```

方法1：

```c++
class Solution {
public:
    int MoreThanHalfNum_Solution(vector<int> numbers) {
        if(numbers.empty()) return 0;
        int middle = numbers.size()>>1;
        int start = 0;
        int end = numbers.size()-1;
        int index = partition(start, end, numbers);
        while(index!=middle)
        {
            //在下标小于中位数，它的middle应该在右边
            if(index<middle)
            {
                start = index+1;
                index = partition(start,end, numbers);
            }
            else
            {
                end = index-1;
                index = partition(start, end, numbers);
            }
        }
        //检查是否有是合格的组数
        int result = numbers[index];
        int count = 0;
        for(int i=0; i<numbers.size(); ++i)
        {
            if(numbers[i] == result)
                count++;
        }
        if(count*2<=numbers.size())
            return false;
        return result;
    }
private:
    //返回元素的中间位置
    int partition(int start,int end,vector<int>& arr)
    {
        int pivot = arr[start];
        size_t position = start;//记录哨兵最后放置的位置
        for(int i=start+1; i<= end; ++i)
        {
            if(arr[i]<pivot)//只放置小的在左边就行
            {
                position++;//遇到一个小的元素，往前走一步做交换
                if(i!=position)//头元素是povit，这个位置最后交换
                    swap(arr[i], arr[position]);
            }
        }
        swap(arr[position], arr[start]);
        return position;
    }
};

```

40、最小的k个数

> 输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。

- 解析：第一种一般可想到是堆排序，且不会修改输入数据，适合在处理海量数据中进行查找，STL中的`set`与`multiset`的底层是红黑树实现的，可以满足在$O(\lg n)$时间内完成查找、删除、插入。第二种方法是partition.

方法1：

```c++
class Solution {
public:
    vector<int> GetLeastNumbers_Solution(vector<int> input, int k) {
        int len=input.size();
        if(len<=0 || k>len || k<=0) return vector<int>();
        vector<int> ans;
        for(int i=0;i<k;++i)//先插入前main的k个后建立大顶堆
            ans.push_back(input[i]);
        //建堆
        make_heap(ans.begin(), ans.end());
        for(int i=k; i<input.size(); ++i)
        {
            if(input[i]<ans[0]) //比堆顶元素还大，那么替换该元素放入ans，然后维持堆性质
            {
                pop_heap(ans.begin(), ans.end()); //最大元素放到末尾；
                ans.pop_back();//弹出最大的元素
                ans.push_back(input[i]); 
                push_heap(ans.begin(), ans.end());//维持堆性质
            }     
        }
        //使其从小到大输出
        sort_heap(ans.begin(),ans.end());
        return ans;
    }
};

```

方法2：

```c++
class Solution {
public:
    vector<int> GetLeastNumbers_Solution(vector<int> input, int k) {
        int len=input.size();
        if(len<=0||k>len || k<=0) return vector<int>();
         
        //仿函数中的greater<T>模板，从大到小排序
        multiset<int, greater<int> > leastNums;
        vector<int>::iterator vec_it=input.begin();
        for(;vec_it!=input.end();vec_it++)
        {
            //将前k个元素插入集合
            if(leastNums.size()<k)
                leastNums.insert(*vec_it);
            else
            {
                //第一个元素是最大值
                multiset<int, greater<int> >::iterator greatest_it=leastNums.begin();
                //如果后续元素<第一个元素，删除第一个，加入当前元素
                if(*vec_it<*(leastNums.begin()))
                {
                    leastNums.erase(greatest_it);
                    leastNums.insert(*vec_it);
                }
            }
        }
         
        return vector<int>(leastNums.begin(),leastNums.end());
    }
};

```

41、数据流中的中位数

> 如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。我们使用Insert()方法读取数据流，使用GetMedian()方法获取当前读取数据的中位数。

- 解析：    假设整个数据容器分割为两部分，位于容器左边的部分比右边的部分小；容器左数组的最右边指向该部分最大的数；同样， 容器右数组的最左边指向该部分最小的数；
  具体实现：左边用最大堆，右边用最小堆；首先保证数据平均分配到两个堆中，因此这两个堆中的数据数目之差不超过1，用偶数的数字都插入最小堆；保证最大堆中的数都小于最小堆，当前插入数字小于最大堆堆顶元素，可以把该堆的堆顶元素排除，插入到最小堆，然后把该数字插入最大堆中.

```c++
class Solution {
private:
        vector<int> min;
        vector<int> max;
public:
        //第0个元素先插入到min中，之后第1个元素插入到max中.若元素有5个，那么0,2,4被插入到min中，可以看到
        //若元素有6个,0,2，4，被插入到min中，1,3,5被插入到max中。
        //因此min只用的元素大于等于max
        //元素为奇数时，返回min的元素
        void Insert(int num)
        {
           if(((min.size()+max.size())&1)==0)//偶数时 ，放入最小堆
           {
              if(max.size()>0 && num<max[0])
              {
                // push_heap (_First, _Last),要先在容器中加入数据，再调用push_heap ()
                 max.push_back(num);//先将元素压入容器
                 push_heap(max.begin(),max.end(),less<int>());//调整最大堆
                 num=max[0];//取出最大堆的最大值
                 //pop_heap(_First, _Last)，要先调用pop_heap()再在容器中删除数据
                 pop_heap(max.begin(),max.end(),less<int>());//删除最大堆的最大值
                 max.pop_back(); //在容器中删除
              }
              min.push_back(num);//压入最小堆
              push_heap(min.begin(),min.end(),greater<int>());//调整最小堆
           }
           else//奇数时候，放入最大堆
           {
              if(min.size()>0 && num>min[0])
              {
                // push_heap (_First, _Last),要先在容器中加入数据，再调用push_heap ()
                 min.push_back(num);//先压入最小堆
                 push_heap(min.begin(),min.end(),greater<int>());//调整最小堆
                 num=min[0];//得到最小堆的最小值（堆顶）
                 //pop_heap(_First, _Last)，要先调用pop_heap()再在容器中删除数据
                 pop_heap(min.begin(),min.end(),greater<int>());//删除最小堆的最大值
                 min.pop_back(); //在容器中删除
              }
              max.push_back(num);//压入数字
              push_heap(max.begin(),max.end(),less<int>());//调整最大堆
           }   
        }
        /*获取中位数*/      
        double GetMedian()
        {
            int size=min.size()+max.size();
            if(size<=0) //没有元素，抛出异常
                return 0;//throw exception("No numbers are available");
            if((size&1)==0)//偶数时，去平均
                return ((double)(max[0]+min[0])/2);
            else//奇数，去最小堆，因为最小堆数据保持和最大堆一样多，或者比最大堆多1个
                return min[0];
        }
};

```

42、连续子数组的最大和

> HZ偶尔会拿些专业问题来忽悠那些非计算机专业的同学。今天测试组开完会后,他又发话了:在古老的一维模式识别中,常常需要计算连续子向量的最大和,当向量全为正数的时候,问题很好解决。但是,如果向量中包含负数,是否应该包含某个负数,并期望旁边的正数会弥补它呢？例如:{6,-3,-2,7,-15,1,2,2},连续子向量的最大和为8(从第0个开始,到第3个为止)。给一个数组，返回它的最大连续子序列的和，你会不会被他忽悠住？(子向量的长度至少是1)

- 解析：1。动态规划问题，设在位置$n-1$处结尾的数组最大和为$f(n-1)$,那前面长度为$n$的数组的最大连续子序列的和为：

$$
f(n)=
\begin{cases}
data(n), & n=0 \bigcup f(n-1) < 0 \\
f(n-1) + data(n), & n \neq 0 \bigvee f(n-1) >0
\end{cases}
$$

可以理解，如果前面的结尾的子串为负数，可以不加；若为正数，可以考虑加上这个值。最后还需要求 $max_{(i=1,2,3,....,N)}f(i)$ 找出最大的和：

```c++
class Solution {
public:
    /*
    dp = max{f(i)},i=0,1,...,N
    f(i) = f(i-1) + data[i] if (i！= 0 && f(i-1)>0) else data[i].
    */
    int FindGreatestSumOfSubArray(vector<int> array) {
        if(array.empty()) return 0;
        vector<int> arr;
        int ans = INT_MIN;
        for(int i=0; i<array.size(); ++i)
        {
           ans = max(ans, dp(i,arr,array));
        }
        return ans;
    }
private:
    //position给出结尾的位置
    //position结尾的最大和存入arr
    //target是原来的数组，用来查看当前的结尾数
    int dp(int position,vector<int>& arr,const vector<int> &target)
    {
        if(position && arr[position-1] > 0)
        {
            int ans = target[position] + arr[position-1];
            arr.push_back(ans);
            return ans;
        }
        else
        {
            arr.push_back(target[position]);
            return target[position];
        }
    }
};

```

2.跟据求和的性质，若为负，丢弃当前和，并记录每一的最大和；

```c++
public:
    int FindGreatestSumOfSubArray(vector<int> array) {
        if(array.empty())
        {
            InvalidInput = true;//判断是否无效输入而非最大和为0
            return 0;
        }
        int nCurSum=0;
        int great =0x80000000;
        for(int i=0; i<array.size();++i)
        {
            if(nCurSum<=0) //若为0或负值，那么略去这个求和，设新的求和为当前数
                nCurSum=array[i]; //
            else
                nCurSum+=array[i];
            if(nCurSum>great)//记录最大子数组和
                great = nCurSum;
        }
        return great;
        
    }
private:
    bool InvalidInput = false;
};

```

43、1~n整数中1出现的次数

> 求出1~13的整数中1出现的次数,并算出100~1300的整数中1出现的次数？为此他特别数了一下1~13中包含1的数字有1、10、11、12、13因此共出现6次,但是对于后面问题他就没辙了。ACMer希望你们帮帮他,并把问题更加普遍化,可以很快的求出任意非负整数区间中1出现的次数（从1 到 n 中1出现的次数）。

- 解析：    方法1:nlog(n)暴力搜索，对每一个数求一个计数1的函数；
  方法2:log(n) 考虑问题本身的性质；

```c++
//方法1：nlog(n)
class Solution {
public:
    /*
    方法1:nlog(n)暴力搜索，对每一个数求一个计数1的函数；
    方法2:log(n) 考虑问题本身的性质；
    */
    int NumberOf1Between1AndN_Solution(int n)
    {
        int number = 0;
        for(int i=1; i<=n; ++i)
        {
            number += Numberi(i);
        }
        return number;
    }
private:
    int Numberi(int i)
    {
        int number=0;
        while(i)
        {
            if(i%10==1)
                number+=1;
            i=i/10;
        }
        return number;
    }
};

```

方法2：参考第三题：https://www.jianshu.com/p/1582fbaf05f7

> 当 N 为 1 位数时，对于 N>=1，1 的个数 f(N) 为1。
> 当 N 为 2 位数时，则个位上1的个数不仅与个位数有关，还和十位数字有关。
> 比如当 N=23 时，个位上 1 的个数有 1、11、21 共3个，十位上1的个数为10，11...19 共10个，所以 1 的个数 f(N) = 3+10 = 13。
> 看出来有规律一：
> 如果 N 的个位数 >=1，则个位出现1的次数为十位数的数字加1；如果 N 的个位数为0，则个位出现 1 的次数等于十位数的数字。
> 十位数上出现1的次数类似，如果N的十位数字等于1，则十位数上出现1的次数为各位数字加1；如果N的十位数字大于1，则十位数上出现1的次数为10。
> 当 N 为 3 位数时，同样分析可得1的个数。如 N=123，可得 1出现次数 = 13+20+24 = 57。当 N 为 4,5...K 位数时，

我们假设 N=abcde，则要计算百位上出现1的数目，则它受到三个因素影响：百位上的数字，百位以下的数字，百位以上的数字。
如果百位上数字为0，则百位上出现1的次数为更高位数字决定。如 N=12013，则百位出现1的数字有100~199， 1000~1199， 2100~2199...11100~111999 共 1200 个，等于百位的更高位数字(12)*当前位数(100)。
如果百位上数字为1，则百位上出现1的次数不仅受更高位影响，还受低位影响。如12113，则百位出现1的情况共有 1200+114=1314 个，也就是高位影响的 12 * 100 + 低位影响的 113+1 = 114 个。
如果百位上数字为其他数字，则百位上出现1的次数仅由更高位决定。如 12213，则百位出现1的情况为 (12+1)*100=1300。

| N=12013 | i 是百位为0，高位是12现在从低位走到高位，即1~12013中，百位出现的1的数如下： |
| ------- | ------------------------------------------------------------ |
| 1       | 100~199                                                      |
| 2       | 1100~1199                                                    |
| 3       | 2100~1299                                                    |
| ..      | ...........                                                  |
| 10      | 9100~9199                                                    |
| 11      | 10100~10199                                                  |
| 12      | 11100~11199                                                  |

总共有12*100个1出现在百位。结论是位数为0时，只受高位数的影响，为高位数的值 * 当前位 。其他位的分析类似。



有以上分析思路，写出下面的代码。其中 low 表示低位数字，curr 表示当前考虑位的数字，high 表示高位数字。一个简单的分析，考虑数字 123，则首先考虑个位，则 curr 为 3，低位为 0，高位为 12；然后考虑十位，此时 curr 为 2，低位为 3，高位为 1。其他的数字可以以此类推，实现代码如下：

```c++
class Solution {
public:
    int NumberOf1Between1AndN_Solution(int n)
    {
        int count = 0;//1的个数
        int i = 1;//乘积因子
        int current = 0,after = 0,before = 0;
        while((n/i)!= 0){           
            current = (n/i)%10; //当前位数字
            before = n/(i*10); //高位数字
            after = n-(n/i)*i; //低位数字
            //如果为0,出现1的次数由高位决定,等于高位数字 * 当前位数
            if (current == 0)
                count += before*i;
            //如果为1,出现1的次数由高位和低位决定,高位*当前位+低位+1
            else if(current == 1)
                count += before * i + after + 1;
            //如果大于1,出现1的次数由高位决定,//（高位数字+1）* 当前位数
            else{
                count += (before + 1) * i;
            }    
            //前移一位
            i = i*10;
        }
        return count;
    }
};

```

45、把数组排成最小的数

> 输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。

- 解析：数组内的数变为string之后做拼接，按照字符串排序便可，拼接之后字符串AB>BA，那么有B<A,应该把B排前面。

```c++
class Solution {
public:
    //数组内的数变为string之后做拼接排序
    string PrintMinNumber(vector<int> numbers) {
        sort(numbers.begin(), numbers.end(), cmp);
        string ans="";
        for(int i=0; i<int(numbers.size());++i) ans+=to_string(numbers[i]);
        return ans;
    }
private:
    static bool cmp(int a, int b)
    {
        string sa=to_string(a), sb =to_string(b);
        return sa +sb < sb + sa;
    }
};

```

做法2：

```c++
//写的比较骚气的整型到string
string itos(int x){
    return (x > 9 ? itos(x / 10) : "") + char(x % 10 + '0');
}
//比较字符函数
bool cmp(int a, int b){
    return itos(a) + itos(b) < itos(b) + itos(a);
}

class Solution {
public:
    string PrintMinNumber(vector<int> a) {
        sort(a.begin(), a.end(), cmp);
        string s = "";
        //转为字符再追加到s
        for(int i = 0; i < int(a.size()); ++i) s += itos(a[i]);
        return s;
    }
};


```

49、第N个丑数。

> 把只包含质因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，因为它包含质因子7。 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。

- 解析：
  空间换时间，记录存储下丑数，丑数按递增大小寻找；
   当前的丑数已经被找到,那么该丑数之前的数字都是排好序的，
   下一个丑数也是2、3、5的倍数，且大于当前丑数；找到每一个2、3、5倍数里最小的作为下一个丑数
   排序好的丑数中，前面的丑数是后面选出的丑数的因子之一，用下标跟随前面的因子与2、3、5的乘积>大于当前丑数
   丑数定义：1是最小的丑数，只能被2或者3或者5整除的数是丑数;

```c++
class Solution {
public:
    int GetUglyNumber_Solution(int index) {
        if(index<=0) return 0;
        vector<int> uglyVec={1}; 
        //作为下标记录2、3、5因子里前面
        int index2=0, index3=0, index5=0;
        while(uglyVec.size()<index)
        {
            uglyVec.push_back(min(uglyVec[index2]*2, min(uglyVec[index3]*3, uglyVec[index5]*5)));
            if(uglyVec[index2]*2== uglyVec.back()) //等号的意义是丑数不重复
                index2++;
            if(uglyVec[index3]*3 == uglyVec.back())
                index3++;
            if(uglyVec[index5]*5== uglyVec.back())
                index5++;
        }
        return uglyVec.back();
    }
};

```

51、逆序对

> 在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组,求出这个数组中的逆序对的总数P。并将P对1000000007取模的结果输出。 即输出P%1000000007

- 解析：

`/*归并排序的改进，把数据分成前后两个数组(递归分到每个数组仅有一个数据项)，`

`合并数组，合并时，出现前面的数组值array[i]大于后面数组值array[j]时；则前面`

`数组array[i]~array[mid]都是大于array[j]的，count += mid+1 - i`

`参考剑指Offer，但是感觉剑指Offer归并过程少了一步拷贝过程。`

`还有就是测试用例输出结果比较大，对每次返回的count mod(1000000007)求余`

`*/`

```c++
/*参考《剑指offer》，有两种思路。第一就是暴力求解法，时间复杂度为o（n^2）,空间复杂度o(1)
第二种思路就是使用归并排序的思想进行处理，时间复杂度o(nlog(n)),空间复杂度0（n）*/
class Solution {
public:
    int InversePairs(vector<int> data) {
        if(data.size()<=1) return 0;//如果少于等于1个元素，直接返回0
        int* copy=new int[data.size()];
        //初始化该数组，该数组作为存放临时排序的结果，最后要将排序的结果复制到原数组中
        for(unsigned int i=0;i<data.size();i++)
            copy[i]=0;
        //调用递归函数求解结果
        int count=InversePairCore(data,copy,0,data.size()-1);
        delete[] copy;//删除临时数组
        return count;
    }
     //程序的主体函数
    int InversePairCore(vector<int>& data,int*& copy,int start,int end)
    {
        if(start==end)
        {
            copy[start]=data[start];
            return 0;
        }
        //将数组拆分成两部分
        int length=(end-start)/2;//这里使用的下标法，下面要用来计算逆序个数；也可以直接使用mid=（start+end）/2
        //分别计算左边部分和右边部分
        int left=InversePairCore(data,copy,start,start+length)%1000000007;
        int right=InversePairCore(data,copy,start+length+1,end)%1000000007;
        //进行逆序计算
        int i=start+length;//前一个数组的最后一个下标
        int j=end;//后一个数组的下标
        int index=end;//辅助数组下标，从最后一个算起
        int count=0;
        while(i>=start && j>=start+length+1)
        {
            if(data[i]>data[j])
            {
                copy[index--]=data[i--];
                //统计长度
                count+=j-start-length;
                if(count>=1000000007)//数值过大求余
                    count%=1000000007;
            }
            else
            {
                copy[index--]=data[j--];
            }
        }
        for(;i>=start;--i)
        {
            copy[index--]=data[i];
        }
        for(;j>=start+length+1;--j)
        {
            copy[index--]=data[j];
        }
        //排序
        for(int i=start; i<=end; i++) {
            data[i] = copy[i];
        }
        //返回最终的结果
        return (count+left+right)%1000000007;
    }
};

```

56、数组中数字出现的次数

> 一个整型数组里除了两个数字之外，其他的数字都出现了两次。请写程序找出这两个只出现一次的数字。要求时间复杂度$O(1)$,空间复杂度$O(n)$.

- 解析：

61、扑克牌顺子

> LL今天心情特别好,因为他去买了一副扑克牌,发现里面居然有2个大王,2个小王(一副牌原本是54张^_^)...他随机从中抽出了5张牌,想测测自己的手气,看看能不能抽到顺子,如果抽到的话,他决定去买体育彩票,嘿嘿！！“红心A,黑桃3,小王,大王,方片5”,“Oh My God!”不是顺子.....LL不高兴了,他想了想,决定大\小 王可以看成任何数字,并且A看作1,J为11,Q为12,K为13。上面的5张牌就可以变成“1,2,3,4,5”(大小王分别看作2和4),“So Lucky!”。LL决定去买体育彩票啦。 现在,要求你使用这幅牌模拟上面的过程,然后告诉我们LL的运气如何， 如果牌能组成顺子就输出true，否则就输出false。为了方便起见,你可以认为大小王是0

- 解析：/先计算0，然后去掉0的部分，遍历这个数组，看相等的时候直接返回0,不连续的时候，累加间隔大小

```c++
class Solution {
public:
    bool IsContinuous( vector<int> numbers) {
        if(numbers.empty()) return false;
        sort(numbers.begin(), numbers.end());
        int countZero=0,unequal=0; //计数0的个数，计算不等数字的间隔
        for(int i=0;i<numbers.size() && numbers[i]==0;++i)
            countZero++;
        //先计算出0的个数，这些0不在我们的规则内；
        for(int i=countZero+1; i<numbers.size(); ++i)
        {
            if(numbers[i]==numbers[i-1])
                return false;
            if(numbers[i-1]+1 != numbers[i])
                unequal= unequal + numbers[i] - numbers[i-1] - 1;
        }
        if(unequal>countZero)
            return false;
        else
            return true;
    }
};

```

62、约斯夫环

> 每年六一儿童节,牛客都会准备一些小礼物去看望孤儿院的小朋友,今年亦是如此。HF作为牛客的资深元老,自然也准备了一些小游戏。其中,有个游戏是这样的:首先,让小朋友们围成一个大圈。然后,他随机指定一个数m,让编号为0的小朋友开始报数。每次喊到m-1的那个小朋友要出列唱首歌,然后可以在礼品箱中任意的挑选礼物,并且不再回到圈中,从他的下一个小朋友开始,继续0...m-1报数....这样下去....直到剩下最后一个小朋友,可以不用表演,并且拿到牛客名贵的“名侦探柯南”典藏版(名额有限哦!!^_^)。请你试着想下,哪个小朋友会得到这份礼品呢？(注：小朋友的编号是从0到n-1)

- : 约瑟夫环两种解法，1循环链表，2是找规律直接算出结果；
  1.

```c++
class Solution {
public:
    int LastRemaining_Solution(int n, int m)
    {
        if(n<=0) return -1;
        list<int> numbers;
        for(unsigned int i=0; i<n;++i)
            numbers.push_back(i);
        list<int>::iterator iter = numbers.begin();
        while(numbers.size()>1)
        {
            //先游走m步，这里的i=1，因为删除第m个
            for(int i=1; i<m; ++i)
            {
                iter++;
                if(iter==numbers.end())
                    iter = numbers.begin();
            }
            //查看是否是迭代器最后一个位置
            list<int>::iterator next=++iter;
            if(next==numbers.end()) next = numbers.begin();
            --iter;
            numbers.erase(iter);
            iter = next;
        }
        return *(iter);
    }
};

```



66、乘积数组

> 给定一个数组A[0,1,...,n-1],请构建一个数组B[0,1,...,n-1],其中B中的元素B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]。不能使用除法。

- 解析：分析递归式如图：
  剑指的思路：

**B[i]的值可以看作下图的矩阵中每行的乘积。**

下三角用连乘可以很容求得，上三角，从下向上也是连乘。

因此我们的思路就很清晰了，先算下三角中的连乘，即我们先算出B[i]中的一部分，然后倒过来按上三角中的分布规律，把另一部分也乘进去。

![841505_1472459965615_8640A8F86FB2AB3117629E2456D8C652.jpg](https://upload-images.jianshu.io/upload_images/5433630-06758a7c1e061db4.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



```c++
//B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]
//从左到右算 B[i]=A[0]*A[1]*...*A[i-1]
//从右到左算B[i]*=A[i+1]*...*A[n-1]
class Solution {
public:
    vector<int> multiply(const vector<int>& A) {
     
        int n=A.size();
        vector<int> b(n);
        int ret=1;
        for(int i=0;i<n;ret*=A[i++]){
            b[i]=ret;
        }
        ret=1;
        for(int i=n-1;i>=0;ret*=A[i--]){
            b[i]*=ret;
        }
        return b;
    }
};
```

64、

> 

- 利用短路原理：

```c++
class Solution {
public:
    int Sum_Solution(int n) {
        int ans = n;
        ans && (ans += Sum_Solution(n - 1));
        return ans;
    }
};
```

65、不用加减乘除法做加法

> 写一个函数，求两个整数之和，要求在函数体内不得使用+、-、*、/四则运算符号。

- 分析：考虑位运算，先相加，不进位，对进位的位置再做相加。

```c++
class Solution {
public:
    int Add(int num1, int num2)
    {
        int sum, carry;
        do
        {
            sum = num1 ^ num2; // 异或运算：0+1、1+0都是1，其他都是0；
            carry = (num1 & num2) << 1; //与运算，只有1+1才为1，然后左移1位；
            num1 = sum;    // 第三部求和是一个循环过程，直到没有进位，也就是num2为0
            num2 = carry;  // 下一次循环中更新sum
        }
        while(num2!=0);
        return num1;
        
    }
};
```





###  树



#### 94. 二叉树的非递归前序遍历

1.  直接进出入栈，拿出数据：
```c
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<int> preorderTraversal(TreeNode* root) {
        vector<int> ans;
        stack<TreeNode*> st;
        if (!root)
            return ans;
        
        st.push(root);
        while(!st.empty())
        {
            TreeNode *top = st.top();
            ans.push_back(top->val);
            st.pop();
            if(top->right)
                st.push(top->right);
            if(top->left)
                st.push(top->left);
        }
        return ans;
    }
};
```
#### 94. 二叉树的非递归中序遍历

1. 
首先是根非空入栈，对子树最左边节点一直进栈；
直到左为空，重置根为栈顶节点，取出数据并出栈；
重置根为根的右边节点；
```
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    //think:首先是根入栈，左边不为空则进栈，直到左为空出栈，然后右边不为空进栈
    vector<int> inorderTraversal(TreeNode* root) {
        std::stack<TreeNode *> Stack;
        vector<int> ans;
        while(root || !Stack.empty())
        {
            if (root != nullptr)
                Stack.push(root);
            while(root && root->left != nullptr) //先验证根是否存在；再验证左节点
            {
                Stack.push(root->left);
                root = root->left;
            }
            root = Stack.top();
            ans.push_back(root->val);
            Stack.pop();
            root = root->right;
        }
        return ans;
    }
};
```

#### 145. 二叉树的非递归后序遍历
* 跟前序遍历很像：第一次访问根节点，用于确定是否还有属于它的子节点需要进栈，若有就进栈，这是第一次遍历该节点；
不同的是：在第二次遍历时出栈。栈顶pop一个新的节点时，是第二次遍历，第二次遍历时出栈。

1. 进两次栈的方法：
* 初始化：根非空进栈两次；
* 循环：栈非空的话：
       * 设置当前出栈元素cur，栈顶出栈；（第一次遍历）
        *  当前出栈元素等于栈顶元素（确保为第一次遍历）的话：若右非空，进栈两次，若左非空，进栈两次；否则的话（第二次遍历）：取出当前cur的值

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<int> postorderTraversal(TreeNode* root) {
        vector<int> ans;
        if(!root) return ans;     
        stack<TreeNode*> Stack;
        TreeNode *cur;
        Stack.push(root);
        Stack.push(root);
        while(!Stack.empty())
        {  
            cur = Stack.top();
            Stack.pop();
            if(!Stack.empty() && cur==Stack.top()) //因为之前出栈，先看是否为空，然后确定目前的栈顶部节点是否访问过，如果cur与栈顶不一样，则已经访问过一次，目前这个可以出栈
            {
                if(cur->right) //后序遍历先进右
                {
                    Stack.push(cur->right);
                    Stack.push(cur->right);
                }
                if (cur->left)//后进左、后进先出
                {
                    Stack.push(cur->left);
                    Stack.push(cur->left);
                }
            }else{
                ans.push_back(cur->val);
            } 
        }
        return ans;
    }
};
```


2.   考虑根节点与栈顶节点，
* 循环：若`root`非空或者栈非空：
      * 若根非空，进栈，设置根为根的左节点；
      * 一直入栈最左的子节点（作为新根节点），直到根为空；
      * 设置一个节点记录栈顶节点，若记录节点top有右节点，且最后出栈的节点不等于记录节点或栈顶节点，根节点重置为栈顶节点的右边节点；
      * 否则栈顶节点无右节点，取值，出栈，重置最后出栈的节点；
```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<int> postorderTraversal(TreeNode* root) {
        vector<int> ans;
        if(!root) return ans;     
        stack<TreeNode*> Stack;
        TreeNode *last=nullptr;
        while(root || !Stack.empty())
        {
            if(root)
            {
                Stack.push(root);
                root = root->left;
            }else{
                //
                TreeNode* top = Stack.top();
                if (top->right && last!=top->right)
                {
                    root = top->right;
                }
                else
                {
                    ans.push_back(top->val);
                    last = top;
                    Stack.pop();
                }
            }
        }
        return ans;
    }
};
```

3.   用前序遍历，最后反转数组；
     pre-order traversal is root-left-right, and post order is left-right-root. modify the code for pre-order to make it root-right-left, and then reverse the output so that we can get left-right-root 
```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<int> postorderTraversal(TreeNode* root) {
        vector<int> ans;
        if(!root) return ans;     
        stack<TreeNode*> Stack;
        TreeNode *tmp;
        Stack.push(root);
        //前左右；左右前=前右左的反转
        while(!Stack.empty())
        {
            tmp = Stack.top();
            ans.push_back(tmp->val);
            Stack.pop();
            if(tmp->left)
                Stack.push(tmp->left);
            if(tmp->right)
                Stack.push(tmp->right);
        }
        vector<int> rans(ans.rbegin(), ans.rend());
        return rans;
    }
};
```


####  98. 验证二叉搜索树
* 主要是根节点大于所有的左子树，小于所有的右子树；怎么样把这个根部的节点信息传递下去。

* 方法1：递归，除了比较根与左右节点，还要比较左右节点与上一次的根节点值是否满足大小关系；
用min判断传入为（当前根节点的）左节点时，比较右节点是否小于上一次根节点值；
用max判断传入为（当前根节点的）右节点时，比较左节点是否大于上一次根节点值；
```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    //Thinking: 全部的子树上的节点都必须大于或者小于根节点才行
    //传入子树递归的时候带上根节点的值，进行比较
    
    bool isValidBST(TreeNode* root) 
    {
        return _isValidBST(root, LONG_MIN, LONG_MAX);
    }
    bool _isValidBST(TreeNode *root,long min, long max)
    {
        if(!root) return true;
        if(root->val<=min || root->val>=max) return false;
        return (_isValidBST(root->left,min,root->val) && _isValidBST(root->right,root->val,max));
    }
};
```



#### 1. 重建二叉树：根据前序和中序遍历的结果重建二叉树。假设输入的遍历结果都不含有重复的数字。
```c++
 * Definition for binary tree
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* reConstructBinaryTree(vector<int> pre,vector<int> vin) {
        if(pre.size() == 0){                    //如果为空，返回NULL
            return NULL;
        }
        //依次是前序遍历左子树，前序遍历右子树，中序遍历左子树，中序遍历右子树
        vector<int> left_pre, right_pre, left_vin, right_vin;
        //前序遍历第一个节点一定为根节点
        TreeNode* head = new TreeNode(pre[0]);
        //找到中序遍历的根节点
        int root = 0;
        //遍历找到中序遍历根节点索引值
        for(int i = 0; i < pre.size(); i++){
            if(pre[0] == vin[i]){
                root = i;
                break;
            }
        }
           //利用中序遍历的根节点，对二叉树节点进行归并
        for(int i = 0; i < root; i++){
            left_vin.push_back(vin[i]);
            left_pre.push_back(pre[i + 1]);            //前序遍历第一个为根节点
        }
        
        for(int i = root + 1; i < pre.size(); i++){
            right_vin.push_back(vin[i]);
            right_pre.push_back(pre[i]);
        }
        
        //递归，再对其进行上述所有步骤，即再区分子树的左、右子子数，直到叶节点
        head->left = reConstructBinaryTree(left_pre, left_vin);
        head->right = reConstructBinaryTree(right_pre, right_vin);
        return head;
    }
};
```

#### 2. 二叉树的下一个节点：给定一颗二叉树和其中的一个节点，如何找到中序遍历序列的下一个节点，每个节点左右子节点指针，还有指向父节点的指针。

```c++
/*
struct TreeLinkNode {
    int val;
    struct TreeLinkNode *left;
    struct TreeLinkNode *right;
    struct TreeLinkNode *next;
    TreeLinkNode(int x) :val(x), left(NULL), right(NULL), next(NULL) {
        
    }
};
*/
class Solution {
public:
    TreeLinkNode* GetNext(TreeLinkNode* pNode)
    {
        if(!pNode) return nullptr;
        TreeLinkNode* pNext=nullptr;
        //该节点存在右子树，返回右子树的最左叶节点
        if(pNode->right!=nullptr)
        {
            pNext = pNode->right;
            while(pNext->left)
                pNext=pNext->left;
            return pNext;
         //该节点无右子树
        }else if(pNode->next!=nullptr)
        {
            TreeLinkNode* pCurrent=pNode;
            TreeLinkNode* pParent=pNode->next;
            //是否是某父节点的左节点;不是的情况且该节点是某父节点的右子节点，向上递归，直到为某一个父节点的左子节点
            while(pParent!=nullptr && pCurrent==pParent->right)
            {
                pCurrent = pParent;
                pParent = pCurrent->next;
            }
            pNext = pParent;
        }
        return pNext;
    }
};
```

#### 26. 树的子结构
>输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）

```c++
class Solution {
public:
    bool HasSubtree(TreeNode* pRoot1, TreeNode* pRoot2)
    {
        bool result=false;
        //非空情况
        if(pRoot1!=nullptr && pRoot2!=nullptr)
        {
            if(pRoot1->val==pRoot2->val)
                result=DoesTree1HaveTree2(pRoot1,pRoot2);
            if(!result)//左右子树上递归
                result=HasSubtree(pRoot1->left,pRoot2);
            if(!result)
                result=HasSubtree(pRoot1->right,pRoot2);
        }
        return result;
    }
private:
    bool DoesTree1HaveTree2(TreeNode* pRoot1,TreeNode* pRoot2)
    {
        if(pRoot2==nullptr)//递归完成返回true
            return true;
        if(pRoot1==nullptr)//未匹配
            return false;
        if(pRoot1->val!=pRoot2->val)
            return false;
        return DoesTree1HaveTree2(pRoot1->left,pRoot2->left) && DoesTree1HaveTree2(pRoot1->right,pRoot2->right);
    }
};
```


#### 27、二叉树的镜像
>操作给定的二叉树，将其变换为源二叉树的镜像。

* 思路就是左右都是空，就不用管了；而只有一个子节点为空，另一个不为空，也必须交换才行；

```c++
class Solution {
public:
    void Mirror(TreeNode *pRoot) {
        if(!pRoot) return; //Empty tree
        //leaf node already shifted
        if (pRoot->left==nullptr && pRoot->right==nullptr)
            return;
        //whatever empty , both child nodes do shift
        cout << pRoot;
        TreeNode* tmp=pRoot->left;
        pRoot->left=pRoot->right;
        pRoot->right=tmp;
        //recursion
        if(pRoot->left!=nullptr)
            Mirror(pRoot->left);
        if(pRoot->right!=nullptr) 
            Mirror(pRoot->right);
    }
};
```


#### 28、对称的二叉树
>请实现一个函数，用来判断一颗二叉树是不是对称的。注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的。

* 解题思路：特例：树上的节点值相同的时候还需要比较空节点位置是否对称；
对称的例子如：
>   8
>    6     6
>   5 7 7 5
```c++
class Solution {
public:
    bool isSymmetrical(TreeNode* pRoot)
    {
        return isSymmetrical(pRoot,pRoot);
    }
private:
    bool isSymmetrical(TreeNode* p1, TreeNode* p2)
    {
        if(p1==nullptr && p2==nullptr)
            return true;
        if(p1==nullptr || p2==nullptr)
            return false;
        if(p1->val != p2->val)
            return false;
        return isSymmetrical(p1->left,p2->right) && isSymmetrical(p1->right, p2->left);
    }
};
```

#### 32、从下到上打印二叉树
>从上往下打印出二叉树的每个节点，同层节点从左至右打印。

* 解：层次遍历用队列存储即可。
```c++
class Solution {
public:
    vector<int> PrintFromTopToBottom(TreeNode* root) {
        vector<int> ans;
        if(!root) return ans;
        queue<TreeNode*> que;
        que.push(root);
        while(!que.empty())
        {
            ans.push_back(que.front()->val);
            TreeNode* top=que.front();
            que.pop();
            if(top->left)
                que.push(top->left);
            if(top->right)
                que.push(top->right);
        }
        return ans;
    }
};
```

#### 32.2、分行从上到下打印二叉树
>从上到下按层打印二叉树，同一层结点从左至右输出。每一层输出一行。

* 解：需要记录当前层中还没有打印的节点数、下一层节点的数目；

```c++
class Solution {
public:
        vector<vector<int> > Print(TreeNode* pRoot) {
            vector<vector<int>> ans;
            if(!pRoot) return ans;
            queue<TreeNode*> que;
            que.push(pRoot);
            int nextNodeNum=1;
            while(!que.empty())
            {   
                vector<int> _ans;
                int curNodeNum=nextNodeNum;//当前层的数目为1，下一层的数目节点为0
                nextNodeNum=0;
                while(curNodeNum)
                {
                    TreeNode* top=que.front();
                    _ans.push_back(top->val);
                    if(top->left)
                    {
                        que.push(top->left);
                        nextNodeNum++;
                    }
                    if(top->right)
                    {
                        que.push(top->right);
                        nextNodeNum++;
                    }
                    que.pop();
                    --curNodeNum;
                }
                ans.push_back(_ans);
            }
            return ans;
        }
};
```
#### 33、二叉搜索树的后序遍历序列
>输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。

* 解析：最末的位置存储的是根节点，然后左右子树的边界通过循环可以找到，进一步判断右子树是否都小于根节点，如果满足，下一步看左右子树若存在，左右子树是否递归满足。

```c++
class Solution {
public:
    bool VerifySquenceOfBST(vector<int> sequence) {
        return Help(sequence, sequence.size());
    }
private:
    bool Help(const vector<int> &sequence, int size)
    /*
    Params:size represents the root node position.
    */
    {
        if(sequence.empty()) return false;
        int i=0;
        //找寻左右子树分界节点
        for(;i<size-1;++i)
            if(sequence[i]>sequence[size-1])
                break;
        //右子树是否满足大于根节点
        for(int j=i;j<size;++j)
            if(sequence[j]<sequence[size-1])
                return false;
        //递归判断左子树
        bool left=true;
        if(i>0)//存在左子树才行
            left = Help(sequence, i);
        bool right=true;
        if(i<size-1)
            right=Help(sequence, size-1);
        return left && right;
    }
};
```

#### 34、二叉树中和为某一值的路径
>输入一颗二叉树的根节点和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。(注意: 在返回值的list中，数组长度大的数组靠前)

* 解析：
方法1. 递归、判断成功的时候把成功的路径压入结果ans；但是每次递归path需要重新拷贝使用；
方法2. 

```c++
class Solution {
public:
    vector<vector<int> > FindPath(TreeNode* root,int expectNumber) {
        vector<vector<int>> ans;
        vector<int> path;
        if(!root) return ans;
        _FindPath(root,expectNumber,path,ans);
        return ans;
    }
private:
    //path不能用引用，递归的时候需要拷贝path
    void _FindPath(TreeNode* root,int num, vector<int> path, vector<vector<int>> &ans)
    {
        if(root)
        {
            path.push_back(root->val);
            if(num==root->val && (!root->left) && (!root->right))
            {    
                ans.push_back(path);
                return;
            }
            _FindPath(root->left,num - root->val, path, ans);
            _FindPath(root->right,num - root->val, path, ans);
        }
    }
};
```


#### 36、二叉搜索树与双向链表
>输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。

* 递归，根节点进入，然后向左递归，出来之后，若有右节点，向右走，走到none在指向root，root在指向它；右子树也一样这样。
```c++
class Solution {
public:
    TreeNode* Convert(TreeNode* pRootOfTree)
    {
        if(pRootOfTree==nullptr) return nullptr;
        pRootOfTree=ConvertNode(pRootOfTree);
        while(pRootOfTree->left) pRootOfTree=pRootOfTree->left;
        return pRootOfTree;
    }
private:
    //返回子树的根节点
    TreeNode* ConvertNode(TreeNode* pRootOfTree)
    {
        if(pRootOfTree==nullptr) return nullptr;
        if(pRootOfTree->left!=nullptr)
        {   
            TreeNode* left = ConvertNode(pRootOfTree->left);
            //左子树的根节点往右走、它的下一个节点是当前的根节点
            while(left->right) left=left->right;
            left->right=pRootOfTree;
            pRootOfTree->left=left;
        }
        if(pRootOfTree->right!=nullptr)
        {
            TreeNode* right = ConvertNode(pRootOfTree->right);
            //右子树的根节点往左走、它的上一个节点是目前的根节点；
            while(right->left) right=right->left;
            right->left=pRootOfTree;
            pRootOfTree->right=right;
        }
        return pRootOfTree;
    }
};
```

#### 37、序列化二叉树
>二叉树的序列化是指：把一棵二叉树按照某种遍历方式的结果以某种格式保存为字符串，从而使得内存中建立起来的二叉树可以持久保存。序列化可以基于先序、中序、后序、层序的二叉树遍历方式来进行修改，序列化的结果是一个字符串，序列化时通过 某种符号表示空节点（#），以 ！ 表示一个结点值的结束（value!）。
>二叉树的反序列化是指：根据某种遍历顺序得到的序列化字符串结果str，重构二叉树。

* 不太熟悉char的这些操作，看别人的答案，非得搞指针传入的默认函数。

```c++
class Solution {
public:
    char* Serialize(TreeNode *root) {    
        if(!root) return "#";
        string r = to_string(root->val);
        r.push_back(',');
        char *left = Serialize(root->left);
        char *right = Serialize(root->right);
        char *ret = new char[strlen(left) + strlen(right) + r.size()];
        strcpy(ret, r.c_str());
        strcat(ret, left);
        strcat(ret, right);
        return ret;
    }
    TreeNode* Deserialize(char *str) {
        return decode(str);
    }
private:
    TreeNode* decode(char* &str)
    {
        if(*str=='#')
        {
            str++;
            return nullptr;
        }
        int num=0;
        while(*str != ',') //每一个值都是char
            num = num*10 + (*(str++)-'0');
        str++;
        TreeNode* root=new TreeNode(num);
        root->left = decode(str);
        root->right = decode(str);
        return root;
    }
};
```

#### 54、请找出其中的第k小的结点
>给定一棵二叉搜索树，请找出其中的第k小的结点。例如， （5，3，7，2，4，6，8）    中，按结点数值大小顺序第三小结点的值为4。

* //思路：二叉搜索树按照中序遍历的顺序打印出来正好就是排序好的顺序。
//     所以，按照中序遍历顺序找到第k个结点就是结果。
1.
```c++
class Solution {
public:
    TreeNode* KthNode(TreeNode* pRoot, int k)
    {
        if(pRoot== nullptr || k==0)
            return nullptr;
        return findKthNode(pRoot, k);
    }
private:
    TreeNode* findKthNode(TreeNode* pRoot, int &k)
    {
        TreeNode* target = nullptr;
        if(pRoot->left!=nullptr)
            target = findKthNode(pRoot->left, k);
        if(target == nullptr)
        {
            if(k==1)
                target = pRoot;
            k--;
        }
        if(target==nullptr && pRoot->right!=nullptr)
            target = findKthNode(pRoot->right, k);
        return target;
    }
};
```
2. 
```c++
class Solution {
public:
    int index = 0;
    TreeNode* KthNode(TreeNode* pRoot, int k)
    {
        if(pRoot!=nullptr)
        {
            TreeNode* node =  KthNode(pRoot->left, k);
            if(node!=nullptr) 
                return node;
            index++;    //中序遍历操作处
            if(index==k)
                return pRoot;
            node = KthNode(pRoot->right, k);
            if(node!=nullptr)
                return node;
        }
        return nullptr;
    }
};
```

#### 55、输入一棵二叉树，求该树的深度。
>输入一棵二叉树，求该树的深度。从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的长度为树的深度。

```c++
class Solution {
public:
    //二叉树的深度为从根节点到叶节点的最长路径长度;
    //如果只有根节点，那么树的深度是1;
    //如果有左右子树，那么树的深度为左子树与右子树中最大的深度+1;
    int TreeDepth(TreeNode* pRoot)
    {
        if(pRoot==nullptr)
            return 0;
        int left = TreeDepth(pRoot->left);
        int right = TreeDepth(pRoot->right);
        return 1+ max(left, right);
    }
};
```

#### 55.2、输入一棵二叉树，判断该二叉树是否是平衡二叉树。
>如果某二叉树中任意节点的左、右子树的深度相差不超过1，那么他就是一颗平衡二叉树。

* 方法1：求节点的左右子树的深度，然后判断是否相差不大于1；
    方法2：由于方法1中重复的遍历了某些节点，因此我们选择后序遍历每一个节点，这样的话在遍历一个节点之前，
    已经遍历了它的左、右子树；遍历的同时记录节点深度；

1.
```c++
class Solution {
public:
    /*
    方法1：求节点的左右子树的深度，然后判断是否相差不大于1；
    */
    bool IsBalanced_Solution(TreeNode* pRoot) {
        if(pRoot==nullptr)
            return true;
        int left = TreeDepth(pRoot->left);
        int right = TreeDepth(pRoot->right);
        int diff = left - right;
        if(diff<-1 || diff>1)
            return false;
        return IsBalanced_Solution(pRoot->left) && IsBalanced_Solution(pRoot->right);
    }
private:
    int TreeDepth(TreeNode* root)
    {
        if(root==nullptr)
            return 0;
        int left = TreeDepth(root->left);
        int right = TreeDepth(root->right);
        return 1+max(left,right);
    }
};
```
2.
```c++
class Solution {
public:
    /*
    方法1：求节点的左右子树的深度，然后判断是否相差不大于1；
    方法2：由于方法1中重复的遍历了某些节点，因此我们选择后序遍历每一个节点，这样的话在遍历一个节点之前，
    已经遍历了它的左、右子树；遍历的同时记录节点深度；
    */
    bool IsBalanced_Solution(TreeNode* pRoot) {
        int depth = 0;
        return IsBalanced(pRoot, depth);
    }
private:
    bool IsBalanced(TreeNode* pRoot, int &depth)
    {
        if(pRoot == nullptr)
        {
            depth=0;
            return true;
        }
        int left, right;
        if(IsBalanced(pRoot->left, left) && IsBalanced(pRoot->right, right))
        {
            int diff = left - right;
            if(diff <= 1 && diff >=-1)
            {
                depth = 1 + (left>right ?left :right);
                return true;
            }
        }
        return false;
    }
};
```

#### 按之字形打印树
> 请实现一个函数按照之字形打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右至左的顺序打印，第三行按照从左到右的顺序打印，其他行以此类推。

* 1.队列+数组翻转来实现，偶数的时候翻转数组再添加；2.双栈实现

```c++
class Solution {
public:
    vector<vector<int> > Print(TreeNode* pRoot) {
        if(pRoot==nullptr) return vector<vector<int> >();
        queue<TreeNode*> que;
        vector<vector<int>> ans;
        que.push(pRoot);
        bool even = false; //偶数的时候翻转数组
        while(!que.empty())
        {
            vector<int> _ans;
            int size=que.size();
            for(int i=0;i<size;++i) //打印完当前行的元素
            {
                TreeNode* front = que.front();
                _ans.push_back(front->val);
                que.pop();
                if(front->left!=nullptr)
                    que.push(front->left);
                if(front->right!=nullptr)
                    que.push(front->right);
            }
            if (even)
            {
                reverse(_ans.begin(),_ans.end());
            }
            even = !even;
            ans.push_back(_ans);
        }
        return ans;
    }
    
};
```

#### 112. Path Sum

问路径上是否有简单路径和等于给定数值
```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    bool hasPathSum(TreeNode* root, int sum) {
        if(root)
        {   
            //叶节点减为0才返回true
            if (!(sum-root->val) && !root->left && !root->right)
                return true;
            return (hasPathSum(root->left, sum-root->val) || hasPathSum(root->right, sum-root->val));
        }
        else
        {
            return 0;
        }
    }
};
```





#### 113. Path Sum II

上一题的基础上，再加上返回所有满足这个要求的路径
1. 直接类似上题解法：
```c++
class Solution {
public:
    vector<vector<int>> pathSum(TreeNode* root, int sum) {
        vector<vector<int> > paths;
        vector<int> path;
        findPaths(root, sum, path, paths);
        return paths;  
    }
private:
    void findPaths(TreeNode* node, int sum, vector<int>& path, vector<vector<int> >& paths) {
        if (!node) return;
        path.push_back(node -> val);
        if (!(node -> left) && !(node -> right) && sum == node -> val)
            paths.push_back(path);
        findPaths(node -> left, sum - node -> val, path, paths);
        findPaths(node -> right, sum - node -> val, path, paths);
        path.pop_back();
    }
};
```



### 递归



#### 1. 斐波那契数列：大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0）。

$$
f(n)=\begin{cases}
  0, & n=0\\
  1, & n=1\\
  f(n-1) +f(n-2), & n>1
\end{cases}
$$

```c++
class Solution {
public:
    int Fibonacci(int n) {
        int table[]={0,1};
        if(n<2) return table[n];
        for(int i=2; i<=n; ++i)
        {
            int tmp = table[0] + table[1];
            table[0] = table[1];
            table[1] =tmp;
        }
        return table[1];
    }
};
```

#### 2. 跳台阶：一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。
```c++
class Solution {
public:
    int jumpFloor(int number) {
        //两种情况：跳一步剩下（number-1）步；
        //跳两步剩下（number-2）步；
        if(number==0)
            return 0;
        if(number==1)
            return 1;
        if(number==2)
            return 2;
        return jumpFloor(number-1) + jumpFloor(number-2);
    }
};
```

题目描述：我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？

* 解析：参考：[https://www.nowcoder.com/questionTerminal/72a5a919508a4251859fb2cfb987a0e6?f=discussion](https://www.nowcoder.com/questionTerminal/72a5a919508a4251859fb2cfb987a0e6?f=discussion)

![image.png](https://upload-images.jianshu.io/upload_images/5433630-9f99003244f7c548.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


```c++
class Solution {
public:
    int rectCover(int number) {
        if(number<=2) return number;
        return rectCover(number-1) + rectCover(number-2);
    }
};
```





### 查找、排序



#### 旋转数组的最小数字
1. 哈希表的主要优点是在$O(1)$时间内查找某一元素，是效率最高的查找方式；但是缺点是需要额外的空间来实现哈希表。

2. 旋转数组的最小数字：

把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。 输入一个非减排序的数组的一个旋转，输出旋转数组的最小元素。 例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。 NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。

```c++
class Solution {
public:
    int minNumberInRotateArray(vector<int> rotateArray) {
        if(rotateArray.empty())
            return 0;
        //头部和尾部位置，指向两个排序好的子数组
        int indexFirst = 0;
        int indexLast = rotateArray.size()-1;
        //若全部数都旋转一次，最小值为首元素
        int indexMiddle = indexFirst;
        while(rotateArray[indexFirst]>=rotateArray[indexLast])
        {   
            //若相邻，那么右边索引在最小值处
            if(indexFirst+1 == indexLast)
                return rotateArray[indexLast];
            
            indexMiddle = (indexLast+indexFirst)/2; 
            //indexMiddle = (indexLast-indexFirst)/2 + indexFirst; 
            
            if((rotateArray[indexFirst] == rotateArray[indexLast]) && (rotateArray[indexLast]==rotateArray[indexMiddle]))
                return search(rotateArray,indexFirst,indexLast);

            if(rotateArray[indexMiddle]>=rotateArray[indexFirst])
            {
                indexFirst = indexMiddle;
            }
            //若不是右子数组最小的位置
            else if(rotateArray[indexMiddle]<=rotateArray[indexLast])
                indexLast = indexMiddle;
        }
    }
private:
    int search(const vector<int> &arr,const int left, const int right)
    {
        int min=arr[left];
        for(int i=left+1; i<=right; ++i)
            if(arr[i]>min) min=arr[i];
        return min;
    }
};
```

#### 统计一个数字在排序数组中出现的次数。

* 因为data中都是整数，所以可以稍微变一下，不是搜索k的两个位置，而是搜索k-0.5和k+0.5 这两个数应该插入的位置，然后相减即可。只是取的距离k最近的值，由于都是整数，原数组中可能存在k-1或者k+1；而k+0.5和k-0.5之间保证只有数字k。

```c++

class Solution {
public:
    int GetNumberOfK(vector<int> data ,int k) {
        return biSearch(data, k+0.5) - biSearch(data, k-0.5) ;
    }
private:
    int biSearch(const vector<int> &data, double k)
    {
        int start=0, end=data.size()-1;
        int mid;
        while(start<=end)
        {
            mid = start + (end-start)/2;
            if(data[mid]>k)
                end = mid-1;
            else if (data[mid]<k)
                start = mid+1;
        }
        //start是这个端点
        return start;
    }
};
```
#### 57、和为s的一对数
>输入一个递增排序的数组和一个数字S，在数组中查找两个数，使得他们的和正好是S，如果有多对数字的和等于S，输出两个数的乘积最小的。

* 解析：    //假设：若b>a,且存在，
    //a + b = s;
    //(a - m ) + (b + m) = s
    //则：(a- m)(b + m)=ab - (b-a)m - m*m < ab；说明外层的乘积更小
```c++
class Solution {
public:
    //假设：若b>a,且存在，
    //a + b = s;
    //(a - m ) + (b + m) = s
    //则：(a- m)(b + m)=ab - (b-a)m - m*m < ab；说明外层的乘积更小
    vector<int> FindNumbersWithSum(vector<int> array,int sum) {
        if(array.empty()) return vector<int>();
        vector<int> ans;
        vector<int>::iterator first=array.begin(); 
        vector<int>::iterator last=array.end()-1;
        while(first<last)
        {
            int fnum = *first, lnum=  *last;
            if(fnum+lnum == sum)
            {
                ans.push_back(fnum);
                ans.push_back(lnum);
                break;
            }
            else if(fnum + lnum < sum)
                first++;
            else
                last--;
        }
        return ans;
    }
};
```



#### 2.和为连续整数的序列
>小明很喜欢数学,有一天他在做数学作业时,要求计算出9~16的和,他马上就写出了正确答案是100。但是他并不满足于此,他在想究竟有多少种连续的正数序列的和为100(至少包括两个数)。没多久,他就得到另一组连续正数和为100的序列:18,19,20,21,22。现在把问题交给你,你能不能也很快的找出所有和为S的连续正数序列? Good Luck!


* 解析：考虑用small和big分别表示序列的最小值和最大值，初始化，small=1与big=2，如果序列和大于sum，那么去掉最小的small也就是说更新small++；小于序列的话，增大big，同时把small到big的和curSum更新；

```c++
class Solution {
public:
    vector<vector<int> > FindContinuousSequence(int sum) {
        if(sum<3) return vector<vector<int>>();
        vector<vector<int>> ans;
        int small=1, big=2, mid=sum/2;
        int curSum=small+big;
        //等于3的时候
        if(curSum==sum) 
            ans.push_back(initVec(small,big));
        while(small<mid)
        {
             //当前的和等于总和
            if(curSum==sum)
            {
                ans.push_back(initVec(small,big));
            }
            //当前的和大于总和；最小的左边未超出中位数
            while(curSum>sum && small <mid)
            {
                curSum-=small;
                small++;
                if(curSum==sum)
                    ans.push_back(initVec(small,big));
            }
             //当前的和小于总和,再加入大一位的数，并填入总和内
            big++;
            curSum+=big;
        }
        return ans;
    }
private:
    vector<int> initVec(int first, int last)
    {
        vector<int> vec;
        for(int i=first;i<=last;++i)
            vec.push_back(i);
        return vec;
    }
};
```





### 动态规划



#### 剪绳子
1. 剪绳子：长度为$n$的绳子，剪成$m$段，各段长度记为$k[0],...,k[m]$,求各段最大乘积多少？$（n>1,m>1）$
* dp：即$f(n)$为把长度$n$的绳子剪成若干段($[1,n-1]$)的最大乘积。
* 用子问题定义原问题：$f(n)= \max_{0<i<n}{f(i) * f(n-i)}$，
* 由此可知应自底向上计算，保存小的结果。
```c++
#include <iostream>
#include<vector>
#include<algorithm>
#include<exception>
#include<set>
#include<map>
using namespace std;

int maxProductAfterCutting(int length)
{
  if(length<2)
		return 0;
	if(length == 2)
		return 1;
	if(length == 3)
	 return 2;
	
	//products[i]代表f[i]
	int *products = new int[length+1] ;
	products[0] = 0; // dumps elems
	products[1] = 1;
	products[2] = 2;
	products[3] = 3;
	int max = 0;
	//遍历长度i
	for(int i=4; i<=length;++i)
	{
		max = 0;
		//切割大小为j,只计算前一半即可;
		for(int j=1; j<=i/2; ++j)
		{	
			//根据上述公式计算：取最大那个
			int product = products[j] * products[i-j];
			if(max < product) max = product;	
		}
		products[i] = max;	

	}
	max = products[length];
	delete[] products;
	return max;
}
int main() {
	cout << maxProductAfterCutting(9) << endl;
}

27
​```c++
* greedy：

​```c++
int maxProductAfterCuttingGreedy(int length)
{
  if(length<2)
		return 0;
	if(length == 2)
		return 1;
	if(length == 3)
	 return 2;
	
	//尽可能去剪长为3的绳子
	int timesOf3 = length / 3;
	// 当绳子为4例外,剪成2*2
	if (length - timesOf3*3==1)
			timesOf3-=1;
	int timesOf2 = (length - timesOf3 * 3) /2;
	return (int)(pow(3, timesOf3) * (int)(pow(2, timesOf2)));
}
```





###  位运算



1、机器数
一个数在计算机中的二进制表示形式,  叫做这个数的机器数。机器数是带符号的，在计算机用一个数的最高位存放符号, 正数为0, 负数为1.

比如，十进制中的数 +3 ，计算机字长为8位，转换成二进制就是00000011。如果是 -3 ，就是 10000011 。

那么，这里的 00000011 和 10000011 就是机器数。

2、真值
因为第一位是符号位，所以机器数的形式值就不等于真正的数值。例如上面的有符号数 10000011，其最高位1代表负，其真正数值是 -3 而不是形式值131（10000011转换成十进制等于131）。所以，为区别起见，将带符号位的机器数对应的真正数值称为机器数的真值。

例：
```c++
0000 0001的真值 = +000 0001 = +1，
1000 0001的真值 = –000 0001 = –1。
```
二. 原码, 反码, 补码的基础概念和计算方法.
在探求为何机器要使用补码之前, 让我们先了解原码, 反码和补码的概念.对于一个数, 计算机要使用一定的编码方式进行存储. 原码, 反码, 补码是机器存储一个具体数字的编码方式.

1. 原码
原码就是符号位加上真值的绝对值, 即用第一位表示符号, 其余位表示值. 比如如果是8位二进制:
```c++
[+1]原 = 0000 0001

[-1]原 = 1000 0001
```
第一位是符号位. 因为第一位是符号位, 所以8位二进制数的取值范围就是:
```c++
[1111 1111 , 0111 1111]
即
[-127 , 127]
```

原码是人脑最容易理解和计算的表示方式.

2. 反码
反码的表示方法是:

正数的反码是其本身

负数的反码是在其原码的基础上, 符号位不变，其余各个位取反.
```c++
[+1] = [00000001]原 = [00000001]反

[-1] = [10000001]原 = [11111110]反
```
可见如果一个反码表示的是负数, 人脑无法直观的看出来它的数值. 通常要将其转换成原码再计算.
3. 补码
补码的表示方法是:

正数的补码就是其本身

负数的补码是在其原码的基础上, 符号位不变, 其余各位取反, 最后+1. (即在反码的基础上+1)

```c++
[+1] = [00000001]原 = [00000001]反 = [00000001]补

[-1] = [10000001]原 = [11111110]反 = [11111111]补
```
对于负数, 补码表示方式也是人脑无法直观看出其数值的. 通常也需要转换成原码在计算其数值.


转：
作者：[张子秋](http://www.cnblogs.com/zhangziqiu/)
出处：[http://www.cnblogs.com/zhangziqiu/](http://www.cnblogs.com/zhangziqiu/)

-------------

####（c语言中移位运算只能用于整数，整数A左移1位得到的结果为A*2，右移1位为A/2取整)。

* 左移运算符$m\operatorname{<<}n$表示把$m$左移$n$位，最左边的$n$位被丢弃，同时在最右边补上$n$个0：
$$
00001010 << 2 = 00101000
$$
* 右移运算符$m>>n$表示把$m$右移$n$位，最右边的$n$位被丢弃：
      1. 如果数字是一个无符号数值，则用0填补最左边的$n$位；
      2. 如果数字是一个有符号数值，则用数值的符号位填补最左边$n$位。也就是说，正数在左边补0，负数在左边补1:

$$
00001010 >> 2 = 00000010\\
10001010 >> 3 = 11110001
$$
其中第二个的符号位是1。

常用`m>>1`表示`m/2`, `m&1`表示`m%2`。

1. 输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。

```c++
class Solution {
public:
     int  NumberOf1(int n) {
         int count=0;
         while(n)
         {
             ++count;
             n = (n-1) & n;
         }
         return count;
     }
};
```

-----
* Reference:
     * [https://www.cnblogs.com/zhangziqiu/archive/2011/03/30/ComputerCode.html](https://www.cnblogs.com/zhangziqiu/archive/2011/03/30/ComputerCode.html)


56、出现一次的数字
> 一个整型数组里除了两个数字之外，其他的数字都出现了两次。请写程序找出这两个只出现一次的数字。

* 解析：    //方法：考虑异或运算，两个想等的数异或运算之后等于0，整个数组做异或运算，其中两个
    //数字不一样的异或运算之后某一位肯定为1，按照这一位为1对数组做划分，使得两个子数组
    //部分只各有一个出现一次的数字
    //两个相同数字异或=0，一个数和0异或还是它本身。

```c++
class Solution {
public:

    void FindNumsAppearOnce(vector<int> data,int* num1,int *num2) {
        if(data.empty() || data.size()<2) return;
        int resultExclusiveOR = 0;
        for(int i=0; i<data.size();++i)
            resultExclusiveOR ^= data[i];
        unsigned int indexOf1 = FindFirstBitIs1(resultExclusiveOR);
        *num1 = *num2=0;
        for(int j=0; j<data.size();++j)
        {
            if(isBit1(data[j], indexOf1)) //按第indexOf1位是否为1划分数组
                *num1 ^= data[j];
            else
                *num2 ^= data[j];
        }
    }
    unsigned int FindFirstBitIs1(int num)
    {
        int indexBit=0;
        //按位与，都为1的时候该位才为1，最低位是1，若num的该位也是1，结果为0，便是找到了；否则为1，继续循环
        while(((num&1)==0) && (indexBit < 8 * sizeof(int)))
        {
            num= num >> 1;
            ++indexBit;
        }
        return indexBit;
    }
    //右移indexBit位，返回该位是否为1；
    bool isBit1(int num, unsigned int indexBit)
    {
        num = num >> indexBit;
        return (num & 1);
    }
};
```

问题：一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。

* 如图：$a_n=2(a_n-1)$；
![image.png](https://upload-images.jianshu.io/upload_images/5433630-85fe57149b32b741.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
```c++
class Solution {
public:
    int jumpFloorII(int number) {
        if(number ==1)
            return 1;
        return 2*jumpFloorII(number-1);
    }
};
```
2.

链接：[https://www.nowcoder.com/questionTerminal/22243d016f6b47f2a6928b4313c85387?f=discussion](https://www.nowcoder.com/questionTerminal/22243d016f6b47f2a6928b4313c85387?f=discussion)
来源：牛客网

其实是隔板问题，假设n个台阶，有n-1个空隙，可以用0~n-1个隔板分割，c(n-1,0)+c(n-1,1)+...+c(n-1,n-1)=2^(n-1)，其中c表示组合。

有人用移位1<<--number，这是最快的。直接连续乘以2不会慢多少，编译器会自动优化。不过移位还是最有启发的！

```
class Solution {
public:
    int jumpFloorII(int number) {
        int a=1;
        return a<<(number-1); //等价于2的（number-1）次方；
    }
};
```





###  回溯



前言

回溯法适合多个步骤组成的问题，每个步骤有多个选项，形成一颗树状。

在叶节点的状态不满足条件，回溯到上一个节点尝试其他的选项。如果再不满足，继续回溯。

#### 1. 矩阵中的路径
>请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一个格子开始，每一步可以在矩阵中向左，向右，向上，向下移动一个格子。如果一条路径经过了矩阵中的某一个格子，则之后不能再次进入这个格子。 例如 a b c e s f c s a d e e 这样的3 X 4 矩阵中包含一条字符串"bcced"的路径，但是矩阵中不包含"abcb"路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入该格子。

```c++
class Solution {
public:
    bool hasPath(char* matrix, int rows, int cols, char* str)
    {
        if(matrix == nullptr || rows<1 || cols<1 || str == nullptr)
            return false;
        //记录是否遍历过，回溯的时候需要重置
        bool *visited = new bool[rows*cols];
        memset(visited, 0, rows * cols);
        
        int pathLength = 0;
        for(int row=0; row<rows; ++row)
            for(int col=0; col<cols;++col)
            {
                if(hasPathCore(matrix,rows,cols,row,col,str,pathLength,visited))
                    return true;
            }
        delete[] visited;
        return false;
    }
private:
    //矩阵里matrix[row][col]开始的是否有路径pathLength,当str[pathLength]==‘\0’时表示已找到该路径
    bool hasPathCore(const char*matrix,int rows, int cols, int row, int col, const char*str,int&pathLength,bool*visited)
    {
        //终止递归的条件之一
        if(str[pathLength]=='\0')
            return true;
        
        bool hasPath=false;
        //路径第pathLength个元素等于char矩阵该位置的元素
        if(row>=0 && row<rows && col>=0 && col<cols && matrix[row*cols+col] == str[pathLength] && !visited[row*cols + col])
        {
            ++pathLength;//下一个字符
            visited[row*cols+col] = true; //已经访问
            //四个方向的递归
            hasPath = hasPathCore(matrix,rows,cols,row-1,col,str,pathLength, visited)
                || hasPathCore(matrix,rows,cols,row+1,col,str,pathLength, visited)
                || hasPathCore(matrix,rows,cols,row,col-1,str,pathLength, visited)
                || hasPathCore(matrix,rows,cols,row,col+1,str,pathLength, visited);
            
            if(!hasPath)//回溯
            {
                --pathLength;
                visited[row*cols+col] = false;
            }
        }
        return hasPath;
    }
};
```

####  2. 机器人的运动范围

>地上有一个m行和n列的方格。一个机器人从坐标0,0的格子开始移动，每一次只能向左，右，上，下四个方向移动一格，但是不能进入行坐标和列坐标的数位之和大于k的格子。 例如，当k为18时，机器人能够进入方格（35,37），因为3+5+3+7 = 18。但是，它不能进入方格（35,38），因为3+5+3+8 = 19。请问该机器人能够达到多少个格子？

```c++
class Solution {
public:
    //threshold:不能进入大于threshold的格子
    int movingCount(int threshold, int rows, int cols)
    {
        if(threshold<0 || rows <= 0 || cols <=0)
            return 0;
        
        bool* visited = new bool[rows*cols];
        memset(visited, 0, rows*cols);
        //返回可以走到的格子数目,从起点0,0开始走
        int count = movingCountCore(threshold,rows,cols,0, 0, visited);
        delete[] visited;
        return count;
    }
private:
    //还是传入具体的坐标值与访问表
    int movingCountCore(int threshold, int rows, int cols, int row, int col,bool*visited)
    {
        //一开始无格子
        int count=0;
        if(check(threshold, rows,cols,row,col,visited))
        {
            //已经走过
            visited[row*cols +col] = true;
            
            count = 1+movingCountCore(threshold,rows,cols,row-1,col,visited)
                + movingCountCore(threshold, rows,cols,row+1,col,visited)
                + movingCountCore(threshold, rows,cols,row,col-1,visited)
                + movingCountCore(threshold, rows,cols,row,col+1,visited);
        }
        return count;
    }
    bool check(int threshold, int rows, int cols, int row, int col,bool* visited)
    {
        //row从0到rows-1范围
        if(row>=0 && row<rows && col>=0 && col<cols && getDigitSum(row)+getDigitSum(col)<=threshold
          && !visited[row*cols+col])
            return true;
        return false;
    }
    //数位之和
    int getDigitSum(int number)
    {
        int sum=0;
        while(number>0)
        {
            sum+=number % 10;
            number /= 10;
        }
        return sum;
    }
};
```



### 字符串、链表



前言
c/c++把常量字符串放到单独的一个内存区域。当几个指针赋值给相同的常量字符串时，他们实际上会指向相同的内存地址：

```c++
int main()
{
    char str1[] = "hello world";
		char str2[] = "helle world";
		char* str3 = "hello world";
		char* str4 = "hello world";
		
		if(str1 == str2)
		 	cout << "str1 and str2 are same.\n";
		else
			cout <<	"str1 and str2 are not same.\n";

		if(str3==str4)
			cout << "str3 and str4 are same.\n";
		else
			cout << "str3 and str4 are not same.\n";

		return 0;	
}


 ./main
str1 and str2 are not same.
str3 and str4 are same.
```

第一个不同是因为两个不同的数组地址；第二个相同是因为常量字符串在内存中只有一个拷贝，他们都指向同一地址。

#### 5. 替换空格
>请实现一个函数，将一个字符串中的每个空格替换成“%20”。例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。

* 解析：先计算增大串长后的长`newStrLength`，用指针2指向该位置，同时指针1指向未增长时串中字符的结尾`'\0'`，从后遍历复制指针1、2上的字符。
```c++
class Solution {
public:
	void replaceSpace(char *str,int length) {
        if(str==nullptr || length<=0)
            return;
        int strLength=0, space=0;
        //先求替换后的字符串大小
        while(str[strLength]!='\0')
        {
            
            if(str[strLength]==' ')
                ++space;
            ++strLength;
        }
        //strLength指向终止符处
        //增大后的串长
        int newStrLength = strLength+2*space;
        if (newStrLength > length)//length为该串总容量
            return;
        while(strLength>=0 && newStrLength>strLength)
        {
            if(str[strLength]==' ')
            {
                str[newStrLength--] = '0';
                str[newStrLength--] = '2';
                str[newStrLength--] = '%';
            }
            else
            {
                str[newStrLength--]=str[strLength]; 
            }
            --strLength;
        }
	}
};
```

#### 19、正则表达式
>请实现一个函数用来匹配包括'.'和'\*'的正则表达式。模式中的字符'.'表示任意一个字符，而'\*'表示它前面的字符可以出现任意次（包含0次）。 在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但是与"aa.a"和"ab*a"均不匹配


* 解：当遇到`*`的时候可以当做前面的字符（一个字符）出现了0次，也就是忽略`*`直接前移2步，也可以当做出现n次，在原处待着不动，让str前移n步去比较；退出条件是看str是否为空且模式也为空就是匹配上，否则没有匹配上，递归的调用看各种情况下返回的是否有一个满足条件。

```c++
class Solution {
public:
    bool match(char* str, char* pattern)
    {
        if(str==nullptr || pattern==nullptr)
            return false;
        return matchCore(str, pattern);
    }
private:
    bool matchCore(char* str, char* pattern)
    {
    //若同时结束对比为匹配
    if(*str=='\0' && *pattern=='\0')
        return true;
    
    if(*str!='\0' && *pattern=='\0')
        return false;
    
    if(*(pattern+1)=='*')//若下一个字符为需要匹配的字符
    {
        //当前以及匹配
        if(*pattern==*str || (*pattern =='.' && *str!='\0'))
        {    
            //move on the next state || stay on current state || 向前移动2步，忽略一个‘*’匹配；
            return matchCore(str+1, pattern+2) || matchCore(str+1,pattern) || matchCore(str, pattern+2);
        }
        else
            //向前移动2步，忽略一个‘*’匹配；
            return matchCore(str,pattern+2);
    }
    // 当前字符以及匹配，去往下面的字符；
    if (*str==*pattern || (*pattern=='.' && *str!='\0'))
        return matchCore(str+1, pattern+1);
    return false;
    }  
};
```

#### 38、字符串的排列
>输入一个字符串,按字典序打印出该字符串中字符的所有排列。例如输入字符串abc,则打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。

* 解析：第一个字符固定为i，剩下其他字符的全排列在后面；i的取值是整个字符串的取值，因此，每次交换头字符，还有后面字符串中的下一个字符；递归结束条件是该字符串已经找到；

```c++
class Solution {
public:
    vector<string> Permutation(string str)
    {
        vector<string> result;
        if(str.empty()) return result;
         
        Permutation(str,result,0);
         
        // 此时得到的result中排列并不是字典顺序，可以单独再排下序
        sort(result.begin(),result.end());
         
        return result;
    }
     
    void Permutation(string str,vector<string> &result,int begin)
    {
        if(begin == str.size()-1) // 递归结束条件：索引已经指向str最后一个元素时
        {
            if(find(result.begin(),result.end(),str) == result.end())
            {
                // 如果result中不存在str，才添加；避免aa和aa重复添加的情况
                result.push_back(str);
            }
        }
        else
        {
            // 第一次循环i与begin相等，相当于第一个位置自身交换，关键在于之后的循环，
            // 之后i != begin，则会交换两个不同位置上的字符，直到begin==str.size()-1，进行输出；
            for(int i=begin;i<str.size();++i)
            {
                swap(str[i],str[begin]);
                Permutation(str,result,begin+1);
                swap(str[i],str[begin]); // 复位，用以恢复之前字符串顺序，达到第一位依次跟其他位交换的目的
            }
        }
    }
     
    void swap(char &fir,char &sec)
    {
        char temp = fir;
        fir = sec;
        sec = temp;
    }
};

```

-------
### 链表
#### 前言
* 内存分配不是在创建链表时一次完成的，而是每次添加一个节点是分配内存；因此没有闲置的内存，链表的空间效率比数组高。


* 指针的指针：
```c++
// Example program
#include <iostream>
#include <string>
#include<vector>
using namespace std;

int main()
{
    int i=1;
    int *pi = &i;
    int **ppi = &pi;
	cout<<"*pi :" << *pi << endl;
	cout<<"&pi :" << &pi << endl;
	cout <<"*ppi :" << *ppi<<endl;
	return 0;	
}

*pi :1
&pi :0x70aa208f64e8
*ppi :0x70aa208f64e4
```

* 链表的尾部添加节点和删除值为某数的节点如下：


```c++
// Example program
#include <iostream>
#include <string>
#include<vector>
using namespace std;

struct ListNode
{
	int value;
	ListNode *pNext;
};
//pHead是二级指针，指向指针的指针
void AddToTail(ListNode** pHead,int value)
{
	ListNode* pNew = new ListNode();
	pNew->value = value;
	pNew->pNext = nullptr;
	
	//是否为根节点
	if((*pHead)==nullptr)
	{
		//取出头指针指向当前新节点
		*pHead = pNew;
	}
	else
	{
		ListNode *pNode=*pHead;
		//通过指针链接新节点在尾部
		while(pNode->pNext!=nullptr)
					pNode = pNode->pNext;
		pNode->pNext =pNew;
	}
}

//删除某值元素
void RemoveNode(ListNode** pHead, int value)
{
	if(pHead==nullptr || *pHead==nullptr)
				return;
	//指向要删除的元素标记指针
	ListNode* pToBeDeleted = nullptr;
	
	
	//判断是否删除位置在头结点
	if((*pHead)->value == value)
	{
		//标记
		pToBeDeleted = *pHead;
		//新的头结点位置在下一个位置
		*pHead=(*pHead)->pNext;
	}
	else
	{
		//根节点的地址
		ListNode* pNode=*pHead;
		while(pNode->pNext != nullptr
		&& pNode->pNext->value!=value)
			pNode=pNode->pNext;
		//找到该元素的位置,为pNode的下一个位置
		if(pNode->pNext!=nullptr && pNode->pNext->value==value)
		{
			//该元素位置下一个位置赋给pToBeDeleted
			pToBeDeleted = pNode->pNext;
			//删除元素前一个位置的pNext链接到删除元素的下一个位置;之后就可以安全的删除了
			pNode->pNext = pNode->pNext->pNext;
		}
	}
	//删除该标记指针
	if(pToBeDeleted!=nullptr)
	{
		delete pToBeDeleted;
		pToBeDeleted = nullptr; //避免出错，赋为空
	}

}
int main()
{
		ListNode *test;
		ListNode** PP = &test;
		AddToTail(PP, 20);
		cout << test->value << endl;
		AddToTail(PP, 30);
		cout << test->pNext->value << endl;
		RemoveNode(PP,20);
		cout << test->value << endl;
		return 0;	
}

20
30
30
```

#### 6. 输入一个链表，按链表值从尾到头的顺序返回一个`ArrayList`。

```c++
class Solution {
public:
    //递归求解，或者用栈来求解。
    vector<int> printListFromTailToHead(ListNode* head) {
        vector<int> ans;
        stack<ListNode*> st;
        while(head!=nullptr)
        {
            st.push(head);
            head = head->next;
        }
        while(!st.empty())
        {
            ans.push_back(st.top()->val);
            st.pop();
        }
        return ans;
            
    }
```
#### 18. 删除链表中重复的元素
>在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。 例如，链表1->2->3->3->4->4->5 处理后为 1->2->5

```c++
class Solution {
public:
    //需要三个指针前中后
    ListNode* deleteDuplication(ListNode* pHead)
    {
        if(pHead==nullptr) return nullptr;
        ListNode *pPreNode = nullptr; //前一个节点
        ListNode *pNode = pHead; //当前节点
        while(pNode!=nullptr)
        {
            ListNode * pNextNode = pNode->next;//重置为当前节点的下一节点
            bool needDelete=false;
            //下一节点和当前节点是否是重复的
            if(pNextNode!=nullptr && pNode->val==pNextNode->val)
                needDelete = true;
            
            if(!needDelete)//没有重复，往下走
            {
                pPreNode = pNode;
                pNode = pNode->next;
            }
            else //重复节点
            {
                //取出值，删除连续等于该值的节点
                int value = pNode->val;
                ListNode* pToBeDel = pNode;
                //连续删除
                while(pToBeDel != nullptr && pToBeDel->val == value)
                {
                    pNextNode = pToBeDel->next;
                    delete pToBeDel;
                    pToBeDel = nullptr;
                    pToBeDel = pNextNode;
                }
            //确定是否是头结点
            if(pPreNode == nullptr)
                pHead = pNextNode;
            else//删除之后重链接删除后的节点
                pPreNode->next = pNextNode;
            pNode=pNextNode;
            }
        }
        return pHead;
    }
};
```


#### 17. 输入一个链表，输出该链表中倒数第k个结点。
* 双指针开始走，第一个走到k-1步时，第二个开始从根节点走。判断链长是否有k，没有解返回。
```c++
class Solution {
public:
    ListNode* FindKthToTail(ListNode* pListHead, unsigned int k) {
        if(!pListHead || k==0) return nullptr;
        ListNode* preListNode=nullptr;
        ListNode* curListNode=pListHead;
        //第一个指针走了k-1步，加上本身是一个位移，总共k位移，第二个指针开始从头结点走1步
        for(unsigned i=1;i<k;++i)
        {
            if(curListNode->next!=nullptr)
            {
                curListNode=curListNode->next;
            }
            else //链表长度不足
            {
                return nullptr;
            }
        }
        //开始走
        preListNode = pListHead;
        //下一个节点指针非空
        while(curListNode->next!=nullptr)
        {
            curListNode=curListNode->next;
            preListNode=preListNode->next;
        }
        return preListNode;
    }
};
```

#### 23.链表中环的入口节点
>给一个链表，若其中包含环，请找出该链表的环的入口结点，否则，输出null。

解：1. 确定链表是否有环；（设置一个走得快的指针是否能追上走得慢的指针；）
        2. 环的大小是多少；
        3. 快指针先走环的大小步数，慢指针再走，必然在环的入口处相遇。

```c++
class Solution {
public:
    ListNode* EntryNodeOfLoop(ListNode* pHead)
    {
        //meetingnode返回该链表是否有环，有的情况下，返回快慢指针相遇的节点
        ListNode* ListMeetNode=MeetingNode(pHead);
        if(ListMeetNode==nullptr)
            return nullptr;
        int countLoop=1;
        ListNode* pNode = ListMeetNode;
        while(pNode->next!=ListMeetNode)
        {
            countLoop++;
            pNode=pNode->next;
        }
        pNode=pHead;
        ListNode *slowNode=pHead;
        //先走k-1步
        for(int i=0;i<countLoop;++i)
        {
            pNode=pNode->next;
        }
        while(pNode!=slowNode)
        {
            pNode=pNode->next;
            slowNode=slowNode->next;
        }
        return slowNode;
    }
private:
    ListNode* MeetingNode(ListNode* head)
    {
        if(head == nullptr)//空链表的情况
            return nullptr;
        ListNode* pslow=head->next;
        if(pslow==nullptr)//无环的情况
            return nullptr;
        ListNode* pfast=pslow->next;//快一步
        while(pfast!= nullptr && pslow!=nullptr)
        {
            if(pfast==pslow)//有环
                return pfast;
            pslow = pslow->next;
            pfast = pfast->next;
            if(pfast!=nullptr)//多走一步
                pfast=pfast->next;
        }
        return nullptr;
    }
};
```

#### 35、复杂链表的复制
> 输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），返回结果为复制后复杂链表的head。（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）


* 解析：重复各个节点，然后关联新的重复节点的random，再然后重链接取出偶数链接的链表。
```c++
class Solution {
public:
    //复制原始链表的任一节点N并创建新节点N'，再把N'链接到N的后边
    void CloneNodes(RandomListNode* pHead)
    {
        RandomListNode* pNode=pHead;
        while(pNode!=NULL)
        {
            RandomListNode* pCloned=new RandomListNode(0);
            pCloned->label=pNode->label;
            pCloned->next=pNode->next;
            pCloned->random=NULL;
              
            pNode->next=pCloned;
              
            pNode=pCloned->next;
        }
    }
    //如果原始链表上的节点N的random指向S，则对应的复制节点N'的random指向S的下一个节点S'
    void ConnectRandomNodes(RandomListNode* pHead)
    {
        RandomListNode* pNode=pHead;
        while(pNode!=NULL)
        {
            RandomListNode* pCloned=pNode->next;
            if(pNode->random!=NULL)
                pCloned->random=pNode->random->next;
            pNode=pCloned->next;
        }
    }
    //把得到的链表拆成两个链表，奇数位置上的结点组成原始链表，偶数位置上的结点组成复制出来的链表
    RandomListNode* ReConnectNodes(RandomListNode* pHead)
    {
        RandomListNode* pNode=pHead;
        RandomListNode* pClonedHead=NULL;
        RandomListNode* pClonedNode=NULL;
          
        //初始化
        if(pNode!=NULL)
        {
            pClonedHead=pClonedNode=pNode->next;
            pNode->next=pClonedNode->next;
            pNode=pNode->next;
              
        }
        //循环
        while(pNode!=NULL)
        {
            pClonedNode->next=pNode->next;
            pClonedNode=pClonedNode->next;
            pNode->next=pClonedNode->next;
            pNode=pNode->next;
        }
          
        return pClonedHead;
          
    }
    //三步合一
    RandomListNode* Clone(RandomListNode* pHead)
    {
        CloneNodes(pHead);
        ConnectRandomNodes(pHead);
        return ReConnectNodes(pHead);
    }
};
```


#### 52、两个链表的第一个公共节点
>输入两个链表，找出它们的第一个公共结点。

* 解：    方法1：对链表1中每一个节点，都遍历一遍链表2，复杂度为O(mn);m\n分别为两个链表的长度；
    方法2：若中间某一个节点相同，之后剩下的节点都是一样的地址存在，因此结尾处都一样，
    设置栈结构，从尾部排除，遇到不相等的元素时，之前的那个元素便是第一个相同的元素点，空间换时间
    的O(m+N)空间换取时间为O(m+n);
    方法3：先遍历两个链表得到最长链表的长度s1与短链表的长度s2，在最长链表上先移动s1-s2,再同时移动
    链表1,2，且不断比较直到相等；

方法3的原因是：链表$A$长度是$LA$，链表$B$长度是$LB$，他们的公共部分长度是$LC$，那么不妨设$LA$长度更长，那么有：$ LA>LB \ge LC$,那么$A$与$B$的公共部分至少在$A$后面的长度还有$B$这么长的时候才有可能。


```c++
class Solution {
public:
    /*
    */
    ListNode* FindFirstCommonNode( ListNode* pHead1, ListNode* pHead2) {
        if(!pHead1 || !pHead2) return nullptr;
        ListNode *pDupHead1 = pHead1;
        ListNode *pDupHead2 = pHead2;
        int  count1=0, count2=0;
        while(pHead1)
        {
            count1++;
            pHead1=pHead1->next;
        }
        while(pHead2)
        {
            count2++;
            pHead2=pHead2->next;
        }
        if(count1>count2) //链表1最长
        {
            int lCount = count1 - count2;
            while(lCount--)
                pDupHead1=pDupHead1->next;
        }
        else
        {
            int lCount = count2 - count1;
            while(lCount--)
               pDupHead2 = pDupHead2->next; 
        }
        while(pDupHead1 && pDupHead2 && (pDupHead1!=pDupHead2))
        {
            pDupHead1 = pDupHead1->next;
            pDupHead2 = pDupHead2->next;
        }
        ListNode * pFirstCommonNode = pDupHead1;
        return pFirstCommonNode;
    }
};
```

#### 50、第一个只出现一次的字符
>在一个字符串(0<=字符串长度<=10000，全部由字母组成)中找到第一个只出现一次的字符,并返回它的位置, 如果没有则返回 -1（需要区分大小写）.

* :方法1：暴力求解,对当前每一个字符，搜索后面的真个字符串，复杂度是O(n^2);
方法2：哈希表：自建一个简单哈希表；
可以把哈希表设为256，因为char是8bit的类型，总共只有256个字符，
可以设置哈希表为负，第一次插入后为正，再次插入同样的字符，设为负，下次只有找出数组非负的位置便是对应的第一次出现的字符

```c++
class Solution {
public:
    int FirstNotRepeatingChar(string str) {
        if(str.empty()) return -1;
        vector<int> arr(26*2,0);
        for(auto i:str)
        {
            if('a'<=i && i<='z')
                arr[int(i-'a')]+=1;
            else
                arr[int(i-'A')+26]+=1;
        }
        
        
        for(int j=0; j<str.size(); ++j)
        {
            int numj;
            if('a'<=str[j] && str[j]<= 'z')
                numj=int( str[j]-'a');
            else
                numj=int( str[j]-'A'+26);
            if(arr[numj] == 1)
                return j;
        }
    }

};
```

#### 58、反转单词顺序字符串
>牛客最近来了一个新员工Fish，每天早晨总是会拿着一本英文杂志，写些句子在本子上。同事Cat对Fish写的内容颇感兴趣，有一天他向Fish借来翻看，但却读不懂它的意思。例如，“student. a am I”。后来才意识到，这家伙原来把句子单词的顺序翻转了，正确的句子应该是“I am a student.”。Cat对一一的翻转这些单词顺序可不在行，你能帮助他么？

* 先反转整个串，再反转每一个单词；
```c++
class Solution {
public:
    string ReverseSentence(string str) {
        if(str.empty()) return str;
        int begin=0,end=0;
        reverse(str.begin(),str.end());
        int size = str.size();
        while(begin<size)
        {
            //起点为空，重置游标
            while(begin<size && str[begin]==' ') begin++;
            end=begin;
            //终点不为空，一直往前走到头
            while(end<size && str[end]!=' ') end++;
            reverse(str.begin()+begin, str.begin()+end);
            //反转之后，起点的顺序是重点的顺序
            begin = end;
        }
        return str;
    }
};
```

#### 2。左旋转字符串
>字符序列S=”abcXYZdef”,要求输出循环左移3位后的结果，即“XYZdefabc”。是不是很简单？OK，搞定它！

* 先翻转这个串的前后两部分，然后再全部一次性翻转
```c++
class Solution {
public:

    string LeftRotateString(string str, int n) {
        if(str.empty() || n<=0) return str;
        reverse(str.begin(),str.begin()+n);
        reverse(str.begin()+n,str.end());
        reverse(str.begin(),str.end());
        return str;
    }
};
```

#### 67、将一个字符串转换成一个整数
>将一个字符串转换成一个整数(实现Integer.valueOf(string)的功能，但是string不符合数字要求时返回0)，要求不能使用字符串转换整数的库函数。 数值为0或者字符串不是一个合法的数值则返回0。

```c++
class Solution {
public:
    int StrToInt(string str) {
        long long int num = 0;
        if(str.empty()) return num;
        int minus;
        if(str[0] == '+')
            minus = 1;
        if(str[0] == '-')
            minus = -1;
        int i;
        if(str[0]!='+' && str[0]!='-') i=0;
        else i=1;
        for(;i<str.size();++i)
        {           
            if(str[i]>='0' && str[i]<='9')
            {
                num = num * 10 + minus * (str[i]-'0');
            }
            else
            {
                num = 0;
                break;
            }
        }
        return (int)num; //类型转换
    }
};
```

#### 50、字符流中第一个出现的字符

方法1：用常规方法来解：
```c++
class Solution
{
public:
  //Insert one char from stringstream
    string str;
    //全部初始化为0
    int hash[256]= {0};
    void Insert(char ch)
    {
        str+=ch;
        hash[ch]++;
    }
  //return the first appearence once char in current stringstream
    char FirstAppearingOnce()
    {
        int size = str.size();
        for(int i=0; i<size; ++i)
        {
            if(hash[str[i]]==1)
                return str[i];
        }
        return '#';
    }
};
```

方法2：用stl来解：
```c++
class Solution
{
public:
  //Insert one char from stringstream
    string str;
    map<char,int> m;    //用map来记录字符出现的次数
    void Insert(char ch)
    {
        str += ch;
        m[ch]++;
    }
  //return the first appearence once char in current stringstream
    char FirstAppearingOnce()
    {
        for(auto it : str)
        {
            if(m[it] == 1)
            {
                return it;
            }
        }
        return '#';
    }

};
```

#### 请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。
例如，字符串"+100","5e2","-123","3.1416"和"-1E-16"都表示数值。 但是"12e","1a3.14","1.2.3","+-5"和"12e+4.3"都不是。
```c++
class Solution {
public:
    bool isNumeric(char* str) {
        // 标记符号、小数点、e是否出现过
        bool sign = false, decimal = false, hasE = false;
        for (int i = 0; i < strlen(str); i++) {
            if (str[i] == 'e' || str[i] == 'E') {
                if (i == strlen(str)-1) return false; // e后面一定要接数字
                if (hasE) return false;  // 不能同时存在两个e
                hasE = true;
            } else if (str[i] == '+' || str[i] == '-') {
                // 第二次出现+-符号，则必须紧接在e之后
                if (sign && str[i-1] != 'e' && str[i-1] != 'E') return false;
                // 第一次出现+-符号，且不是在字符串开头，则也必须紧接在e之后
                if (!sign && i > 0 && str[i-1] != 'e' && str[i-1] != 'E') return false;
                sign = true;
            } else if (str[i] == '.') {
              // e后面不能接小数点，小数点不能出现两次
                if (hasE || decimal) return false;
                decimal = true;
            } else if (str[i] < '0' || str[i] > '9') // 不合法字符
                return false;
        }
        return true;
    }
};
```

