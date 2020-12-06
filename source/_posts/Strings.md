---
title: Strings
mathjax: true
date: 2019-09-16 21:25:49
tags: Algorithm
categories: 算法与数据结构
visible:
---

#### 判断两个字符串是否为变形词

> 给定两个字符串str1和str2，如果str1和str2中出现的字符种类出现的一样且每种字符出现的次数也一样，那么str1和str2互为变形词。请判断str1和str2是否为变形词。
>
> ```c++
> 输入包括3行，第一行包含两个整数n，m(1 \leq n,m\leq 10^5)(1≤n,m≤105)分别代表str1和str2的长度，第二行和第三行为两个字符串，分别代表str1和str2。
> ```

* 解析：利用hash表对字符串1元素累加，然后对串2中出现在串1的元素累减；判断value是否都是0便可；

  ```c++
  int main()
  {
      int n, m; cin >> n >> m;
      
      string str1, str2;
      cin >> str1 >> str2;
      if  (str1.size()!= n || str2.size() != m || n != m) {cout << "false"; return 0;}
  
      unordered_map<char, int> table;
      for (auto c : str1)
          table[c]++;
      for(auto c : str2)
      {
          table[c]--;
      }
  
      for (auto it = table.begin(); it != table.end(); it++)
      {
          if (it->second != 0)
          {cout << "false"; return 0;}
      }
      cout << "true" << endl;
  }
  ```

  #### 判断两个字符串是否互为旋转词

  > 如果一个字符串为str，把字符串的前面任意部分挪到后面形成的字符串交str的旋转词。比如str=“12345”，str的旋转串有“12345”、“45123”等等。给定两个字符串，判断是否为旋转词。

  

  * 解析：这里用KMP算法，判断串1与串2是否互为旋转词，可以把串2拼接上串2等到拼接串2，若串1是拼接串2的子串，那么肯定互为旋转词，判断是不是子串，可以用KMP来做。时间空间复杂度都是$O(N)$，满足情况。

    

    ```c++
    #include <iostream>
    #include <string>
    #include <vector>
    using namespace std;
    
    int KMP(const string& str1, const string& pattern)
    {
        int n = str1.size(), m = pattern.size();
        vector<int> next(m, 0);
        int k = 0;
        for (int i = 1; i < m; i++)
        {
            while(k > 0 && pattern[k] != pattern[i]) k = next[k - 1];
            if (pattern[i] == pattern[k])
                next[i] = ++k;
        }
        k = 0;
        for (int i = 0; i < n; i++)
        {
            while(k > 0 && str1[i] != pattern[k]) k = next[k - 1];
            if (str1[i] == pattern[k]) k++;
            if (k == m)
            {
                return i - k + 1;
            }
        }
        return -1;
    }
    
    int brute(const string& str1, const string& str2)
    {
        for (int i = 0; i < str1.size() - str2.size(); i++)
        {
            int tmp = i;
            for (int j = 0; j < str2.size(); j++)
            {
                if (str1[tmp++] != str2[j])
                    break;
            }
            if (tmp == str2.size())
                return i;
        }
        return -1;
    }
    
    int main(){
        int n, m; cin >> n >> m;
        string str1, str2; cin >> str1 >> str2;
        if (n != m || str1.size() != n || str2.size() != m)
        {cout <<"NO"; return 0;}
        string combine = str2 + str2;
        cout << (KMP(combine, str1) >= 0 ? "YES" : "NO") << endl;
    }
    ```

  
  #### 将整数字符串转成整数值
  
  > 给定一个字符串str，如果str符合日常书写的整数形式，并且属于32位整数范围，返回str代表的整数值，否则返回0。
  
  
  
  * 解析：判断数字的符号；判断数字的范围，超出数字范围直接返回false；若为0，必须是在i的最后位置上；判断范围是否在int内。
  
    
  
  ```c++
  #include<climits>
  #include<string>
  #include<iostream>
  using namespace std;
  
  
  bool isNormalString(const string& str, long &num)
  {
      if (str.empty()) return 0;
      int tmp = 0;
      bool sign = true; 
      for (int i = 0; i < str.size(); i++)
      {
          /*判断数字的符号*/
          if (i == 0 && str[i] == '-')
          {
              sign = false;
              continue;
          }
          /*判断数字的范围，超出数字范围直接返回false*/
          tmp = str[i] - '0';
          
          if (tmp < 0 || tmp > 9) return false;
          /*若为0，必须是在i的最后位置上*/
          if (tmp == 0 && i != str.size() - 1)   
          {
              return false;
          }
          num = num * 10 + tmp;
          /*判断范围是否在int内*/
          if (sign && num > INT_MAX  || !sign && -num < INT_MIN)
              return false;
      }
      if (!sign) num = - num;
      return true;
  }
  
  int main()
  {
      string str; cin >> str;
      long num = 0;
      if (isNormalString(str, num))
          cout << num;
      else 
          cout << 0;
  }
  ```
  
  

#### 字符串的统计字符串

> 给定一个字符串str，返回str的统计字符串。例如“aaabbbbcccd”的统计字符串为“a_3_b_4_c_3_d_1”。
>
> ```c++
> offerofferzainaliiiiii
> 输出：
> o_1_f_2_e_1_r_1_o_1_f_2_e_1_r_1_z_1_a_1_i_1_n_1_a_1_l_1_i_6
> ```



* 解析：c++ 直接 `ans += str[i - 1] + "_" + to_string(num) `会报错；

```c++
string decoder(const string &str)
{
    int num = 1;
    string ans = "";
    if (str.empty()) return ans;
    for (size_t i = 1; i <= str.length(); i++)
    {
        if (i < str.size() && str[i] == str[i - 1])
            num += 1;
        else
        {
            if (!ans.empty()) ans += "_";
            //ans += str[i - 1] + "_" + to_string(num) 会报错
            ans = ans + str[i - 1] + "_" + to_string(num);
        }
    }
    return ans;
}

int main()
{
    string str;
    cin >> str;
    cout << decoder(str) << endl;
}
```

#### 判断数组中所有的数字是否只出现一次

> 给定一个个数字arr，判断数组arr中是否所有的数字都只出现过一次。
>
> ```c++
> 1.时间复杂度O（n）。
> 2.额外空间复杂度O（1）。
> ```

* 解析：方法1：时间空间复杂度为O(n)的方法，利用hashtable查找；
  方法2：非递归的堆排序:

  它的父节点子节点坐标为：

  ```pseudocode
  iParent(i) = floor((i-1) / 2) where floor functions map a real number to the smallest leading integer.
  iLeftChild(i)  = 2*i + 1
  iRightChild(i) = 2*i + 2
  ```

  最大堆排序的过程：

  1、建立堆：循环从最后一个父节点到第一个父节点开始建立堆；把子节点与父节点对比，子节点大就与父节点交换；然后父节点的索引变为子节点的索引，子节点重新根据父节点计算索引，原因是交换了父节点，可能导致下游的子树不满足最大堆性质；

  2、循环交换第一个父节点与最尾部节点，然后对【0，尾部index-1】的部分重新维持堆性质。

```c++
/*方法1
int main()
{
    int N; cin >> N;
    map<int, int> table;
    for (int i = 0; i < N; i++)
    {
        int  tmp; cin >> tmp;
        if(table[tmp] != 0){
            cout << "NO" << endl;
            return 0;
        }
        else
            table[tmp]++;
    }
    cout << "YES" << endl;
    return 0;
}
*/
#include<iostream>
#include<vector>
using namespace std;

void max_heapify(vector<int> &arr, int start, int end)
{
    int dad = start;
    int son = dad * 2 + 1;
    while(son <= end)
    {
        if (son + 1 <= end && arr[son + 1] > arr[son])
            son++;
        if (arr[dad] > arr[son])
            return;
        else
        {
            swap(arr[dad], arr[son]);
            dad = son;
            son = dad * 2 + 1;
        }
    }
}
void heap_sort(vector<int>& arr){
    for (int i = arr.size() / 2 - 1; i >= 0; i--)
    {
        max_heapify(arr, i, arr.size() - 1);
    }
    for (int i = arr.size() - 1; i > 0; i--)
    {
        swap(arr[i], arr[0]);
        max_heapify(arr, 0, i - 1);
    }
}
 
int main(){
    int N; cin >> N;
    vector<int> table(N, 0);
    for(int i = 0; i < N; i++)
        cin >> table[i];
    heap_sort(table);
    bool flag = true;
    for(int i = 0;i < table.size() - 1; ++i){
        if(table[i] == table[i + 1]){
            flag = false;
        }
    }
    cout<<(flag ? "YES": "NO")<<endl;
    return 0;
}
```



#### 在有序但是含有空的数组中查找字符串

> 给定一个字符串数组strs[]，在strs中有些位置为null，但在不为null的位置上，其字符串是按照字典序由小到大出现的。在给定一个字符串str，请返回str在strs中出现的最左位置，如果strs中不存在str请输出“-1”。
>
> ```c++
> 输出包含多行，第一行包含一个整数n代表strs的长度，第二行一个字符串，代表str，接下来n行，每行包含一个字符串构成，字符串只包含小写字符，如果这一行为“0”，代表该行字符串为空，这n行字符串代表strs。（数据保证当字符串不为空时，所有字符均为小写字母；保证所有的字符串长度都小于10，1 \leq n \leq 10^51≤n≤105）
> ```



* 解析：顺序排好的字符串，用二分查找第一个出现的字符，但是中间插入了很多“0”字符，中间搜索到0的时候，往左移动，找到最左边非0的位置。再做一般性比较。

```c++
int searchLeftFirstChar(const vector<string> &table,const string &target)
{
    if (target.empty()) return -1;
    int end = table.size() - 1, start = 0;
    int res = -1;
    while(start <= end)
    {
        int mid = start + ((end - start) >> 1);
        //非空位置，且等于目标，那么前面可能也等于目标是，移动end到mid前一位
        //cout << " mid " << mid << " ";
        //cout << table[mid] << endl;
        if (table[mid] != "0" && table[mid].compare(target) == 0)
        {
            res = mid;
            end = mid - 1;
        }
        //不等且
        else if (table[mid] != "0")
        {
            if (table[mid].compare(target) > 0)
                end = mid - 1;
            else
                start = mid + 1;
        }
        else //mid目前为空，试着找
        {
            int i = mid;
            while(i >= start && table[i] == "0") i--;
            if (i < start || table[i] < target) //左边第一个非空字符小于目标或者左边都是空，目标不在这里
                start = mid + 1;
            else
            {
                res = (table[i].compare(target) > 0) ? res: i;
                end =  i - 1;
            }
        }
    }
    return res;
}

int main()
{
    int N; cin >> N;
    string target; cin >> target;
    vector<string> table(N);
    for (int i = 0; i < N; i++)
    {
        cin >> table[i];
    }
    cout << searchLeftFirstChar(table, target) << endl;
    return 0;
}
```





#### 字符串的调整I

> 给定一个字符串chas[],其中只含有字母字符和“*”字符，现在想把所有“*”全部挪到chas的左边，字母字符移到chas的右边。完成调整函数。
>
> ```pseudocode
> qw**23
> 
> **qw23
> ```

* 从后往前排，用一个标记*，一个标记字符。

```c++
int main()
{
    string target;
    cin >> target;
    for (int i = target.size()-1, j = -1; i >= 0; i--)
    {
        if (target[i] == '*' && j < 0) j = i;
        else if (target[i] != '*')
        {
            if (j >= 0)
                swap(target[i], target[j--]);
        }
    }
    cout << target << endl;
}
```



#### 字符串的调整II

> 给定一个字符类型的数组chas[],chas右半区全是空字符，左半区不含有空字符。现在想将左半区的空格字符串替换成“%20”，假设chas右半区足够大，可以满足替换需要的空间，请完成替换函数。
>
> ```pseudocode
> a  b    c
> 
> a%20%20b%20%20%20%20c
> ```

* 解析：c++中的string a; cin >> a; 只能输入单字符串，因为它会以空白符作为退出；因此要读入一段句子，可以用`getline(cin, a)`；

  先遍历a计算出空白符的数目，然后知道了string变形后的大小，利用resize实现。然后从后面依次替换。

```c++
#include<iostream>
#include <string>
#include<vector>
using namespace std;


void replace(string &target)
{
    if (target.empty()) return;
    int numOfSpace = 0, length = target.size();
    for (auto c : target)
        if (c == ' ')
            numOfSpace++;
    target.resize(length + 2 * numOfSpace);
    for (int i = target.size() - 1, j = length - 1; j >= 0; i--, j--)
    {
        if (target[j] == ' ')
        {
            target[i--] = '0'; target[i--] = '2'; target[i] = '%';
        }
        else
        {
            target[i] = target[j];
        }
    }
}

int main()
{
    string target;
    getline(cin ,target);
    replace(target);
    cout << target << endl;
}
```





#### 翻转字符串（1）

> 给定字符类型的数组chas，请在单词间做逆序调整。只要做到单词的顺序逆序即可，对空格的位置没有要求。
>
> ```c++
> i am a student
> 
> i ma a tneduts
> ```

* 解析：

```c++
void reverse(string &target)
{
    if (target.empty()) return;
    for (int i = 0, j = 0; i < target.size(); i++)
        if (target[i] == ' ' || i == target.size() - 1)
        {
            int k = (i == target.size() - 1) ? i : i - 1;
            while(j < k)
            {
                swap(target[j++], target[k--]);
            }
            j = i + 1;
        }
}

int main()
{
    string target;
    getline(cin, target);
    reverse(target);
    cout << target << endl;
}
```

python: hehe,挺有意思的。

```python
chas = list(input().split())
for i in range(len(chas)):
    chas[i] = chas[i][::-1]
print(' '.join(chas))
```



#### 翻转字符串（2）

> 给一个字符类型的数组chas和一个整数size，请把大小为size的左半区整体右移到右半区，右半区整体移动到左边。
>
> ```pseudocode
> 3
> abcdefg
> 
> defgabc
> ```

* 解析：双重反转，先对第一段反转，然后是第二段，最后全部翻转；

```c++
int main()
{
    int N; cin >> N;
    string target;
    cin >> target;
    //cbadefg
    reverse(target.begin(), target.begin() + N);
    //cbagfed
    reverse(target.begin() + N, target.end());
    //defgabc
    reverse(target.begin(), target.end());
    cout << target << endl;
}
```

python：

```python
num = int(input())
chas = input()
print(chas[num: ] + chas[ :num])
```

方法2：

* 解析：画出来一个AB123456，把前两个交换到最后的实例；先交换一次：561234AB；之后得到最后的AB不用交换了；剩下561234；交换小的那部分得到341256；剩下3412；交换得到1234，左右部分都相等了交换结束。

```c++
void exchange(string &s, int start, int end, int size)
{
    int i = end - size + 1;
    while(size-- != 0) 
    {
        swap(s[start++], s[i++]);
    }
}


void rotate(string &s, int size)
{
    if (s.empty()) return;
    int end = s.size() - 1, start = 0;
    int lsize = size;
    int rsize = s.size() - size;
    int minSize = min(lsize, rsize);
    int dir = lsize - rsize; 
    while(true)
    {
        exchange(s, start, end, minSize);
        if (dir > 0)
        {
            start += minSize;
            lsize = dir;
        }
        else if (dir < 0)
        {
            end -= minSize;
            rsize = -dir;
        }
        else
        {
            break;
        }
        dir = lsize - rsize;
        minSize = min(lsize, rsize);
    }
}

int main()
{
    int N; cin >> N;
    string target;
    cin >> target;
    rotate(target, N);
    cout << target << endl;
}
```



#### 完美洗牌问题（1）

> 给定一个长度为偶数的数组arr，长度记为2*N。前n个为左部分，后n个为右部分。arr可以表示为\{L_1,L_2...L_{n-1},L_n,R_1,R_2...R_{n-1},R_n \}{*L*1,*L*2...*L**n*−1,*L**n*,*R*1,*R*2...*R**n*−1,*R**n*},请将数组调整成\{ R_1,L_1,R_2,L_2,...R_n,L_n \}{*R*1,*L*1,*R*2,*L*2,...*R**n*,*L**n*}。
>
> ```c++
> 6
> 1 2 3 4 5 6
> 
> output:
> 4 1 5 2 6 3
> ```

* 解析：要求时间复杂度O(n),空间复杂度O(1)。
* 参考当数组的长度满足2*N == （3^k）-1时，才能使用公式替换；
  因此对普通偶数长度的数组需要先切块；每块都左右部分交换之后，对该块利用公式替换；    


```c++
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;


void rotate(vector<int> &arr, int start, int  end)
{
    while(start < end)
    {
        swap(arr[start++], arr[end--]);
    }
}

//第一步：切割arr为k块，数组位置i使得(3^k)-1中的k最大的数i位置；然后剩下的块大小也如此计算

void shuffle(vector<int> &arr, int start, int end)
{
    if (arr.empty() || start >= end) return;
    int size = end - start + 1;
    //1：找到最大的块大小，剩下的部分
    int k = 0; 
    while(pow(3, k) -1 <= size) k++;
    int residue = size - pow(3, k - 1) + 1; 
    //剩余部分不为0,那么要进行翻转
    if (residue != 0)
    {
        rotate(arr, start + (size - residue) / 2, start + size / 2 - 1);
        rotate(arr, start + size / 2, start + size - residue / 2 - 1);
        rotate(arr, start + (size - residue) / 2, start + size - residue / 2 - 1);
    }
    for (int i = 1; i < k; i++) {
        int time = start + pow(3, i - 1) - 1;
        int index = time;
        int nextIndex = -1;
        int next_temp = 0;
        int cur_temp = arr[index];
        //这里处理的都是满足公式长度的比如【L1,L2,L3,L4,R1,R2,R3,R4】
        while (nextIndex != time) {
            if (index <= start + (size - residue) / 2 - 1) //数组左边部分满足的递推公式
                nextIndex = 2 * index - start + 1;
            else //数组右边部分满足的递推公式nextIndex = 2*(i-N)-1，多乘了一个start长度被减去
                nextIndex = 2 * (index - (size - residue) / 2) - start;
            next_temp = arr[nextIndex];
            arr[nextIndex] = cur_temp;
            index = nextIndex;
            cur_temp = next_temp;
        }
    }
    shuffle(arr, end - residue + 1, end);
}

int main()
{
    int n; cin >> n;
    vector<int> arr(n);
    for (int i = 0; i < n; i++)
        cin >> arr[i];
    shuffle(arr, 0, arr.size() - 1);
    for (auto num : arr)
        cout << num << " ";
}
```



#### 完美洗牌问题（2）

> 给定一个数组arr，请将数组调整为依次相邻的数字，总是先<=、再>=的关系，并交替下去。比如数组中有五个数字，调整成[a,b,c,d,e],使之满足a<=b>=c<=d>=e。
>
> ```pseudocode
> 6
> 1 2 3 4 5 6
> output:
> 1 3 2 5 4 6
> ```

* 解析：1。先用非递归堆排序排好序，若数组长度为N是奇数，那么对【1，N-1】范围利用刚才的shuffle便是结果，如果N是偶数，则洗完拍之后，每两个数交换位置便是结果。

```c++
//先排序再说
void max_heapify(vector<int> &arr, int downPosition, int Magin)
{
    int dad = downPosition;
    int son = dad * 2 + 1;
    while(son <= Magin){
        if (son + 1 <= Magin && arr[son + 1] > arr[son])
            son++;
        if (arr[dad] > arr[son])
            return;
        else
        {
            swap(arr[dad], arr[son]);
            dad = son;
            son = dad * 2 + 1;
        }
    }
}

void heapSort(vector<int> &arr)
{
    if (arr.empty()) return;
    for (int i = arr.size() / 2 - 1; i >= 0; i--)
    {
        max_heapify(arr, i, arr.size() - 1);
    }
    for (int i = arr.size() -1; i > 0; i--)
    {
        swap(arr[0], arr[i]);
        max_heapify(arr, 0, i - 1);
    }
}

void rotate(vector<int> &arr, int l, int r)
{
    while(l < r)
    {
        swap(arr[l++], arr[r--]);
    }
}
// ab1234
void shuffle(vector<int> &arr, int start, int end)
{
    if (arr.empty() || start >= end) return;
    int size = end - start + 1;
    //find maximum block size 
    int k = 0;
    while(pow(3, k) - 1 <= size) k++;
    int residue = size - pow(3, k - 1) + 1; 
    if(residue != 0)
    {
        rotate(arr, start + (size - residue) / 2, start + size / 2 - 1);
        rotate(arr, start + size / 2, start + size - residue / 2 - 1);
        rotate(arr, start + (size - residue) / 2, start + size - residue / 2 - 1);
    }
    for (int i = 1; i < k; i++)
    {
        int time = start + pow(3, i - 1) - 1;
        int index = time;
        int nextIndex = -1, next_tmp = 0;
        int cur_tmp = arr[index];
        while(nextIndex != time)
        {
            if (index <= start + (size - residue) / 2 - 1)
                nextIndex = 2 * (index) - start + 1;
            else 
                nextIndex = 2 * (index - (size - residue) / 2) - start;
            next_tmp = arr[nextIndex];
            arr[nextIndex] = cur_tmp;
            index = nextIndex;
            cur_tmp = next_tmp;
        }
    }
    shuffle(arr, end - residue + 1, end);
}
int main()
{
    int N; cin >> N;
    vector<int> arr(N);
    for (int i = 0; i < N; i++)
        cin >> arr[i];
    heapSort(arr);
    if ((arr.size() & 1) == 1)
        shuffle(arr, 1, arr.size() - 1);
    else
    {
        shuffle(arr, 0, arr.size() - 1);
        for (int i = 0; i < arr.size() - 1; i+=2)
        {
            swap(arr[i], arr[i + 1]);
        }
    }
    for (auto target : arr)
        cout << target << " ";
    cout << endl;
}
```

#### 删除多余的字符得到字典序最小的字符串

> 给一个全是小写字母的字符串str，删除多余字符，使得每种字符只保留一个，并且让最终结果字符串字典序最小。
>
> ```pseudocode
> acbc
> output:
> abc
> ```

* 解析：先用一个map记录里面每个字符出现的数目，然后重新遍历字符串，同时减去map中对应字符出现值，当减为0 的时候，说明后续字符串中没有了这个字符；那么在当前遍历的串里找字符最小的作为串s的第一个元素；之后该定位字符的位置到减为0位置的元素在map中都加1，重新在该定位位置的下一处遍历，同时map中该定位字符不在考虑，设为-1。

```c++
int main()
{
    string s; cin >> s;
    map<char, int> table;
    for (auto  c : s)
        table[c]++;
    char minChar = 'z', minIndex = -1;
    int j = 0, k = 0;
    for (int i = 0; i < s.size(); i++)
    {
        if (table[s[i]] <= 0) continue;
        if (minChar >= s[i])
        {
            minChar = s[i];  minIndex = i;
        }
        if (--table[s[i]] == 0)
        {
            s[j++] = minChar;
            table[minChar] = -1;
            minChar = 'z';
            k = minIndex;
            //复原之前的位置;
            for (int r = minIndex + 1; r <= i; r++)
                if (table[s[r]] >= 0)
                    table[s[r]]++;
            i = k;
        }
    }
    for (int i = 0; i < j; i++)
        cout << s[i];
}
```



#### 数组中两个字符串的最小距离

> 给定一个字符串数组strs，再给定两个字符串str1和str2，返回在strs中str1和str2的最小距离，如果str1或str2为null，或不在strs中，返回-1。
>
> ```markdown
> 输入包含有多行，第一输入一个整数n（1 \leq n \leq 10^5）（1≤n≤105），代表数组strs的长度，第二行有两个字符串分别代表str1和str2，接下来n行，每行一个字符串，代表数组strs (保证题目中出现的所有字符串长度均小于等于10)。
> ```
>
> ```pseudocode
> 5
> QWER 666
> QWER
> 1234
> qwe
> 666
> QWER
> 
> output:
> 1
> ```
>
> ```
> 时间复杂度O（n），额外空间复杂度O（1）
> ```

* 解析：遍历一次，满足同时都在strs中的时候，记录下该次是否是最小距离。

```c++
#include <iostream>
#include <vector>
#include <string>
#include <climits>
#include <algorithm>
using namespace std;

int search(const vector<string> &target, const string &s1,const string &s2)
{
    int s1pos = -1, s2pos = -1;
    int minSize = INT_MAX;
    for (int i = 0; i < target.size(); i++)
    {
        if (target[i] == s1)
            s1pos = i;
        else if (target[i] == s2)
            s2pos = i;
        else
            continue;
        if (s1pos >=0 && s2pos >= 0)
            minSize = min(abs(s2pos - s1pos), minSize);
    }
    if (s1pos < 0 || s2pos < 0)
        return -1;
    else 
        return minSize;
}
int main()
{
    int N; cin >> N;
    string s1, s2; cin >> s1 >> s2;
    vector<string> target(N);
    for (int i = 0; i < N; i++)
        cin >> target[i];
    if (s1.empty() || s2.empty())
    {
        cout << -1;
        return 0;
    }
    cout << search(target, s1, s2) << endl;
}
```

#### 字符串的转换路径问题

> 给定两个字符串，记为start和to，再给定一个字符串列表list，list中一定包含to，list中没有重复的字符串。所有的字符串都是小写的。规定start每次只能改变一个字符，最终的目标是彻底变成to，但是每次变成新字符串必须在list中存在。请返回所有的最短的变换路径（按照字典序最小的顺序输出）。
>
> ```
> 如果存在转换的路径，请先输出“YES”，然后按照字典序最小的顺序输出所有路径。如果不存在请输出“NO”。
> ```
>
> ```
> 8
> abc cab
> cab
> acc
> cbc
> ccc
> cac
> cbb
> aab
> abb
> ```
>
> ```pseudocode
> YES
> abc -> abb -> aab -> cab
> abc -> abb -> cbb -> cab
> abc -> cbc -> cac -> cab
> abc -> cbc -> cbb -> cab
> ```

* 解析：

  步骤1：把start加入list，根据list生成每一个字符串的nexts信息。nexts具体指的是如果只改变一个字符，
  该字符串可能变成哪些字符串（它们应该包含在这n个串中，或者是start字符串，to字符串）。
  步骤2：nexts表等于一张图，每个字符串相当于图中的一个点，nexts信息相当于邻居节点，用宽度优先遍历求出
  每一个字符串到start的最短距离。
  步骤3：从start开始深度优先搜索，保证每一步走到的字符串的距离（步骤2计算的）是逐渐增加的，不然不是一条
  合理的路径，若到达了to，则记录下来。

  * 陷阱：传map或者vector之类的时候不要用`const`关键字，不然通过键索引出来结果会报错；

```c++
#include <iostream>
#include <set>
#include <map>
#include <vector>
#include <algorithm>
#include <string>
#include <queue>
using namespace std;

//step 1
map<string, vector<string>> 
getNext(const vector<string> &table, const string &start, const string &to)
{
    set<string> sset(table.begin(), table.end());
    sset.insert(start);
    map<string, vector<string>> nexts;
    string tmp2;
    for (auto tmp1 : sset)
    {
        vector<string> _next;
        for (size_t i = 0; i < tmp1.size(); i++)
        {
            for (char c = 'a'; c <= 'z'; c++)
            {
                if (tmp1[i] != c)
                {
                    tmp2 = tmp1; tmp2[i] = c;
                    if (sset.count(tmp2))
                    {
                        _next.push_back(tmp2);
                    }
                }
            }
        }
        nexts[tmp1] = _next;
    }
    return nexts;
}
//step 2
map<string, int>
getDistance(map<string, vector<string>> &nexts,string start)
{
    map<string, int> dis{{start, 0}}; // start is the initial node of graph nexts.
    set<string> sset; sset.insert(start); //check for strings that already visited.
    queue<string> que;
    que.push(start);
    string tmp;
    vector<string> _next;
    while(!que.empty())
    {
        tmp = que.front(); que.pop(); // visiting node 
        _next = nexts[tmp]; //neiborhood (nodes)strings.
        for (auto s : _next)
        {
            //之前宽度搜索没有遍历到的字符串才计算距离; 
            //only care about non-visited strings.
            if(!sset.count(s))
            {
                sset.insert(s);
                dis[s] = dis[tmp] + 1; //trick:labeled distance of strings to starts string. 
                que.push(s);
            }
        }
    }
    return dis;
}

//step 3
void
getShortestPaths(string start,string to, 
                map<string, vector<string>> &nexts, map<string, int> &dis,
                vector<string> path, vector< vector<string>> &res)
{
    path.push_back(start);
    if (to == start)
    {
        res.push_back(path);
    }
    else
    {
        vector<string> neibor = nexts[start];
        for (auto s : neibor)
        {
            if (dis[s] == dis[start] + 1)
            {
                getShortestPaths(s, to, nexts, dis, path, res);
            }
        }
    }
    return;
}

int main()
{
    int N; cin >> N;
    string start, to; cin >> start >> to;
    vector<string> table(N);
    for (int i = 0 ; i < N; i++)
        cin >> table[i];
    map<string, vector<string>> nexts;
    nexts = getNext(table, start, to);
    /* check nexts is right
    for (auto it = nexts.begin(); it != nexts.end(); it++)
    {
        cout << it->first << "  ";
        for (auto nx : it->second)
            cout << nx << ",";
        cout << endl;
    }*/
    map<string, int> distance;
    distance = getDistance(nexts, start);
    vector<string> path;
    vector< vector<string>> res;
    getShortestPaths(start, to, nexts, distance, path, res);
    if (res.empty())
        cout << "NO";
    else
    {
        cout << "YES" << endl;
        sort(res.begin(), res.end());
        for (int i = 0; i < res.size(); i++)
        {
            for (int j = 0; j < res[i].size(); j++)
            {
                cout << res[i][j];
                if ((j + 1) != res[i].size())
                    cout << " -> ";
            }
            cout << endl;
        }
    }
    return 0;
}
```

#### 添加最少的字符让字符串变为回文字符串（1）

> 给定一个字符串str，如果可以在str的任意位置添加字符，请返回在添加字符最少的情况下，让str整体都是回文字符串的一种结果。
>
> ```pseudocode
> AB
> output:
> ABA
> ```

```c++
#include <bits/stdc++.h>
using namespace std;

/*
step1:
dp[i][j]: represent str[i..j] need how much chars to be a plalindrome string
when str[i..j] has one char, dp[i][j] = 0;
when str[i..j] has two chars, if str[i] == str[j], dp[i][j] = 0; else dp[i][j]=1;
when str[i..j] has more than two chars, if str[i] == str[j], dp[i][j] = dp[i+1][j-1].
else dp[i][j] = 1 + min(dp[i+1][j],dp[i][j-1]);
dp only depend on down and left and left-down postion, thus we compute dp from down to up, left to right.

step2:

*/

//画一个小矩形图4*4,可以看出来i不超过j，因此只对上三角做计算，初始化字符数目为2的正好是边界；
//依赖关系也可以看出先计算出左列，再计算右列。
string
plaindrome(const string &s)
{
    if (s.size() <= 1) return s;
    //step1: get dp
    vector< vector<int>> dp(s.size(), vector<int>(s.size(), 0));
    for ( int j = 1; j < s.size(); j++)
    {
        dp[j - 1][j] = s[j - 1] == s[j] ? 0 : 1;
        for (int i = j - 2; i >= 0; i--)
            if (s[i] == s[j])
                dp[i][j] = dp[i + 1][j - 1];
            else
                dp[i][j] = 1 + min(dp[i + 1][j], dp[i][j - 1]);
    }
    //step2: get res
    int len = s.size() + dp[0][s.size() - 1];
    string res(len,' ');
    int  i = 0, j = s.size() - 1;
    int resl = 0, resr = len - 1;
    while(i <= j)
    {
        if (s[i] == s[j])
        {
            res[resl++] = s[i++];
            res[resr--] = s[j--];
        }
        else if (dp[i][j - 1] < dp[i + 1][j])
        {
            res[resl++] = s[j];
            res[resr--] = s[j--];
        }
        else
        {
            res[resl++] = s[i];
            res[resr--] = s[i++];
        }
    }
    return res;
}

int main()
{
    string s; cin >> s;
    cout << plaindrome(s) << endl;
}
```



#### 添加最少的字符让字符串变成回文串（2）

> 给定一个字符串str，再给定str的最长回文子序列字符串strlps, 请返回在添加字符最少的情况下，让str整体都是回文字符串的一种结果。进阶问题比原问题多了一个参数，请做到时间复杂度比原问题的实现低。
>
> ```
> 输出包含两行，第一行包含一个字符串代表str（ 1 \leq length_{str} \leq 5000）（1≤lengthstr≤5000）,第二行包含一个字符串，代表strips。
> ```
>
> ```pseudocode
> A1B21C
> 121
> ```
>
> ```pseudocode
> AC1B2B1CA
> ```
>
> ```
> str=“A1B21C"，strlps="121"，返回“AC1B2B1CA”或者“CA1B2B1AC”，总之，只要是添加的字符数最少，只返回其中一种结果即可。
> ```

* 解析：必然是一种结果，substr本身在原来的str中，所以str生成的回文串必然包含它们，而且不用额外的添加了，那么str变成的回文串长度为 2 * str.size() - substr.size() ，对应substr来把str的前后字符按个放到res里。

```c++
#include <bits/stdc++.h>
using namespace std;

string palindrome(string str, string substr)
{
    if (str.size() < 2) return str;
    string res(str.size() * 2 - substr.size(),' ');
    int resl = 0, resr = res.size() - 1; //index of result string
    int l = 0, r = str.size() - 1; //index of main str
    int subl = 0, subr = substr.size() - 1; // index of substr
    while(subl <= subr)
    {
        if (substr[subl] == substr[subr])
        {
            while(str[l] != substr[subl])
            {
                res[resl++] = str[l];
                res[resr--] = str[l++];
            }
            while(str[r] != substr[subr])
            {
                res[resl++] =  str[r];
                res[resr--] = str[r--];
            }
            res[resl++] = substr[subl++]; l++;
            res[resr--] = substr[subr--]; r--;
        }
    }
    return res;
}

int main()
{
    string str1, str2;
    cin >> str1 >> str2;
    cout << palindrome(str1, str2) << endl;
}
```

#### 括号字符串的有效性

> 给定一个字符串str，判断是不是整体有效的括号字符串(整体有效：即存在一种括号匹配方案，使每个括号字符均能找到对应的反向括号，且字符串中不包含非括号字符)。
>
> ```pseudocode
> 输出一行，如果str是整体有效的括号字符串，请输出“YES”，否则输出“NO”。
> ```

* 解析：做括号判断，直接遍历一次，然后非有效字符直接返回false；用一个balance记录左右字符出现的次数；当右括号太多的时候直接返回false；最后看左右符号是否相等。

```c++
#include <bits/stdc++.h>
using namespace std;

bool validParenthesis(const string &s)
{
    if (s.empty()) return false;
    int balance = 0;
    for (int i = 0; i < s.size(); i++)
    {
        if (s[i] != '(' && s[i] != ')')
            return false;
        if (s[i] == ')' && --balance < 0)
            return false;
        if (s[i] == '(')
            balance++;
    }
    return balance == 0;
}
int main()
{
    string s; cin >> s;
    cout << (validParenthesis(s)? "YES": "NO") << endl;
}
```



#### 括号字符串的最长有效长度

> 给定一个括号字符串str，返回最长的能够完全正确匹配括号字符字串的长度。
>
> ```
> (()())
> 6
> ())
> 2
> ```
>
> ```
> 时间复杂度O（n），额外空间复杂度O（n）。
> ```

* 解析：

  ```c++
  //dp[i]:str[0...i]中必须以str[i]结尾的最长有效字符串长度；
  //str[i] == '('那么dp[i]=0；
  //str[i] == ')'，比如 （（）（）（）），i - dp[i - 1] -1才到达第一个（符号的位置，发现匹配上；
  //比如 （）（（））；也就是i - dp[i - 1] -1匹配为(())，此时第一对（）也是有效考虑的情况了，所以加上它；
  //比如 ()(()))时，i - dp[i - 1] -1 < 0,就不考虑有效了。
  //
  ```

  

```c++
#include <bits/stdc++.h>
using namespace std;
int Dp(const string &s)
{
    if (s.size() < 2) return 0;
    vector<int> dp(s.size(), 0);
    int index = 0;
    int res = 0;
    for (int i = 1; i < s.size(); i++)
    {
        index = i - dp[i - 1] - 1;
        if (index >= 0 && s[i] == ')')
        {
            if (s[index] == '(')
            {
                dp[i] = dp[i - 1] + 2 + (index - 1 >= 0 ? dp[index - 1] : 0);
            }
        }
        res = max(res, dp[i]);
    }
    return res;
}

int main()
{
    string s; cin >> s;
    cout << Dp(s) << endl;
}
```

