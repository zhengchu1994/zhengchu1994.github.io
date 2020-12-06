---
title: CodeWar
mathjax: true
date: 2020-06-21 22:07:06
tags: Algorithm
categories: 算法与数据结构
visible: 
---





--------------





# CodeWar-WEEK 2

------------



#### 14. 最长公共前缀

* Scala

```scala
object Solution {
    def longestCommonPrefix(strs: Array[String]): String = {
        if(strs.isEmpty || strs.size == 0) return "";
        for {i <- 0  to strs(0).size 
            s <- strs}
        if (i >= s.size || strs(0).charAt(i) != s.charAt(i)) return s.substring(0, i);
        strs(0);
    }
}
```







#### 三数之和



思路：先排序，对排序后的每个数$a_i$，该数位置$i$后的数组进行双指针扫描。考虑排除重复的结果，若

与之前$a_{i-1}$相同，则跳过；对于双指针部分，尾部的元素$a_r$若与$a_{r+1}$相同，跳过。

* JAVA

```java
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        for(int i = 0; i < nums.length; i++){
            if(i - 1 >= 0 && nums[i] == nums[i - 1]) continue;
            int l = i + 1, r = nums.length - 1;
            while(l < r){
                while(r >= l && r + 1 < nums.length && nums[r] == nums[r + 1]) r--;
                if(l < r){
                    int target = nums[l] + nums[r];
                    if(target == -nums[i]){
                        res.add(Arrays.asList(nums[i], nums[l], nums[r]));
                        l++;
                        r--;
                    }else if( target > -nums[i]) r--;
                    else l++;
                }
            }
        }
        return res;
    }
}
```





#### \238. Product of Array Except Self

类型：脑经急转弯

```c++
class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums) {
        vector<int> res(nums.size(), 1);
        for(int i = 1; i < nums.size(); i++)
        {
            res[i] = res[i - 1] * nums[i - 1];
        }
        for(int j = nums.size() - 2, last = nums.back(); j >=0; j--)
        {
            res[j] *= last;
           a last *= nums[j];
        }
        return res;
    }
};
```





#### \347. Top K Frequent Elements



```c++
class Solution {
public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        if(nums.empty()) return {};
        unordered_map<int, int> hash;
        for(auto num : nums) hash[num]++;
        vector<pair<int, int>> heap;
        for(auto it : hash) heap.push_back({it.second, it.first});
        make_heap(heap.begin(), heap.end());
        vector<int> res;
        while(k--){
            res.push_back(heap.front().second);
            pop_heap(heap.begin(), heap.end());
            heap.pop_back();
        }
        return res;
    } 
};
```





####192. Word Frequency

```bash
# Read from the file words.txt and output the word frequency list to stdout.
cat words.txt | tr -s ' ' '\n' | sort | uniq -c | sort -r | awk  '{print $2, $1 }'
```



```bash
Very Nice solution. Since I am not very thorough with Shell, so I thought of executing every seperate block. Just sharing the output, since it might help someone visualize the solution after reading it.
➜ Desktop cat Words.txt
the day is sunny the the
the sunny is is
➜ Desktop cat Words.txt| tr -s ' ' '\n'
the
day
is
sunny
the
the
the
sunny
is
is
➜ Desktop cat Words.txt| tr -s ' ' '\n' | sort
day
is
is
is
sunny
sunny
the
the
the
the
➜ Desktop cat Words.txt| tr -s ' ' '\n' | sort | uniq -c
1 day
3 is
2 sunny
4 the
➜ Desktop cat Words.txt| tr -s ' ' '\n' | sort | uniq -c | sort -r
4 the
3 is
2 sunny
1 day
➜ Desktop cat Words.txt| tr -s ' ' '\n' | sort | uniq -c | sort -r | awk '{print $2, $1}'
the 4
is 3
sunny 2
day 1

```



#### \195. Tenth Line



```bash
# Read from the file file.txt and output the tenth line to stdout.
i=0
while ((i++ < 10))
do 
    read line
done < file.txt
echo $line
```



####  \713. Subarray Product Less Than K

* 思路：维持一个连乘小于k的最大长度子数组区间，若大于等于k时，从左边开始，减小窗口的大小。
* 合理性：遇到当前数x时，若可以放进区间[i，j]内，说明，该区间开头i，i+1。。。j, 都能与x组成一个连乘最大的结果，包括x自己本身是一个，所以每次满足放入区间，可以加上放入x后的区间长度个。

```c++
class Solution {
public:
    int numSubarrayProductLessThanK(vector<int>& nums, int k) {
        if(!nums.size()) return 0;
        int res = 0, pro = 1;
        for(int i = 0, j = 0; j < nums.size(); j++){
            pro *= nums[j];
            while(i <= j && pro >= k){
                pro /= nums[i++];
            }
            res += j - i + 1;
        }
        return res;
    }
};
```



#### \973. K Closest Points to Origin

* 思路：堆排序

```c++
class Solution {
public:
    vector<vector<int>> kClosest(vector<vector<int>>& points, int K) {
        vector<vector<int>> heap;
        make_heap(heap.begin(), heap.end());
        for(auto point : points){
            int dis = point[0] * point[0] + point[1] * point[1];
            heap.push_back({dis, point[0], point[1]});
            push_heap(heap.begin(), heap.end());
            if (heap.size() > K){
                pop_heap(heap.begin(), heap.end());
                heap.pop_back();
            }
        }
        vector<vector<int>> res;
        for(auto it : heap) res.push_back({it[1], it[2]});
        return res;
    }
};

//using priority_queue

class Solution {
public:
    vector<vector<int>> kClosest(vector<vector<int>>& points, int K) {
        priority_queue<vector<int>, vector<vector<int>>, compare> heap(points.begin(), points.end());
        vector<vector<int>> res;
        for(int i = 0; i < K; i++){
            res.push_back(heap.top());
            heap.pop();
        }
        return res;
    }
private:
    struct compare{
        bool operator() (vector<int>& p, vector<int>& q){
            return p[0] * p[0] + p[1] * p[1] > q[0] * q[0] + q[1] * q[1];
        }
    };
};
```



#### \215. Kth Largest Element in an Array



```c++
class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        partial_sort(nums.begin(), nums.begin() + k, nums.end(), greater<int>());
        return nums[k - 1];
    }
};
```





####  \692. Top K Frequent Words

* 定义一个枚举pair的堆：`priority_queue<pair<string, int>, vector<pair<string, int>>, Comp> heap;`

```c++
class Solution {
public:
    vector<string> topKFrequent(vector<string>& words, int k) {
        unordered_map<string, int> hash;
        for(auto word : words) hash[word]++;
        priority_queue<pair<string, int>, vector<pair<string, int>>, Comp> heap;
        for(const auto& h : hash){
            heap.push(h);
            if(heap.size() > k) heap.pop();
        }
        
        vector<string> res;
        while(heap.size()){
            res.insert(res.begin(), heap.top().first);
            heap.pop();
        }
        return res;
    }
private:
    struct Comp{
        Comp() {}
        ~Comp() {}
        bool operator()(const pair<string, int>& a, const pair<string, int>& b){
            return a.second > b.second || (a.second == b.second && a.first < b.first);
        }
    };
};
```



#### \501. Find Mode in Binary Search Tree

* Morris中序遍历：$O(1)$空间复杂度。
* Refer：https://www.acwing.com/blog/content/414/

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    vector<int> findMode(TreeNode* root) {
        if(!root) return {};
        if(!root->left && !root->right) return {root->val};
        vector<pair<int, int>> res;
        while(root){
            if(root->left == NULL)
            {
                if(res.size())
                {
                    if(res.back().first == root->val) res.back().second++;
                    else 
                        res.push_back({root->val, 1});
                }
                else
                    res.push_back({root->val, 1});
                root = root->right;
            }
            else
            {
                TreeNode* tmp = root->left;
                while (tmp->right != NULL && tmp->right != root)
                    tmp = tmp->right;
                
                if(tmp->right == NULL)
                {
                    tmp->right = root;
                    root = root->left;
                }
                else
                {
                    if(res.size())
                    {
                        if(res.back().first == root->val) res.back().second++;
                        else 
                            res.push_back({root->val, 1});
                    }
                    else
                        res.push_back({root->val, 1});
                    tmp->right = NULL;
                    root = root->right;
                }
            }
        }
        sort(res.begin(), res.end(), [](const pair<int,int>& a, const pair<int,int>& b)
             {return a.second > b.second;});
        vector<int> ans;
        int max = res[0].second;
        for(auto num : res)
            if(num.second < max) break;
            else
                ans.push_back(num.first);
        return ans; 
    }
        
};
```





#### \99. Recover Binary Search Tree

* Morris中序遍历
* 交换存在两种情况：
  * 相邻逆序：如（1, 2, 4, 3, 5）中的（4，3），交换即可；
  * 非相邻逆序：如（1， 5， 3，2，8 ）中的（5， 2）；即第一个逆序对的第一位，第二个逆序对的第的第二位，二者做交换。

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    void recoverTree(TreeNode* root) {
        TreeNode *tmp = NULL, *first=NULL, *second=NULL;
        while(root)
        {
            if(!root->left)
            {
                if(tmp && tmp->val > root->val)
                {
                    if(!second) first = tmp, second = root;
                    else second = root;
                }
                tmp = root;
                root = root->right;
            }
            else
            {
                TreeNode* next = root->left;
                while(next->right && next->right != root) next = next->right;
                if(!next->right)
                {
                    next->right = root;
                    root = root->left;
                }
                else
                {
                    tmp->right = NULL;
                    if(tmp && tmp->val > root->val)
                    {
                        if(!second) first = tmp, second = root;
                        else second = root;
                    }
                    tmp = root;
                    root = root->right;
                }
            }
        }
        swap(first->val, second->val);
    }
};
```

#### \679. 24 Game

* 递归、模拟：`next_permutation`：使用前先排序，不然枚举里没有这个排好序的序列。

```c++
class Solution {
public:
    bool judgePoint24(vector<int>& nums) {
        return judge24({nums.begin(), nums.end()});
    }
    static bool judge24(vector<double> nums){
        int n = nums.size();
        if(n == 1) return abs(nums[0] - 24) < 1e-10;
        
        sort(nums.begin(), nums.end());
        do
        {
            vector<double> temp(nums.begin(), nums.end() - 1);
            auto a = nums[n-1], b = nums[n-2];
            for(auto it : {a + b, b - a, a * b, a? b/a:0})
            {
                temp.back() = it;
                if(judge24(temp)) return true;
            }
        } while(next_permutation(nums.begin(), nums.end()));
        return false;
    }
};
```



python:

```python
class Solution:
    def judgePoint24(self, nums: List[int]) -> bool:
        if len(nums) == 1:
            return math.isclose(nums[0], 24)
        return any(self.judgePoint24([x] + test)
                   for a, b, *test in itertools.permutations(nums)
                   for x in [a+b, a-b, a * b, a and b/a]
        )
```





#### \662. Maximum Width of Binary Tree



* BFS：利用树的性质，假设节点带有标号，子节点是父节点的2倍或2倍加1，但直接算标号会溢出，所以每次可以把下一层的子节点一次放入队列，记录标号时先减去上一轮的标号。

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    int widthOfBinaryTree(TreeNode* root) {
        int res = 0;
        queue< pair<TreeNode*, int>> q;
        q.push({root, 0});
        while(!q.empty())
        {
            int n = q.size();
            res = max(res, q.back().second - q.front().second + 1);
            int offset = q.front().second;
            for(int i = 0; i < n; i++)
            {
                auto node = q.front(); q.pop();
                if(node.first->left) q.push({node.first->left, 2*(node.second - offset)});
                if(node.first->right) q.push({node.first->right, 2*(node.second-offset) + 1});
            }
        }
        return res;
    }
};
```



#### \508. Most Frequent Subtree Sum

* 递归搞定

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    int dfs(unordered_map<int, int>& hash,TreeNode* root,int& M)
    {
        if(!root) return 0;
        int left = dfs(hash, root->left, M);
        int right = dfs(hash, root->right, M);
        int key = left + right + root->val;
        hash[key]++;
        M = max(hash[key], M);
        return key;
    }
    vector<int> findFrequentTreeSum(TreeNode* root) {
        if(!root) return {};
        unordered_map<int, int> hash;
        int MAX = 0;
        dfs(hash, root, MAX);
        vector<int> res;
        for(auto h : hash)
            if(h.second == MAX) res.push_back(h.first);
        return res;
    }
};
```





#### \572. Subtree of Another Tree



* DFS~

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    bool dfs(TreeNode* s, TreeNode* t, bool eq){
        if(!s){
            if(!t) return true;
            return false;
        }
        if(!t) return false;
        if(s->val == t->val) if(dfs(s->left, t->left, !eq) && dfs(s->right, t->right, !eq)) return true;
        if(!eq) return dfs(s->left, t, eq) || dfs(s->right, t, eq);
        return false;
    }
    bool isSubtree(TreeNode* s, TreeNode* t) {
        return dfs(s, t, false);
    }
};
```





#### \503. Next Greater Element II

* 单调栈，但是得循环第二遍，因为要考虑逆序情况下的结果。

```c++
class Solution {
public:
    vector<int> nextGreaterElements(vector<int>& nums) {
        vector<int> res(nums.size(), 0);
        stack<int> st;
        for(int i = 0; i < nums.size();i++)
        {
            while(st.size() && nums[st.top()] < nums[i])
            {
                res[st.top()] = nums[i];
                st.pop();
            }
            st.push(i);
        }
        for(int i = 0; i < nums.size(); i++)
        {
            while(st.size() && nums[i] > nums[st.top()])
            {
                res[st.top()] = nums[i];
                st.pop();
            }
        }
        while(st.size())
        {
            res[st.top()] = -1;
            st.pop();
        } 
        return res;
    }
};
```



#### \556. Next Greater Element III

* `next_permutation`搞定

```c++
class Solution {
public:
    int nextGreaterElement(int n) {
        vector<int> temp;
        int m = n;
        while(m){
            int t = m % 10;
            m /= 10;
            temp.push_back(t);
        }
        int res = INT_MAX;
        sort(temp.begin(), temp.end());
        do{
            int cur = 0;
            for(auto num : temp)
            {
                if(cur > (INT_MAX - num)/10) break;
                cur = cur * 10 + num;
                
            }
            if(cur > n) res = min(res, cur);
        }while(next_permutation(temp.begin(), temp.end()));
        if(res == INT_MAX) res = -1;
        return res;
    }
};
```





#### \520. Detect Capital

```c++
class Solution {
public:
    bool detectCapitalUse(string word) {
        bool first_upper = (word[0] >= 'A' && word[0] <= 'Z');
        bool all_little = !first_upper, all_upper = first_upper;
        for(int i = 1; i < word.size(); i++)
        {
            if(word[i] >= 'a' && word[i] <= 'z') all_upper = false;
            else {all_little = false; first_upper = false;}
        }
        return all_upper || all_little || first_upper;
    }
};
```



####  \93. Restore IP Addresses

* 递归有讲究， 是在开头为字母为0的时候递归一次。。

```c++
class Solution {
public:
    vector<string> restoreIpAddresses(string s) {
        vector<string> res;
        dfs(s, 0, 0, "", res);
        return res;
    }
    void dfs(const string& s, int st, int step, string cand, vector<string>& res){
        if(st == s.size() && step == 4){
            cand.erase(cand.end() - 1);
            res.push_back(cand);
            return;
        }
        if((s.size() - st) > (4 - step) * 3) return;
        if((s.size() - st) < (4 - step)) return;
        int num = 0;
        for(int i = st; i < st + 3; i++)
        {
            num = num * 10 + (s[i] - '0');
            if(num > 255) return;
            cand += s[i];
            dfs(s, i + 1, step + 1, cand + '.', res);
            if(num == 0) return; //让开头为0的模式就一个0，所以只搜索一次
        }
    }
};
```





----------







#     CodeWar-WEEK 3

-----------





#### \696. Count Binary Substrings

* 每次排查一段同样字符的长度，与上一段不同样字符的长度做比较，取最小的值就是新增的配对结果书数目。

```C++
class Solution {
public:
    int countBinarySubstrings(string s) {
        int first = 0, second = 0;
        int i = 0, ans = 0;
        while(i < s.size()){
            int j = i;
            if(!first) 
            {
                while(j< s.size() && s[i] == s[j]) {first++;j++;}
                i = j;
            }
            while(j < s.size() && s[i] == s[j]) {second++; j++;}
            i = j;
            ans += min(first, second);
            first = second; second = 0;
        }
        return ans;
    }
};
```







#### 墨水

> 问题：有A（红）B（蓝）两瓶墨水，从A中取一勺倒到B中，混匀后，再从B中取等量墨水，倒到A中。问，是A中蓝墨水多还是B中红墨水多？



x，y分别为两瓶墨水的容量，取$x/tx$倒入$y$中后，y的容量是$y + x/tx$，

再取$(y + x/t)/ty$倒入$x$中后，

x最终的容量是$x - x/tx + (y + x/tx)/ty$。

y最终的容量是$y + x/tx - (y + x/tx)/ty$。

其中，$x/tx = (y + x/tx)/ty = T$。

假设$y = x$，必有$ty>tx$，

x加入y后，y中x占的比例为$T/(y + T)$，

那么x在$(y + x/tx)/ty$中有：$(y + T)/ty * T/(y + T) = T/ty$，

那么y在$(y + x/tx)/ty$中有：$(y + T)/ty * [1- T/(y + T)] = y/ty$

x加入$(y + x/tx)/ty$后，y在x中有$y/ty$ .

y中拿走$(y + x/tx)/ty$后，x在y中有$T - T/ty$。

$T - T/ty - y/ty => T*ty - T - y = (y + x/tx)/ty * ty - x/tx - y = 0$

：所以一样多。





####  \95. Unique Binary Search Trees II

* 思路：用第i个数字作为root，划分成左子树与右子树；所有的左子树与右子树都存在数组里，与根结点的所有组合就是结果。

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    vector<TreeNode*> generateTrees(int n) {
        if(n <= 0) return vector<TreeNode*>{};
        return dfs(1, n);
    }
    vector<TreeNode*> dfs(int st, int end){
        if(st>end) return vector<TreeNode*> {NULL};
        vector<TreeNode*> ans;
        for(int i = st; i <= end; i++)
        {
            auto left = dfs(st, i - 1);
            auto right = dfs(i + 1, end);
            
            for(int j = 0; j < left.size(); j++)
                for(int k = 0; k < right.size(); k++)
                {
                    TreeNode *root = new TreeNode(i);
                    root->left = left[j];
                    root->right = right[k];
                    ans.emplace_back(root);
                }
        }
        return ans;
    }
};
```



python:

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def generateTrees(self, n: int) -> List[TreeNode]:
        def tree(st, ed):
            return [TreeNode(root, l, r) for root in range(st, ed)
                   for l in tree(st, root)
                   for r in tree(root+1, ed)] or [None]
        return tree(1, n + 1) if n else []
```





####  \241. Different Ways to Add Parentheses

* 骚操作。。。枚举子问题。。

```c++
class Solution {
public:
    vector<int> diffWaysToCompute(string input) {
        vector<int> res;
        for(int i = 0; i < input.size(); i++)
        {
            if(ispunct(input[i])){
                for(auto a : diffWaysToCompute(input.substr(0, i)))
                    for(auto b : diffWaysToCompute(input.substr(i+1)))
                    {
                        char c = input[i];
                        res.push_back(c == '+'? a + b : c == '-' ? a-b: a * b);
                    }
            }
        }
        return res.size() ? res : vector<int>{stoi(input)};
    }
};
```





#### \23. Merge k Sorted Lists

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        priority_queue<pair<ListNode*, int>, vector<pair<ListNode*, int>>, Comp> pq;
        int i = 0;
        while(i < lists.size()) 
        {
            if(!lists[i]) lists.erase(lists.begin() + i);
            else i++;
        }
        for(int i = 0; i < lists.size(); i++)
        {
            pq.push({lists[i], i});
        }
        ListNode * dummy = new ListNode(-1);
        ListNode * tmp = dummy;
            
        while(!pq.empty()){
            auto node = pq.top(); pq.pop();
            tmp->next = node.first; tmp = tmp->next;
            lists[node.second] = lists[node.second]->next;
            if(lists[node.second] != nullptr) 
                pq.push({lists[node.second], node.second});
        }
        return dummy->next;
    }
    struct Comp{
        Comp() {};
        ~Comp() {};
        bool operator()(const pair<ListNode*,int>& a, const pair<ListNode*, int>& b){
            return a.first->val > b.first->val;
        } 
    };
};
```





#### \373. Find K Pairs with Smallest Sums

```python
"""
heapq.merge(*iterables, key=None, reverse=False)
Merge multiple sorted inputs into a single sorted output (for example, merge timestamped entries from multiple log files). Returns an iterator over the sorted values.
"""

#k路归并
class Solution:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        streams = map(lambda a: ([a + b, a, b] for b in nums2), nums1)
        stream = heapq.merge(*streams)
        return [nums[1:] for nums in itertools.islice(stream, k)]
      
# 支队长候选结果进行堆插入
class Solution:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        queue = []
        def insert(i, j):
            if i < len(nums1) and j < len(nums2):
                heapq.heappush(queue, [nums1[i] + nums2[j], i, j])
        insert(0, 0)
        ans = []
        while queue and len(ans) < k:
            _, i, j = heapq.heappop(queue)
            insert(i, j + 1)
            if j == 0:
                insert(i + 1, 0)
            ans.append([nums1[i], nums2[j]])
        return ans
```





####  \719. Find K-th Smallest Pair Distance

* **二分**：第k小的数，等价于寻找至少有k-1个数小于等于x且最小的x。

```c++
class Solution {
public:
    int smallestDistancePair(vector<int>& nums, int k) {
        sort(nums.begin(), nums.end());
        int l = 0;
        int r = nums.back() - nums[0];
        for(int cnt = 0; l < r; cnt = 0)
        {
            int mid = l + r >> 1;
            
            for(int i = 0, j = 0; i < nums.size(); i++)
            {
                while(j < nums.size() && nums[j] <= nums[i] + mid) j++;
                cnt += j - i - 1;
            }
            
            if(cnt < k) l = mid + 1;
            else r = mid;
        }
        return r;
    }
};
```



#### \668. Kth Smallest Number in Multiplication Table

* 二分：乘积矩阵的有序性进行二分。

```c++
class Solution {
public:
    int findKthNumber(int m, int n, int k) {
        int l = 1, r = m * n;
        for(int cnt = 0; l < r; cnt = 0)
        {
            int mid = l + r >> 1;
            for(int i = 1; i <= n; i++)
            {
                int t = min(mid / i, m);
                cnt += t;
            }
            
            if(cnt < k) l = mid + 1;
            else r = mid;
        }
        return r;
    }
};
```



#### \378. Kth Smallest Element in a Sorted Matrix

* 二分查找的正确性：count(x)表示比x小的个数，二分找count(x) == k-1的情况下，x最小的值，显然有一些数不存在matrix中，他们在matrix中永远找不到k-1个小于的值；只有存在于matrix中，才有可能找到k-1个小于的数字在matrix中。

```c++
class Solution {
public:
    int kthSmallest(vector<vector<int>>& matrix, int k) {
        int n = matrix.size();
        int l = matrix[0][0], r = matrix.back().back();
        for(int cnt = 0; l < r; cnt = 0)
        {
            int mid = l + r >> 1;
            for(int i = 0, j = n - 1; i < matrix.size(); i++, j = n-1)
            {
                while(j>=0 && matrix[i][j] > mid) j--;
                cnt += j + 1;
            }
            if(cnt < k) l = mid + 1;
            else r = mid;
            
        }
        return r;
    }
};
```





####  \786. K-th Smallest Prime Fraction

* 二分、优先队列都可以做。。还有$O(n)$的做法，看不懂看不懂

```c++
class Solution {
public:
    vector<int> kthSmallestPrimeFraction(vector<int>& A, int K) {
        int n = A.size();
        double l = 0.0, r = 1.0, mid;
        while(l < r)
        {
            mid = (l + r) * 0.5;
            int cnt = 0;
            for(int i = 1; i < n; i++)
            {
                double it = A[i] * mid;
                auto p = upper_bound(A.begin(), A.end(), it) - A.begin();
                cnt += p;
            }
            if(cnt == K) break;
            if(cnt > K) r = mid;
            else l = mid;       
        }        

        vector<int> ans;
        pair<int, int> p{-1, -1};
        for(int i = 1; i < n; i++)
        {
            int tmp = mid * A[i];
            for(int j = i - 1; j >= 0; j--)
            {
                if(A[j] <= tmp)
                {
                    if(p.first == -1 || p.second * A[j] > p.first * A[i])
                        p = {A[j], A[i]};
                    break;
                }
            }
        }
        ans.push_back(p.first);
        ans.push_back(p.second);
        return ans;
    }
};
```



####  \377. Combination Sum IV

* Dp(k): 价值为k的方案数目，（在整个nums中无限取）

```c++
class Solution {
public:
    //dp(j)：价值为j的方案数目
    int combinationSum4(vector<int>& nums, int target) {
        int n = nums.size();
        vector<unsigned long long> dp(target + 1, 0);
        dp[0] = 1;
        for(int i = 1; i <= target; i++)
        {
            for(auto num : nums)
            {
                if(i >= num) dp[i] += dp[i - num];
            }
        }
        return dp[target];
    }
};
```



#### \349. Intersection of Two Arrays

```c++
class Solution {
public:
    vector<int> intersection(vector<int>& nums1, vector<int>& nums2) {
        set<int> s(nums1.begin(), nums1.end());
        vector<int> res;
        for(auto num : nums2)
            if(s.erase(num))
                res.push_back(num);
        return res;
    }
};
```





#### \174. Dungeon Game

* DP:依赖关系是从右下角到左上角～

```c++
class Solution {
public:
    int calculateMinimumHP(vector<vector<int>>& dungeon) {
        int n = dungeon.size(), m = dungeon[0].size();
        for(int i = n - 1; i >= 0; i--)
        {
            for(int j = m - 1; j >= 0; j--)
            {
                if(i == n - 1 && j == m - 1) 
                    dungeon[i][j] = max(1, 1 - dungeon[i][j]);
                else if(i == n - 1)
                        dungeon[i][j] = max(dungeon[i][j + 1] - dungeon[i][j], 1);
                else if(j == m - 1)
                        dungeon[i][j] = max(dungeon[i + 1][j] - dungeon[i][j], 1);
                else
                    dungeon[i][j] = min(
                        max(dungeon[i + 1][j] - dungeon[i][j], 1),
                        max(dungeon[i][j + 1] - dungeon[i][j], 1));
            }
        }
        return dungeon[0][0];
    }
};
```







####  \216. Combination Sum III

* DFS: 防止冗余出现，每次都缩小起点st与后面的坐标。

```c++
class Solution {
public:
    vector<vector<int>> ans;
    int N, K;
    vector<vector<int>> combinationSum3(int k, int n) {
        N = n, K = k;
        vector<int> used(10, 0), cur;
        dfs(0, cur, 1, used, 0);
        return ans;
    }
    void dfs(int k, vector<int>& cur,int st, vector<int>& used, int sum)
    {
        if(k == K)
        {
            if(sum == N)
                ans.push_back(cur);
            return;
        }
        for(int i = st; i <= 9; i++)
        {
            if(used[i - 1]) continue;
            used[i - 1] = 1;
            cur.push_back(i);
            dfs(k + 1, cur, i + 1, used, sum + i);
            cur.pop_back();
            used[i - 1] = 0;
        }
    }
};
```



#### \64. Minimum Path Sum

```c++
class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        int n = grid.size(), m = grid[0].size();
        vector<vector<int>> dp(n, vector<int>(m, 0));
        
        for(int i = 0; i < n; i ++)
            for(int j = 0; j < m; j++)
            {
                if(i == 0 && j == 0) dp[i][j] = grid[i][j];
                else if(i == 0)
                    dp[i][j] = dp[i][j-1] + grid[i][j];
                else if(j == 0)
                    dp[i][j] = dp[i - 1][j] + grid[i][j];
                else
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
            }
        return dp[n - 1][m - 1];
    }
};
```





####  \502. IPO

* 堆：思路是维持一个候选集合的堆。

```python
class Solution:
    def findMaximizedCapital(self, k: int, W: int, Profits: List[int], Capital: List[int]) -> int:
        pq = []
        candidate = sorted(zip(Capital, Profits))[::-1]
        for _ in range(k):
            while candidate and candidate[-1][0] <= W:
                heapq.heappush(pq, -candidate.pop()[1])
            if pq:
                W -= heapq.heappop(pq)
        return W
```

c++:

```c++
class Solution {
public:
    int findMaximizedCapital(int k, int W, vector<int>& Profits, vector<int>& Capital) {
        int n = Profits.size();
        vector<pair<int, int>> vec;
        priority_queue<int> heap;
        for(int i = 0; i < n; i++)
            vec.push_back({Capital[i], Profits[i]});
        sort(vec.begin(), vec.end());
        int i = 0;
        while(k--)
        {
            while(i < n && vec[i].first <= W)
                heap.push(vec[i++].second);
            if(heap.size())
            {
                W += heap.top();
                heap.pop();
            }
        }
        return W;
    }
};
```



----------

# CodeWar-WEEK 4

---------



#### \1219. Path with Maximum Gold

* 范围比较小，就DFS。

```c++
class Solution {
public:
    int n, m, ans; 
    int dx[4] = {-1, 1, 0, 0};
    int dy[4] = {0, 0, -1, 1};
    int getMaximumGold(vector<vector<int>>& grid) {
        n = grid.size(), m = grid[0].size(), ans = 0;
        
        for(int i = 0; i < n; i++)
            for(int j = 0; j < m; j++)
            {
                vector<vector<int>> res(n, vector<int>(m, 0));
                if(grid[i][j] != 0)
                {
                    dfs(res, grid, i, j, grid[i][j]);
                }
            }
        return ans;
    }
    void dfs(vector<vector<int>>& res, const vector<vector<int>>& grid, int x, int y, int cal)
    {
        res[x][y] = 1;
        ans = max(cal, ans);
        for(int i = 0; i < 4; i++)
        {
            int a = dx[i] + x, b = dy[i] + y;
            if(a >= 0 && a < n && b >= 0 && b < m && !res[a][b] && grid[a][b])
            {
                res[a][b] = 1;
                dfs(res, grid, a, b, cal + grid[a][b]);
                res[a][b] = 0;
            }
        }
    }
};
```





#### \498. Diagonal Traverse

* 类型：模拟

```c++
class Solution {
public:
    int n, m;
    vector<int> findDiagonalOrder(vector<vector<int>>& matrix) {
        if(matrix.empty()) return {};
        n = matrix.size(), m = matrix[0].size();
        int i = 0, j = 0;
        bool up = true;
        vector<int> res;
        while(i < n && j < m)
        {
            res.push_back(matrix[i][j]);
            if(up)
            {
                if(i == 0 && j != m - 1){j++; up = !up;}
                else if(j == m - 1){i++; up = !up;}
                else{i--;j++;}
            }
            else
            {
                if(j == 0 && i != n - 1){i++; up = !up;}
                else if(i == n - 1){j++; up = !up;}
                else{i++; j--;}
            }
        }
        return res;
    }
};
```





#### \67. Add Binary 

* 模拟

  ```c++
  class Solution {
  public:
      void cal(string& res, int idx, const string& s, int& pre)
      {
          while(idx < s.size())
          {
              int num = stoi(s.substr(idx, 1));
              int tmp = (num + pre) % 2;
              pre = (num + pre) / 2;
              res += to_string(tmp);
              idx++;
          }
      }
      string addBinary(string a, string b) {
          string res = "";
          reverse(a.begin(), a.end());
          reverse(b.begin(), b.end());
          int pre = 0, i = 0;
          while(i < a.size() && i < b.size())
          {
              int ai = stoi(a.substr(i, 1));
              int bi = stoi(b.substr(i, 1));
              int tmp = (ai + bi + pre) % 2;
              pre = (ai + bi + pre) / 2;
              res += to_string(tmp);
              i++;
          }
          if(i < a.size()) cal(res, i, a, pre);
          if(i < b.size()) cal(res, i, b, pre);
          if(pre) res += to_string(pre);
          reverse(res.begin(), res.end());
          return res;
      }
  };
  ```



#### \892. Surface Area of 3D Shapes

* 先计算每一个小方块的表面积，再减去挨在一起的时候应该去掉的面积。

```c++
class Solution {
public:
    int surfaceArea(vector<vector<int>>& grid) {
        int n = grid.size(), m = grid[0].size();
        int res = 0;
        for(int i = 0; i < n; i++)
        {
            for(int j = 0; j < m; j++)
            {
                if(grid[i][j])
                {
                    res += grid[i][j] * 4 + 2;
                    if(i != 0) res -= min(grid[i-1][j], grid[i][j]) * 2;
                    if(j != 0) res -= min(grid[i][j-1], grid[i][j]) * 2;
                }
            }
        }
        return res;
    }
};
```



#### \1025. Divisor Game

* 类型：博弈论，迭代加深记录结果避免超时。

```c++
class Solution {
public:
    vector<int> dp;
    bool divisorGame(int N) {
        dp = vector<int>(N + 1, -1);
        return choose(N);
    }
    bool choose(int n)
    {
        if(dp[n] != -1) return dp[n];
        if(n == 1) return false;
        for(int i = 1; i < n; i++)
            if(n % i == 0)
            {
                dp[n - i] = choose(n - i);
                if(!dp[n - i]) return true;
            }
        return false;
    }
};
```



####  \902. Numbers At Most N Given Digit Set

* 数位dp：
* 考虑从D中选择数位都小于目标N的数位长度K的可选方案有$(\text{len}(D))^k，k=1，2，3，...,K-1 $;
* 当D中选择数位的长度等于K时，从最低位考虑，低位的选择小于对应N的位数上的数字时，方案数为D此时的低位到最末尾的位数个数s，在D中选s个放入数位中补上，等于$(\text{len}(D) )^s$。

```c++
class Solution {
public:
    int atMostNGivenDigitSet(vector<string>& D, int N) {
        string num = to_string(N);
        int n = D.size(), m = num.size();
        vector<int> dp(m + 1, 0);
        dp[m] = 1; //最后一位相等时，算一个数
        for(int i = m - 1; i >= 0; i--)
        {
            int dig = stoi(num.substr(i, 1));
            for(auto c: D)
            {
                if(stoi(c) < dig) 
                    dp[i] += pow(n, m - i - 1);
                else if(stoi(c) == dig)
                    dp[i] += dp[i + 1];
            }
        }
        
        for(int i = 1; i < m; i++)
            dp[0] += pow(n, i);
        return dp[0];
    }
};
```



####  \504. Base 7

* 题目意思：把一个数转换为7进制（抱歉没看到题目，打扰了）

```c++
class Solution {
public:
    string convertToBase7(int num) {
        string res = "";
        int n = abs(num);
        while(n)
        {
            res = to_string(n % 7) + res;
            n /= 7;
        }
        return num > 0 ? res : num < 0 ? "-" + res : "0";
    }
};
```



####  \1455. Check If a Word Occurs As a Prefix of Any Word in a Sentence

* 模拟

```python
class Solution {
public:
    int isPrefixOfWord(string sentence, string searchWord) {
        int word_cnt = 1;
        int j = 0;
        for(int i = 0; i < sentence.size() && j < searchWord.size(); i++)
        {
            word_cnt += sentence[i] == ' ';
            if(sentence[i] == searchWord[j])
                if(j == 0 && (i == 0 || sentence[i-1] == ' '))
                        j++;
                else if(j > 0)
                    j++;
            else j = 0;
        }
        if(j == searchWord.size()) return word_cnt;
        return -1;
    }
};
```



####  \506. Relative Ranks

* pair排序

```c++
class Solution {
public:
    vector<string> findRelativeRanks(vector<int>& nums) {
        int n = nums.size();
        priority_queue<pair<int, int> > pq;
        for(int i = 0; i < n; i++)
            pq.push(make_pair(nums[i], i));
        
        vector<string> res(n, "");
        int i = 0;
        while(pq.size())
        {
            auto t = pq.top(); pq.pop();
            if(i >= 3) res[t.second] = to_string(i + 1);
            else if(i == 0) res[t.second] = "Gold Medal";
            else if(i == 1) res[t.second] = "Silver Medal";
            else if(i == 2) res[t.second] = "Bronze Medal";
            i++;
        }
        return res;
    }
};
```



####  \1497. Check If Array Pairs Are Divisible by k

* 一开始做复杂了。。。超时。。但是复习了下STL..，也算收获？QAQ

  ```c++
  can.erase(find(can.begin(), can.end(), val));
  int sum = accumulate(arr.begin(), arr.end(), start);
  ```



* 先把数组大小放缩在k以内，对于负数`i`，添加它对应的`k-i`。对于本身能够整除k的数，需要的也是能整除k的数，所以需要偶数个0。

```c++
class Solution {
public:
    bool canArrange(vector<int>& arr, int k) {
        if(k == 1) return true;
        vector<int> can(k, 0);
        for(auto num : arr)
        {
            int t = num % k;
            if(t < 0) 
                t = k + t;
            can[t] += 1;
        }        
        for(int i = 1; i < can.size(); i++)
            if(can[i] != can[k-i]) return false;
        return can[0] % 2 == 0;
    }
};
```



####  \557. Reverse Words in a String III

```c++
class Solution {
public:
    string reverseWords(string s) {
        string res = "", word = "";
        int i = 0;
        while(i < s.length())
        {
            if(s[i] == ' ') res += word + " ", word = "";
            else
                word = s[i] + word;
            i++;
        }
        res += word;
        return res;
    }
};
```

in-place：

```c++
class Solution {
public:
    string reverseWords(string s) {
        size_t front = 0;
        for(int i=0; i <= s.length(); i++)
        {
            if(i == s.length() || s[i] == ' ')
            {
                reverse(&s[front], &s[i]);
                front = i + 1;
            }
        }
        return s;
    }
};
```



#### \541. Reverse String II

* 比`static_cast<size_t>` 快。

```c++
class Solution {
public:
    string reverseStr(string s, int k) {
        int n = s.size();
        for(int i = 0; i < n; i+= 2*k) 
            reverse(s.begin() + i, s.begin() + min(i + k, n));
        return s;
    }
};
```



#### \451. Sort Characters By Frequency

* hash表：
* c++ `lambda`表达式：

```c++

class Solution {
public:
    string frequencySort(string s) {
        unordered_map<char, int> hash;
        
        for(auto c : s)
            hash[c]++;
        vector<pair<char, int>> res(hash.begin(), hash.end());     
        sort(res.begin(), res.end(), [&](auto a, auto b)
             {return a.second > b.second;});
        string ans = "";
        for(auto it : res)
            while(it.second--) ans += it.first;
        return ans;
    }
};
```



#### \387. First Unique Character in a String

* Hash

```c++
class Solution {
public:
    int firstUniqChar(string s) {
        unordered_map<char, int> hash;
        for(int i = 0; i < s.size(); ++i)
            if(hash[s[i]] == 1 || hash[s[i]] == -1) hash[s[i]] = -1;
            else
                hash[s[i]]++;
        
        for(int i = 0; i < s.size(); ++i)
            if(hash[s[i]] == 1) return i;
        return -1;
    }
};
```



#### \43. Multiply Strings

* 去年头条面我的题，现在想想面试官真好。。

```c++
class Solution {
public:
    string add(string a, string b)
    {
        int pre = 0, i = a.size() - 1, j = b.size() - 1;
        string res = "";
        while(i >= 0 || j >= 0 || pre)
        {
            int n = pre;
            if(j >= 0 ) { n += (b[j] - '0'); j--;}
            if(i >= 0 ) { n += (a[i] - '0'); i--;}
            res = to_string(n % 10) + res;
            pre = n / 10;
        }
        return res;
    }
    string multiply(string num1, string num2) {
        if(num2 == "0" || num1 == "0") return "0";
        string res = "0";
        for(int i = num2.size() - 1; i >= 0; i--)
        {
            string tmp = "";
            int pre = 0;
            for(int j = num1.size() - 1; j >= 0; j--)
            {
                int n = (num1[j] - '0') * (num2[i] - '0') + pre;
                tmp = to_string(n % 10) + tmp;
                pre = n / 10;
            }
            if(pre) tmp = to_string(pre) + tmp;
            int pos = num2.size() - 1 - i;
            while(pos--) tmp += "0";
            res = add(res, tmp);
        }
        return res;
    }
};
```



#### \878. Nth Magical Number

* 二分 + gcd

* 能被`a`或者`b`整除的数，有多少个`<=x`：
  * `x / a + x / b - x / lcm`，其中`lcm`是a和b的最小公倍数。

```c++
class Solution {
public:
    int gcd(int a, int b)
    {
        if(b) return gcd(b, a % b);
        return  a;
    }
    long mod = 1e9 + 7;
    int nthMagicalNumber(int N, int A, int B) {
        int g = gcd(A, B);
        int lcm = A * B / g;
        long l = 2, r = 1e14;
        while(l < r)
        {
            long mid = l + r >> 1;
            if(mid / A + mid / B - mid / lcm  < N) l = mid + 1;
            else r = mid;
        }
        return r % mod;
    }
};
```





####  \1109. Corporate Flight Bookings

* 前缀和；

* 在一段区间[a, b]内加上一个数x，等价于在a处加x，b + 1处 -x，
* 计算时res[i] += res[i - 1]；使得总是区间[a, b]内计算被加上x；而[b+1...]不会被影响

```c++
class Solution {
public:
    vector<int> corpFlightBookings(vector<vector<int>>& bookings, int n) {
        vector<int> res(n + 1, 0);
        for(int i = 0; i < bookings.size(); i++)
        {
            res[bookings[i][0] - 1] += bookings[i][2];
            res[bookings[i][1]] -= bookings[i][2];
        }
        for(int i = 1; i < n; i++)
            res[i] += res[i - 1];
        res.pop_back();
        return res;
    }
};
```





####  \922. Sort Array By Parity II

* 双指针

```c++
class Solution {
public:
    vector<int> sortArrayByParityII(vector<int>& A) {
        for(int odds = 1, evens = 0; odds < A.size() && evens < A.size(); )
        {
            while(odds < A.size() && A[odds] % 2 == 1) odds += 2;
            while(evens < A.size() && A[evens] % 2 == 0) evens += 2;
            if(odds >= A.size() || evens >= A.size()) return A;
            swap(A[odds], A[evens]);
        }
        return A;
    }
};
```



#### \1346. Check If N and Its Double Exist

* hash

```c++
class Solution {
public:
    bool checkIfExist(vector<int>& arr) {
        set<int> st(arr.begin(), arr.end());
        auto low = lower_bound(arr.begin(), arr.end(), 0);
        auto high = upper_bound(arr.begin(), arr.end(), 0);
        if(low != high) return true;
        for(auto n : st)
            if(n != 0 && st.count(n * 2) != 0) return true;
        return false;
    }
};
```



#### \106. Construct Binary Tree from Inorder and Postorder Traversal

* 模拟

* 对中序遍历数组依据根结点做分割；

  

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    TreeNode* Build(vector<int>& inorder, int ist, int iend, vector<int>& postorder, int pst, int pend)
    {
        if(ist > iend || pst > pend) return nullptr;
        int n = postorder[pend];
        TreeNode* root = new TreeNode(n);
        int r = ist;
        while(r <= iend && inorder[r] != n) r++;
        root->left = Build(inorder, ist, r - 1, postorder, pst, pst + r - ist - 1);
        root->right = Build(inorder, r + 1, iend, postorder, pst + r - ist, pend - 1);
        return root;
    }
    TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {
        return Build(inorder, 0, inorder.size() - 1, postorder,  0, postorder.size()  - 1);
    }

};
```



#### \778. Swim in Rising Water

* BFS

```c++
class Solution {
public:
    int dx[4] = {0, 0, -1, 1};
    int dy[4] = {-1, 1, 0, 0};
    int swimInWater(vector<vector<int>>& grid) {
        int n = grid.size(), m = grid[0].size();
        vector<vector<int>> st(n, vector<int>(m, 0));
        priority_queue<vector<int>, vector<vector<int>>, Compare> pq;
        pq.push({grid[0][0], 0, 0});
        st[0][0] = 1;
        int res = 0;
        while(pq.size())
        {
            auto node = pq.top(); pq.pop();
            res = max(res, node[0]);
            if(node[1] == n - 1 && node[2] == m - 1) break;
            for(int i = 0; i < 4; i++)
            {
                int a = node[1] + dx[i], b = node[2] + dy[i];
                if(a>=0 && a < n && b >=0 && b < m && !st[a][b])
                {
                    st[a][b] = 1;
                    pq.push({grid[a][b], a, b});
                }
            } 
        }
        return res;
    }
struct Compare {
    bool operator()(vector<int> const & a, vector<int> const & b)
    { return a[0] > b[0]; }
    };
};
```





# CodeWar-WEEK 5

####  \581. Shortest Unsorted Continuous Subarray

1、Sort time; O(n) time complexity

```python
class Solution:
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        return len("".join(("."," ")[m == n] for m, n in zip(nums,sorted(nums))).strip())
```

2、O(n) time、O(1) time complexity

* 双指针搜索：出现拐点的最左右两端

```c++
class Solution {
public:
    int findUnsortedSubarray(vector<int>& nums) {
        int n = nums.size(), end = -2, st = -1;
        int mi = nums[n - 1], mx = nums[0];
        for(int i = 1; i < n; i++)
        {
            mx = max(mx, nums[i]);
            mi = min(mi, nums[n - i - 1]);
            if(nums[i] < mx) end = i;
            if(nums[n - i - 1] > mi) st = n- i - 1;
        }
        return end - st + 1;
    }
};
```



#### \1338. Reduce Array Size to The Half

* 一般题

```c++
class Solution {
public:
    int minSetSize(vector<int>& arr) {
        int n = arr.size() / 2;
        unordered_map<int, int> hash;
        for(auto num : arr) hash[num]++;
        vector<pair<int, int>> nums(hash.begin(), hash.end());
        sort(nums.begin(), nums.end(), [&](const pair<int, int>& a,
                                          const pair<int, int>& b)
             {return a.second > b.second;});
        int res = 0, count = 0;
        for(auto num : nums)
        {
            count += num.second;
            res++;
            if(count >= n) return res;
        }
        return n * 2;
    }
};
```





#### \741. Cherry Pickup

* 1、想的是DP一次取最大的值，然后记录该取值的路径，按照路径把已经采样的格子置为0；然后重新dp第二次；该方法失败。

```c++

//方法1:贪心失败；
/*失败样例
[[1,1,1,1,0,0,0],

[0,0,0,1,0,0,0],

[0,0,0,1,0,0,1],

[1,0,0,1,0,0,0],

[0,0,0,1,0,0,0],

[0,0,0,1,0,0,0],

[0,0,0,1,1,1,1]]

*/
class Solution {
public:
    int n, m, INF = -1e9;
    int cherryPickup(vector<vector<int>>& grid) {
        n = grid.size(), m = grid[0].size();
        // path[i][j] =  1 or 2: right = 1, down = 2;
        vector<vector<int>> path(n, vector<int>(m, 0));
        vector<vector<int>> dp(n, vector<int>(m, 0));
        DP(grid, path, dp);
        for(int i = 0; i < n; i++)
        {
            for(int j = 0; j < m; j++)
            {
                cout << dp[i][j] << " ";
            }
            cout << endl;
        }
        int res = dp[n-1][m-1];
        if(res == INF) return 0;
        grid[0][0] = 0;
        int i = n - 1, j = m - 1;
        while(i != 0 || j != 0)
        {
            grid[i][j] = 0;
            if(path[i][j] == 2) i--;
            else j--;
        }
        cout << count << endl;
        for(auto & vec : dp)
            fill(vec.begin(), vec.end(), 0);
        for(int i = 0; i < n; i++)
        {
            for(int j = 0; j < m; j++)
            {
                cout << grid[i][j] << " ";
            }
            cout << endl;
        }
        DP(grid, path, dp);
        return res + dp[n - 1][m - 1];
    }
    void DP(vector<vector<int>>& grid, vector<vector<int>>& path, vector<vector<int>>& dp)
    {
        for(int i = 0; i < n; i ++)
        {
            for(int j = 0; j < m; j++)
            {
                if(grid[i][j] == -1) dp[i][j] = INF;
                else if(i == 0 || j == 0)
                {
                    dp[i][j] = grid[i][j];
                    if(j > 0)
                    {
                        if(dp[i][j - 1] == INF) dp[i][j] = INF;
                        else {
                            dp[i][j] += dp[i][j - 1]; //right1
                            path[i][j] = 1;
                        }
                    }
                    if(i > 0)
                    {
                        if(dp[i - 1][j] == INF) dp[i][j] = INF;
                        else {
                            dp[i][j] += dp[i-1][j]; //down2
                            path[i][j] = 2;
                        }
                    }
                }
                else
                {
                    if(dp[i][j - 1] != INF) dp[i][j] = dp[i][j - 1];
                    if(dp[i - 1][j] != INF)
                    {
                        if(dp[i - 1][j] >= dp[i][j]) //down2
                        {
                            path[i][j] = 2;
                            dp[i][j] = dp[i - 1][j];
                        }
                        else
                            path[i][j] = 1; //right1
                        dp[i][j] += grid[i][j];
                    }
                    if(dp[i][j] == 0) dp[i][j]= INF;
                }
            }
        }
    }
};
```

2、dfs+memo：path1 + path2 同时进行有效搜索的空间里取结果最大的值。

```c++
class Solution {
public:
    int n, m, INF = -1e8;
    vector<vector<vector<int>>> memo;
    int cherryPickup(vector<vector<int>>& grid) {
        n = grid.size(), m = grid[0].size();
        memo = vector(n, vector(m, vector<int>(n, INF)));
        //search the total cherries:(x1, y1) + (x2, y2)
        return max(0,dfs(grid, 0, 0, 0, 0));
    }
    int dfs(vector<vector<int>> &grid, int x1, int y1, int x2, int y2)
    {
        if(x1 >= n || y1 >= m || x2 >= n || y2 >= m || grid[x1][y1] == -1 ||
          grid[x2][y2] == -1) 
            return INF;
        if(memo[x1][y1][x2] != INF) return memo[x1][y1][x2];
        int cherries = grid[x1][y1];
        if(x1 != x2) cherries += grid[x2][y2];
        if(x1 == n - 1 && y1 == m - 1) return cherries;
        if(x2 == n - 1 && y2 == m - 1) return cherries;
        int down_right = dfs(grid, x1 + 1, y1, x2, y2 + 1);
        int down_down = dfs(grid, x1 + 1, y1, x2 + 1, y2);
        int right_down = dfs(grid, x1, y1 + 1, x2 + 1, y2);
        int right_right = dfs(grid, x1, y1 + 1, x2, y2 + 1);
        cherries += max(max(down_right, down_down), max(right_down, right_right));
        memo[x1][y1][x2] = cherries;
        return cherries;
    }
};
```







####  \315. Count of Smaller Numbers After Self



```c++
/*
Runtime: 328 ms, faster than 5.08% of C++ online submissions for Count of Smaller Numbers After Self.
Memory Usage: 10.5 MB, less than 71.34% of C++ online submissions for Count of Smaller Numbers After Self.
*/
class Solution {
public:
    vector<int> countSmaller(vector<int>& nums) {
        if(nums.empty()) return {};
        int n = nums.size();
        vector<int> res(n, 0);
        vector<int> arr;
        arr.push_back(nums.back());
        for(int i = n - 2; i >= 0; i--)
        {
            auto low = lower_bound(arr.begin(), arr.end(), nums[i]);
            res[i] = low - arr.begin();
            arr.insert(low,nums[i]);
        }
        return res;
    }
};
```



####  acwing数据流中的中位数

* 算法1：二分

```python
from bisect import bisect
class Solution:
    def __init__(self):
        self.nums = []
        
    def insert(self, num):
        """
        :type num: int
        :rtype: void
        """
        self.nums.insert(bisect(self.nums, num), num)
        
    def getMedian(self):
        """
        :rtype: float
        """
        m = len(self.nums) // 2
        return (self.nums[m] + self.nums[m - 1]) / 2 if len(self.nums) % 2 == 0 else self.nums[m]
```

* 算法2：双堆

```python
import heapq
class Solution:
    def __init__(self):
        self.miheap = []
        self.mxheap = []
    def insert(self, num):
        """
        :type num: int
        :rtype: void
        """
        if not self.miheap and not self.mxheap:
            heapq.heappush(self.mxheap, -num)
        elif len(self.miheap) > len(self.mxheap):
            if self.miheap[0] <= num:
                heapq.heappush(self.mxheap, -heapq.heappop(self.miheap))
                heapq.heappush(self.miheap, num)
            else:
                heapq.heappush(self.mxheap, -num)
        elif len(self.miheap) < len(self.mxheap):
            if -self.mxheap[0] >= num:
                heapq.heappush(self.miheap, -heapq.heappop(self.mxheap))
                heapq.heappush(self.mxheap, -num)
            else:
                heapq.heappush(self.miheap, num)
        else:
            if num < self.miheap[0]:
                heapq.heappush(self.mxheap, -num)
            else:
                heapq.heappush(self.miheap, num)
    def getMedian(self):
        """
        :rtype: float
        """
        if len(self.miheap) > len(self.mxheap):
            return self.miheap[0]
        elif len(self.mxheap) > len(self.miheap):
            return -self.mxheap[0]
        else:
            return (-self.mxheap[0] + self.miheap[0]) / 2
```





#### acwing把数组排成最小的数--python-sort

第一种做法：

* 长度不一的时候，用短的元素的前缀填补到短元素的后面，然后比较；
* 若一直相等，做相加的数值比较；
```python
from functools import cmp_to_key
#letterlist.sort(key=cmp_to_key(keyL))
class Solution(object):
    def printMinNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: str
        """
        def func(x, y):
            if len(x) > len(y):
                t = y[:] + y[:len(x) - len(y)]
                for i, j in zip(x, t):
                    if i > j:
                        return 1
                    elif i < j:
                        return -1
                xt, tx = int(x + y), int(y + x)
                if xt > tx:
                    return 1
                elif xt < tx:
                    return -1
                return 0
            elif len(x) <= len(y):
                t = x[:] + x[:len(y) - len(x)]
                for i, j in zip(t, y):
                    if i > j:
                        return 1
                    elif i < j:
                        return -1
                ty, yt = int(x + y), int(y + x)
                if ty > yt:
                    return 1
                elif ty < yt:
                    return -1
                return 0
        nums = sorted(list(map(str,nums)), key = cmp_to_key(func))
        return "".join(map(str, nums))
```
优化第一种方法直接拼接进行比较：

```python
from functools import cmp_to_key
class Solution(object):
    def printMinNumber(self, nums):
        """ 
        :type nums: List[int]
        :rtype: str
        """
        def func(x, y):
            t, g = x + y, y + x
            for i, j in zip(t, g):
                if i > j:
                    return 1
                elif i < j:
                    return -1
            return 0
        nums = sorted(list(map(str,nums)), key = cmp_to_key(func))
        return "".join(map(str, nums))
```





####  Acwing序列化二叉树--python-DFS与栈模拟

1. idea：以传tuple的形式分割左右子树与根结点；

```
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        if not root:
            return "."
        left = self.serialize(root.left)
        right = self.serialize(root.right)
        s = (left, right, root.val)
        return s

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        if not data or data[-1] == '.': return None
        left, right, val = data
        root = TreeNode(val)
        root.left = self.deserialize(left)
        root.right = self.deserialize(right)
        return root
```

2. 栈模拟

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        if not root:
            return []
            
        q = [root]
        res = []
        while q:
            node = q.pop(0)
            if not node:
                res.append('null')
            else:
                res.append(node. val)
                q.append(node.left)
                q.append(node.right)
        return res

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        if len(data) == 0: return None
        if data[0] =='null': return None
        
        root = TreeNode(data.pop(0))
        q = [root]
        LF = True
        while data:
            tmp = data.pop(0)
            if tmp != 'null':
                node = TreeNode(tmp)
                q.append(node)
                if LF:
                    q[0].left = node
                else:
                    q[0].right = node
            if not LF:
                q.pop(0)
            LF = not LF
        return root
```



#### acwing 数字序列中某一位的数字--Python

用count计算当前按数位计算有多少个数，num计算10进制数，当num每次跃升muti*10的倍数，位数大小p增加，muti更新，直到计算到要求的答案附近。

```python
class Solution(object):
    def digitAtIndex(self, n):
        """
        :type n: int
        :rtype: int
        """
        count, num, p, muti = 0, 0, 1, 1
        while count + p * 10 < n:
            count += p * 10
            num += 10 
            if num >= muti * 10:
                p += 1
                muti = num
        # num多加了一个数，所以size为0，计算是在第一个数上，也就是num本身
        # size不为0，num + size也正要取数位的数上
        size, pos = (n - count) // p, (n - count) % p
        num += size
        return int(str(num)[pos])
```



#### AcWing 62. 丑数



一个数N分解为$p_1^{(x1)}*p_2^{(x2)}..*p_n^{(xn)}$，$p_i$只能是1、2、3、5时便是丑数；

为求第n个丑数，那么每次都从集合中选出最小的丑数，分别乘以2，3，5后放入集合中，然后删除掉该最小的丑数，重复n-1次后，下一个最小的丑数便是第n个丑数。

```python
class Solution(object):
    def getUglyNumber(self, n):
        """
        :type n: int
        :rtype: int
        """
        ugly = set([1])
        while n - 1 > 0:
            n -= 1
            num = min(ugly)
            ugly.remove(num)
            ugly.add(num * 2)
            ugly.add(num * 3)
            ugly.add(num * 5)
        return min(ugly)

作者：zc2077
链接：https://www.acwing.com/activity/content/code/content/267067/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```



####  \66. 两个链表的第一个公共结点

若是有一个公共节点x，那么后面的长度都一样，那么把长链表先走几步使得和短链表一样长，那么x前面的长度也一样了，一起走就到x了

```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def findFirstCommonNode(self, headA, headB):
        """
        :type headA, headB: ListNode
        :rtype: ListNode
        """
        a, b = 0, 0
        da, db = headA, headB
        while da:
            a += 1
            da = da.next
        while db:
            b += 1
            db = db.next
        da, db = headA, headB
        while b > a:
            b -= 1
            db = db.next
        while a > b:
            a -= 1
            da = da.next
        while da and db:
            if da == db:
                return da
            da, db = da.next, db.next
        return None
```



#### AcWing 79. 滑动窗口的最大值

- 单调栈：递减的单调栈；栈内记录数组下标，当栈的底部坐标在滑动窗口之外， 则弹出；同时维持栈内递减；当遍历的元素有k个时，开始放置答案。
```python
class Solution(object):
    def maxInWindows(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        res, count, st = [], 0, []
        for i, num in enumerate(nums):
            if st and i - st[0] >= k:
                st.pop(0)
            while st and nums[st[-1]] <= num:
                st.pop()
            st.append(i)
            count += 1
            if count >= k:
                res.append(nums[st[0]])
        return res
```





#### \80. 骰子的点数

（我看我一个月前写的代码，这谁写的。。。）

* `f[n]`是投`n`次骰子得到的点数序列，有
* $f[n] = (a_{n}, a_{n+1},\dots, a_{6 * n})$；
* $f[n-1] = (b_{n - 1}, b_{n},\dots, b_{6 * (n - 1)})$
* 得到`a_{n} = b_{n - 1}(n=1)`，`a_{n + 1} = b_{n-1}(n=2) + b_{n}(n=1)`....其中（n=1）代表投第n次时骰子数值为1，因此投第`n`次的点数是上一次前最多6个数的和。

```python
class Solution(object):
    def numberOfDice(self, n):
        """
        :type n: int
        :rtype: List[int]
        """
        init = [1, 1, 1, 1, 1, 1]
        for _ in range(2, n + 1):
            res, window, count = [], [], 0
            for num in init:
                if window and len(window) == 6:
                        count -= window.pop(0)
                window.append(num)
                count += num
                res.append(count)
            while window:
                count -= window.pop(0)
                if count:
                    res.append(count)
            init = res
        return init
```





#### AcWing 88. 树中两个结点的最低公共祖先

* 对当前节点r的左子树、右子树递归，若左右都查找到了p、q节点，那么r便是最低公共祖先；
* 若只有一边查到了p、q节点，说明，p、q节点中的其中一个是最低公共祖先，直接返回该节点；
* 边界：空节点，返回空指针；
* 边界：若当前节点属于p、q中的某一个，直接返回该节点；
```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if not root:
            return None
        if root in [p, q]:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if left and right:
            return root
        if left:
            return left
        if right:
            return right
```





#### \82. 圆圈中最后剩下的数字

1. 模拟

```python
class Solution(object):
    def lastRemaining(self, n, m):
        """
        :type n: int
        :type m: int
        :rtype: int
        """
        nums = list(range(n))
        while len(nums) != 1:
            index = m % len(nums) - 1 if m % len(nums) != 0 else len(nums) - 1
            nums = nums[index + 1:] + nums[:index]
        return nums[0]
```

2， 递归

![image-20200713104251787](https://tva1.sinaimg.cn/large/007S8ZIlly1ggp5dcvtxwj31bc0rqtf3.jpg)

```python
# n>=1，最后一次为f(1, m) = 0，取余数从2开始；
class Solution(object):
    def lastRemaining(self, n, m):
        """
        :type n: int
        :type m: int
        :rtype: int
        """
        start = 0
        for i in range(2, n + 1):
            start = (start + m) % i
        return start
```



####  AcWing 52. 数组中出现次数超过一半的数字

* 重复的数，count++；
* 不一样的数，count--；

```python
class Solution(object):
    def moreThanHalfNum_Solution(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        count, res = 1, nums[0]
        for num in nums[1:]:
            if res != num:
                count -= 1
                if count == 0:
                    res = num
                    count = 1
            else:
                count += 1
        return res
```



####  \49. 二叉搜索树与双向链表

- 在中序遍历的过程中，链接上一个节点pre与当前遍历节点cur：当前节点cur的左指针指向pre，若pre非空，pre的右指针指向cur；
- pre变为cur；
- 对cur的右子树进行相同操作；

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
    TreeNode* pre = NULL;
    TreeNode* convert(TreeNode* root) {
        dfs(root);
        while(root && root->left)
            root = root->left;
        return root;
    }
    TreeNode* dfs(TreeNode* root){
        if(!root) return NULL;
        if(root->left) dfs(root->left);
        
        root->left = pre;
        if(pre) pre->right = root;
        pre = root;
        dfs(root->right);
    }
    
};
```

python：

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def convert(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        pre = None
        def dfs(root):
            if not root:
                return None
            if root.left:
                dfs(root.left)
            nonlocal pre
            root.left = pre
            if pre: pre.right = root
            pre = root
            dfs(root.right)
        dfs(root)
        while root and root.left:
            root = root.left
        return root
```





--------

# CodeWar-WEEK 6



####  AcWing 81. 扑克牌的顺子

* 模拟

```python
class Solution(object):
    def isContinuous(self, numbers):
        """
        :type numbers: List[int]
        :rtype: bool
        """
        king = 0
        numbers.sort()
        if len(numbers) < 5: return False
        for i, num in enumerate(numbers):
            if num == 0:
                king += 1
            if i:
                t = num - numbers[i - 1]
                if num and t == 0: return False
                if t != 1 and numbers[i - 1] == 0:
                    continue
                if t - 1 <= king:
                    king = king - t + 1
                else:
                    return False
        return True
```



#### \87. 把字符串转换成整数

```python
class Solution(object):
    def strToInt(self, str):
        """
        :type str: str
        :rtype: int
        """
        
        str = str.strip()
        if len(str) == 0: return 0
        sym, end = -1, len(str)
        for i, c in enumerate(str):
            if c in ['+', '-']:
                sym = i
            elif c > '9' or c < '0':
                end = i
                break
        if end == 0 or sym + 1 == end: return 0
        t = sym
        while t + 1 < len(str) and str[t + 1] == '0':
            t += 1
        res = int(str[t + 1: end])
        res = -res if str[sym] == '-' else res
        return 2**31-1 if res > 2**31 -1 else -2**31 if res < -2**31 else res  
```





####  \86. 构建乘积数组

* 模拟

```python
class Solution(object):
    def multiply(self, A):
        """
        :type A: List[int]
        :rtype: List[int]
        """
        if not A:
            return A
        res, pro = [1], A[0]
        for num in A[1:]:
            res.append(pro)
            pro *= num
        pro = 1
        for i in range(len(A) - 1, -1, -1):
            res[i] *= pro
            pro *= A[i]
        return res
```





####  AcWing 85. 不用加减乘除做加法

```c++
class Solution {
public:
    int add(int num1, int num2){
        while(num2){
            int carry = unsigned(num1 & num2) << 1;
            num1 = num1 ^ num2;
            num2 = carry;
        }
        return num1;
    }
};
```



#### AcWing 83. 股票的最大利润

*  dp：只要当前股价比之前最小的股价大，就计算一下，看是不是最大利润

```c++
class Solution {
public: 
    int CONST = 1e9;
    int maxDiff(vector<int>& nums) {
        if(nums.size() <= 1) return 0;
        int res = 0;
        int mi = CONST;
        for(int i = 0; i < nums.size(); i++)
        {
            mi = min(nums[i], mi);
            if(nums[i] > mi) 
                res = max(res, nums[i] - mi);
        }
        return res;
    }
};
```



####  AcWing 59. 把数字翻译成字符串

* dp: 对于一位字符的出现，都可以把前面的字符可以翻译的次数加上，再考虑前一位字符与当前字符是否满足翻译条件；

```python
class Solution:
    def getTranslationCount(self, s):
        """
        :type s: str
        :rtype: int
        """
        dp = [0 for _ in range(len(s))]
        dp[0], pre = 1, int(s[0])
        for i, c in enumerate(s[1:], 1):
            num = int(c) + pre * 10
            dp[i] += dp[i - 1]
            if pre != 0 and num <=25:
                if i >= 2:
                    dp[i] += dp[i - 2]
                else:
                    dp[i] += 1
            pre = int(c)
        return dp[-1]
```





#### AcWing 61. 最长不含重复字符的子字符串

* 双指针：若右指针指向重复数字，移动左指针；当左指针等于右指针所指重复数字，删除、移动左指针、并退出；
```python
class Solution:
    def longestSubstringWithoutDuplication(self, s):
        """
        :type s: str
        :rtype: int
        """
        i, j, res, dic = 0, 0, 0, {}
        while j < len(s):
            if s[j] in dic:
                while True:
                    if s[i] == s[j]:
                        dic.pop(s[i])
                        i += 1
                        break
                    dic.pop(s[i])
                    i += 1
            dic[s[j]] = j
            res = max(res, j - i + 1)
            j += 1
        return res
```



#### \64. 字符流中第一个只出现一次的字符



```python
from collections import deque, defaultdict
class Solution:  
    def __init__(self):
        self.dic = defaultdict(int)
        self.q = deque()
        
    def firstAppearingOnce(self):
        """
        :rtype: str
        """
        return '#' if not self.q else self.q[0]
   
    def insert(self, char):
        """
        :type char: str
        :rtype: void
        """
        self.dic[char] += 1
        if self.dic[char] > 1:
            # while的原因，扫描到后面的时候，已经放入q里的元素不再是只出现一次
            #因此while去掉已经重复的元素
            while self.q and self.dic[self.q[0]] > 1:
                self.q.popleft()
        else:
            self.q.append(char)
```



#### \65. 数组中的逆序对

- 利用归并排序，在比较的时候计算逆序对的数量。
- 需要一个额外的tmp数组，将归并的结果暂时保存起来，归并结束后，tmp数组是一个有序的数组，将该数组拷贝到原先排序的数组上。

```python
class Solution(object):
    def inversePairs(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        def mergeSort(st, end):
            if st >= end: return 0
            mid = st + end >> 1
            res = mergeSort(st, mid) + mergeSort(mid + 1, end)
            i, j, tmp = st, mid + 1, []
            while i <= mid and j <= end:
                if nums[i] <= nums[j]:
                    tmp.append(nums[i])
                    i += 1
                else:
                    tmp.append(nums[j])
                    j += 1
                    res += mid - i + 1
            while i <= mid:
                tmp.append(nums[i])
                i += 1
            while j <= end:
                tmp.append(nums[j])
                j += 1
            for num in tmp:
                nums[st] = num
                st += 1
            return res
        return mergeSort(0, len(nums) - 1)
```





#### AcWing 74. 数组中唯一只出现一次的数字

```python
class Solution(object):
    def findNumberAppearingOnce(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        res = 0
        for i in range(31, -1, -1):
            count = 0
            for num in nums:
                if num >> i & 1:
                    count += 1
            if count % 3 == 1:
                res = (res * 2) + 1
            else:
                res *= 2
        return res
```





#### AcWing 56. 从1到n整数中1出现的次数  

```python
class Solution(object):
    def numberOf1Between1AndN_Solution(self, n):
        """
        :type n: int
        :rtype: int
        """
        res, n = 0, str(n)
        for i, v in enumerate(n):
            if i: res += pow(10,len(n) - i - 1) * int(n[:i])
            if int(v) == 1:
                if i == len(n) - 1:
                    res += 1
                else:
                    res += 1 + int(n[i + 1:])
            elif int(v) > 1:
                res += 10 ** (len(n) - i - 1)
        return res

作者：zc2077
链接：https://www.acwing.com/activity/content/code/content/383035/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。。
```





#### \179. Largest Number

* 跟上面排序成一个最小数组相反的问题。

```c++
class Solution {
public:
    string largestNumber(vector<int>& nums) {
        vector<string> vec;
        for(auto num : nums) vec.push_back(to_string(num));
        sort(vec.begin(), vec.end(), [&](auto& a, auto&b)
             {string al = a + b, bl = b + a; return al > bl;});
        string res = "";
        for(auto s: vec)
            res += s;
        if(res[0] == '0') res = "0";
        return res;
    }
};
```



#### word search

* dfs

```c++
class Solution {
public:
    int row, col;
    string wd;
    bool exist(vector<vector<char>>& board, string word) {
        row = board.size();
        col = row ? board[0].size(): 0;
        wd = word;
        if(!row || !col) return false;
        for(int i = 0; i < row; i++)
            for(int j = 0; j < col; j++)
                if(board[i][j] == word[0])
                    if(dfs(board, i, j, 0)) return true;
        return false;
    }
    
    bool dfs(vector<vector<char>>& board, int i, int j, int pos)
    {
        if(board[i][j] != wd[pos]) return false;
        if(pos + 1 == wd.size()) return true;
        char c = board[i][j];
        board[i][j] = '.';
        if(i>=1 && dfs(board, i-1, j, pos + 1)) return true;
        if(j>=1 && dfs(board, i, j-1, pos + 1)) return true;
        if(i+1<row && dfs(board,i+1, j, pos + 1)) return true;
        if(j+1 < col && dfs(board, i, j + 1, pos + 1)) return true;
        board[i][j] = c;
        return false; 
    }
};
```



#### [Word Search II](https://leetcode.com/problems/word-search-ii/)

* Trie树上做DFS

```python
class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        root = {}
        for word in words:
            node = root
            for char in word:
                node = node.setdefault(char, {})
            node['#'] = True
        row, col = len(board), len(board[0])
        def dfs(i, j, pre, root, visited):
            if '#' in root: 
                res.add(pre)
            for _i, _j in ((0, 1), (1, 0), (-1, 0), (0, -1)):
                a, b = _i + i, _j + j
                if -1 < a and a < row and -1 < b and b < col and board[a][b] in root and (a, b) not in visited:
                    dfs(a, b, pre + board[a][b], root[board[a][b]], visited | {(a, b)})
            
        res = set()
        for i in range(row):
            for j in range(col):
                if board[i][j] in root:
                    dfs(i, j, board[i][j], root[board[i][j]], {(i, j)})
        return res
```

2. 

```c++
class TrieNode{
    public:
        string words = "";
        vector<TrieNode*> nodes;
        TrieNode():nodes(26, nullptr){}
};

class Solution {
public:
    vector<string> res;
    int row, col;
    vector<string> findWords(vector<vector<char>>& board, vector<string>& words) {
        row = board.size();
        col = row ? board[0].size() : 0;
        if(!row || !col) return res;
        
        TrieNode* root = new TrieNode();
        for(int i = 0; i < words.size(); i++)
        {
            TrieNode* cur = root;
            string tmp = "";
            for(int j = 0; j < words[i].size(); j++)
            {
                tmp += words[i][j];
                int n = words[i][j] - 'a';
                if(!cur->nodes[n]) cur->nodes[n] = new TrieNode();
                cur = cur->nodes[n];
            }
            cur->words = tmp;
        }
        
        for(int i = 0; i < row; i++)
            for(int j = 0; j < col; j++)
                dfs(board, root, i, j);
        return res;
    }
    
    void dfs(vector<vector<char>>& board, TrieNode* root, int i, int j)
    {
        char c = board[i][j];
        if(c == '.' || !root->nodes[c - 'a']) return;
        root = root->nodes[c - 'a'];
        if(root->words != ""){
            res.push_back(root->words);
            root->words = "";
            // 不return；继续搜索
        }
        board[i][j] = '.';
        if(i >= 1) dfs(board, root, i - 1, j);
        if(j >= 1) dfs(board, root, i, j - 1);
        if(i + 1 < row) dfs(board, root, i + 1, j);
        if(j + 1 < col) dfs(board, root, i, j + 1);
        board[i][j] = c;
    }
};
```



#### \217. Contains Duplicate

```python
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        return len(set(nums)) < len(nums)
```

#### \219. Contains Duplicate II

```python
class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        mp = {}
        for i, num in enumerate(nums):
            if num in mp and i - mp[num] <= k: return True
            mp[num] = i
        return False
```

#### \220. Contains Duplicate III

waiting～





####  \29. Divide Two Integers

思路ref：https://www.acwing.com/solution/content/87/

```c++
class Solution {
public:
    int divide(int dividend, int divisor) {
        if(dividend == INT_MIN && divisor == -1) return INT_MAX;
        const int HALF_INT_MIN = INT_MIN / 2;
        int x = dividend, y = divisor;
        bool sign = (x > 0) ^ (y > 0);
        
        if(x > 0) x = -x;
        if(y > 0) y = -y;
        
        vector<pair<int, int>> can;
        for(int t1 = y, t2 = -1; t1 >= x; t1 += t1, t2 += t2)
        {
            can.emplace_back(t1, t2);
            if(t1 < HALF_INT_MIN) break;
        }
        int ans = 0;
        for(int i = can.size() - 1; i >= 0; i--)
            if(can[i].first >= x)
            {
                x -= can[i].first;
                ans += can[i].second;
            }
        return sign > 0 ? ans : -ans;
        
    }
};
```

