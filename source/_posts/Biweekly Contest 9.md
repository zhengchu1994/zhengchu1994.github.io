---
title: LeetCode-Biweekly Contest 9
mathjax: true
date: 2019-09-20 21:53:32
tags: [Leetcode, Algorithm]
categories: 算法与数据结构
visible:
---


## Biweekly Contest 9



#### 1196. How Many Apples Can You Put into the Basket

>You have some apples, where `arr[i]` is the weight of the `i`-th apple.  You also have a basket that can carry up to `5000` units of weight.
>
>Return the maximum number of apples you can put in the basket.
>
>```pseudocode
>Input: arr = [100,200,150,1000]
>Output: 4
>Explanation: All 4 apples can be carried by the basket since their sum of weights is 1450.
>```

* 解析：

  ```c++
  class Solution {
  public:
      int maxNumberOfApples(vector<int>& arr) {
          sort(arr.begin(),arr.end());
          int res = 0;
          int weight = 5000;
          for (auto num : arr)
          {
              if (weight - num > 0)
              {
                  res++;
                  weight -= num; 
              }
              else
                  break;
          }
          return res;
      }
  };
  ```

  



#### 1197. Minimum Knight Moves

> In an **infinite** chess board with coordinates from `-infinity` to `+infinity`, you have a **knight** at square `[0, 0]`.
>
> A knight has 8 possible moves it can make, as illustrated below. Each move is two squares in a cardinal direction, then one square in an orthogonal direction.
>
> Return the minimum number of steps needed to move the knight to the square `[x, y]`.  It is guaranteed the answer exists.
>
> **Constraints:**
>
> - `|x| + |y| <= 300`

* 解析：到达【x,y】的最小步数，等价于到达 【||x|| ,||y||】，即正值的位置。用图的BFS解。

我的两个答案都超时：

```python
import queue
class Solution:
    def minKnightMoves(self, x: int, y: int) -> int:
        x, y = abs(x), abs(y)
        xstep = [2,  2,  1,  1,  -2, -2,  -1, -1]
        ystep = [1, -1,  2, -2,  -1,  1,  -2,  2]
        s = set()
        q = queue.Queue()
        pos = (0, 0)
        s.add(pos)
        q.put(list(pos) + [0])
        while not q.empty():
            out = q.get()
            for xs, ys in zip(xstep, ystep):
                pos = xs + out[0], ys + out[1]
                if -5 <= abs(pos[0]) + abs(pos[1]) <= 302:
                    if pos == (x, y):
                        return out[2] + 1
                    if pos not in s:
                        s.add(pos)
                        q.put(list(pos) + [out[2] + 1])
```

这个也错误（）：

```c++
const int MAXL = 302;
const int STEP = 8;

class Solution { 
    //走位
    const int xstep[STEP] = {2,  2,  1,  1,  -2, -2,  -1, -1};
    const int ystep[STEP] = {1, -1,  2, -2,  -1,  1,  -2,  2};
    
    struct edge
    {
        edge(int x, int y, int step): v(x), u(y), steps(step) {}
        int v, u;
        int steps;
        bool operator<(const edge & y) const
        {
            if (v < y.v) return true;
            return false;
        }
    };
    int X, Y; //终点结果
    int res; //答案
    //记录边是否已经访问过
    set<edge> sset;
    deque<edge> que;
    
    void BFS(int x, int y, int steps = 0)
    {   
        //从起点开始宽度优先搜索
        edge e = {x, y, steps}; 
        if (sset.count(e) != 0) return;
        
        que.push_front(e);
        sset.insert(edge(x, y, -1));
        if (x == X && y == Y)
        {
            res = steps;
            return;
        }
        
        while(!que.empty())
        {
            edge t = que.front(); que.pop_front();
            for (int i = 0; i < STEP; i++)
            {
                int v = t.v + xstep[i];
                int u = t.u + ystep[i];
                int _step = t.steps + 1;
                if (v == X && u == Y)
                {
                    res = _step;
                    return;
                }
                edge next = {v, u, _step};
                if (sset.count(edge(v, u, -1)) == 0)
                {
                    sset.insert(next);
                    que.push_back(next);
                }
            }
        }
    }
    int init(int x, int y)
    {
        this->X = x;
        this->Y = y;
        res = 17;
        BFS(0, 0);
        return res;
    }
public:
    int minKnightMoves(int x, int y) {
        int res = init(x, y);
        return res;
    }
};
```



别人的方法：不用queue，直接用list更加方便，改了下个别地方：

```python
class Solution(object):
    memo = {(0, 0): 0}
    queue = [(0, 0, 0)]
    for x, y, d in queue:
        for dx, dy in ((2, -1), (2, 1), (-2, -1), (-2, 1), (1, -2), (1, 2), (-1, -2), (-1, 2)):
            nx = x+dx
            ny = y+dy
            if 0 <= abs(nx) + abs(ny) <= 302: #加速
                if (nx, ny) not in memo:
                    memo[nx,ny] = d+1
                    queue.append((nx, ny, d+1))
    def minKnightMoves(self, x, y):
        x = abs(x)
        y = abs(y)
        return Solution.memo[x,y]
```

```c++
#define maxn 410
typedef pair<int,int> pii;
int ret[maxn][maxn];
int dx[]={1,2, 2, 1,-1,-2,-2,-1};
int dy[]={2,1,-1,-2,-2,-1, 1, 2};
class Solution {
public:
    int minKnightMoves(int x, int y) {
        x=abs(x),y=abs(y);
        memset(ret,-1,sizeof(ret));
        ret[0][0]=0;
        queue<pii> q;
        q.push(make_pair(0,0));
        int xL=x+20,yL=y+20;
        while (!q.empty())
        {
            pii v=q.front();q.pop();
            int px=v.first,py=v.second;
            for(int k=0;k<8;k++)
            {
                int nx=px+dx[k];
                int ny=py+dy[k];
                if (0<=nx && nx<=xL && 0<=ny && ny<=yL)
                {
                  if (ret[nx][ny]==-1)
                  {
                      ret[nx][ny]=ret[px][py]+1;
                      q.push(make_pair(nx,ny));
                  }
                    
                }
            }
        }
        return ret[x][y];
    }
};
```

这个代码是最concise的了：其他的比较看不懂。。 - -

```c++
class Solution {
public:
    int dx[8]={1, 2, 2, 1, -1, -2, -2, -1};
    int dy[8]={-2, -1, 1, 2, 2, 1, -1, -2};
    int dist[400][400];
    queue<pair<int, int> > que; //记录成对输入的栈
    int minKnightMoves(int x, int y) {
        if (x<0) x=-x;
        if (y<0) y=-y;
        memset(dist, -1, sizeof(dist));
        while(!que.empty()) que.pop(); //事先想着要弹空栈啊
        
        dist[10][10]=0; que.push(make_pair(10, 10));
        while(!que.empty()){
            int x=que.front().first, y=que.front().second;
            que.pop();
            for (int k=0; k<8; k++){
                int nx=x+dx[k], ny=y+dy[k];
                if (nx<0 || ny<0 || nx>350 || ny>350) continue;
                if (dist[nx][ny]==-1){
                    dist[nx][ny]=dist[x][y]+1;
                    que.push(make_pair(nx, ny));
                }
            }
        }
        
        return dist[x+10][y+10];
    }
};
```



#### 1198. Find Smallest Common Element in All Rows

>Given a matrix `mat` where every row is sorted in **increasing** order, return the **smallest common element** in all rows.
>
>If there is no common element, return `-1`.
>
>```pseudocode
>Input: mat = [[1,2,3,4,5],[2,4,5,8,10],[3,5,7,9,11],[1,3,5,7,9]]
>Output: 5
>```

* 复习下集合操作：



```c
class Solution:
    def smallestCommonElement(self, mat: List[List[int]]) -> int:
        s = set(mat[0])
        for i in range(1,len(mat)):
            s = s & set(mat[i])
            if len(s) == 0:
                return -1
        return min(s)
```






#### 1199. Minimum Time to Build Blocks

>You are given a list of blocks, where `blocks[i] = t` means that the `i`-th block needs `t` units of time to be built. A block can only be built by exactly one worker.
>
>A worker can either split into two workers (number of workers increases by one) or build a block then go home. Both decisions cost some time.
>
>The time cost of spliting one worker into two workers is given as an integer `split`. Note that if two workers split at the same time, they split in parallel so the cost would be `split`.
>
>Output the minimum time needed to build all blocks.
>
>Initially, there is only **one** worker.
>
>```pseudocode
>Input: blocks = [1,2,3], split = 1
>Output: 4
>Explanation: Split 1 worker into 2, then assign the first worker to the last block and split the second worker into 2.
>Then, use the two unassigned workers to build the first two blocks.
>The cost is 1 + max(3, 1 + max(1, 2)) = 4.
>```





* workers可以并行分裂，比如 1--->2，需要一次分裂，1---->2*2, 需要2次分裂，1-->2 * n，需要n次分裂。

假设目标blocks数组的大小为M，2个为一段，整个数组被分成小于等于n的小段。

每一个小段拥有一个split值，如【1,2,3】被分为【1,2】还有【3】，【3】带有一个split值，这个值的意思是花时间找人，所以这个split必须用完，才表示新的工人过来开始干活。新的工人过来

```python
class Solution:
    def minBuildTime(self, blocks: List[int], split: int) -> int:
        while len(blocks) > 1:
            blocks.sort()
            a = blocks[0]
            b = blocks[1]
            res = split + max(a, b)
            blocks = [res] + blocks[2:]
        return blocks[0]
```

```python
class Solution:
    def minBuildTime(self, blocks: List[int], split: int) -> int:
        heapq.heapify(blocks)
        while len(blocks) > 1:
            block_1 = heapq.heappop(blocks)
            block_2 = heapq.heappop(blocks)
            new_block = max(block_1, block_2) + split
            heapq.heappush(blocks, new_block)
        return blocks[0]
```

c++:

```c++
class Solution {
public:
    int minBuildTime(vector<int>& blocks, int split) {
        priority_queue<int, vector<int>, greater<>> pq;
        for (auto &v: blocks) pq.push(v);
        while (pq.size() > 1)
        {
            int b1 = pq.top(); pq.pop();
            int b2 = pq.top(); pq.pop();
            int res = split + max(b1, b2);
            pq.push(res);
        }
        return pq.top();
    }
};
```

