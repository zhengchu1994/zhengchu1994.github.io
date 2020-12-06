---
title: 算法Basic-BFS
mathjax: true
date: 2019-10-20 21:53:32
tags: [Acwing, Algorithm]
categories: 算法与数据结构
visible:
---





## Flood Fill

* 优点：可以在线性时间内找到某个点所在的联通块；

  

####  池塘计数 

```c++
#include <iostream>
#include <cstring>
using namespace std;

const int N = 1002;

char M[N][N];
bool st[N][N];
typedef pair<int, int> PII;
PII q[N * N];
int n, m;

void bfs(int tx, int ty)
{
    int hh = 0, tt = 0;
    q[tt++] = {tx, ty};
    st[tx][ty] = true;
    while(hh <= tt)
    {
        auto t = q[hh++];
        int x = t.first, y = t.second;
        for(int i = -1; i <= 1; i++)
            for(int j = -1; j <= 1; j ++)
            {
                if(i == j && j == 0) continue;
                int a = x + i, b = y + j;
                if(a >= n || a < 0 || b >= m || b < 0) continue;
                if(M[a][b] == '.' || st[a][b]) continue;
                st[a][b] = true;
                q[tt++] = {a, b};
            }
    }
}

int main()
{
    scanf("%d%d", &n, &m);
    for(int i = 0; i < n; i ++) scanf("%s", &M[i]);
    
    int res = 0;
    for(int i = 0; i < n; i ++)
        for(int j = 0; j < m; j ++)
            if(M[i][j] == 'W' && !st[i][j])
            {
                res++;
                bfs(i, j);
            }
    
    printf("%d\n", res);
    
    return 0;
}
```

#### 城堡问题  

```c++
//可以用位数是否为1判断该位置是否存在墙;
#include <iostream>
#include <algorithm>
using namespace std;

typedef pair<int,int> PII;

int m, n;
const int N =  51;
int g[N][N];
bool st[N][N];
PII q[N * N];

//1表示西墙，2表示北墙，4表示东墙，8表示南墙，
int idx[4] = {0, -1, 0, 1}, idy[4] = {-1, 0, 1, 0};

int bfs(int dx, int dy)
{
    int hh = 0, tt = 0;
    
    q[tt++] = {dx, dy};
    st[dx][dy] = true;
    
    while(hh <= tt)
    {
        auto pos = q[hh++];
        int a = pos.first, b = pos.second;
        for(int i = 0; i < 4; i ++)
        {
            int x = a + idx[i], y = b + idy[i];
            if(x >= m || x < 0 || y >= n || y < 0) continue;
            if(g[a][b] >> i & 1) continue;
            if(st[x][y]) continue;
            st[x][y] = true;
            q[tt++] = {x, y};
        }
    }
    return tt;
}
int main()
{
    cin >> m >> n;
    for(int i = 0; i < m; i ++)
        for(int j = 0; j < n; j ++)
             cin >> g[i][j];
             
    int res = 0, area = 0;
    for(int i = 0; i < m; i ++)
        for(int j = 0 ; j < n; j ++)
        {
            if(!st[i][j])
            {
                area = max(area, bfs(i, j));
                res ++;
            }
        }
    cout << res << endl;
    cout << area << endl;
    return 0;
}
```



####  山峰和山谷 

```c++
#include <iostream>
#include <algorithm>
using namespace std;


typedef pair<int, int> PII;
const int N = 1010, M = N * N;

int n;
int f[N][N];
bool st[N][N];
PII q[M];


void bfs(int dx, int dy, bool &pk, bool &vl)
{
    int hh = 0, tt = 0;
    q[tt++] = {dx, dy};
    st[dx][dy] = true;
    
    while(hh < tt)
    {
        auto pos = q[hh++];
        
        int x = pos.first, y = pos.second;
        for(int i = x - 1; i <= x + 1; i ++)
            for(int j = y - 1; j <= y + 1; j++)
            {
                if(i == x && j == y) continue;
                if(i >= n || i < 0 || j >= n || j < 0) continue;
                if (f[i][j] != f[x][y]) //边界
                {
                    if (f[i][j] > f[x][y]) pk = false;
                    else vl = false;
                }
                else if (!st[i][j])
                {
                    st[i][j] = true;
                    q[tt++] = {i, j};
                }
            }
    }
}


int main()
{
    scanf("%d", &n);
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j ++)
            scanf("%d", &f[i][j]);
    
    int peak = 0, valley = 0;
    for(int i = 0; i < n; i ++)
        for(int j = 0; j < n; j ++)
        {
            bool is_peak = true, is_valley = true;
            if(!st[i][j])
            {
                bfs(i, j, is_peak, is_valley);
                if(is_peak) peak++;
                if(is_valley) valley++;
            }
        }
        
    printf("%d %d\n", peak, valley);
    
    return 0;
}
```



## 最短路模型

BFS，当边权重一样的时候便可以用bfs找到最短路，是dijkstra算法的特例；dijkstra是用堆来维护队列，每次取出最小值；当全部边的权重都是1，则是按照层的顺序来遍历所有点，因此所有点到起点的距离是严格递增的，因此队首一定是最小值，所以说所有边权是1，队列相当于Dijkstra中的优先队列，队头就是一个当前最小值。

队列保持：①两段性（最多情况）；②单调性：队列中的元素距离源点的距离是单调递增的。



#### 迷宫问题

```c++
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

int n;

typedef pair<int, int> PII;
const int N = 1010;
int g[N][N];
PII mem[N][N];
PII q[N * N];

int idx[] = {0, 0, -1, 1};
int idy[] = {1, -1, 0 , 0};
void bfs()
{
    memset(mem, -1, sizeof mem);
    int hh = 0, tt = 0;
    q[0] = {n - 1 , n - 1};
    mem[n - 1][n - 1] = {n - 1, n - 1};
    while(hh <= tt)
    {
        auto ps = q[hh++];
        int x = ps.first,  y = ps.second;
        if (x == 0  && y == 0) break;
        for(int i = 0; i < 4; i ++)
        {
            int a = x + idx[i], b = y + idy[i];
            if(a >= n || a < 0 || b >= n || b < 0 || mem[a][b].first != -1 || g[a][b]) continue;
            q[++ tt] = {a, b};
            mem[a][b] = {x, y};
        }
    }
}
int main()
{
    cin >> n;
    for(int i = 0; i < n; i ++)
        for(int j = 0 ; j < n; j ++)
            cin >> g[i][j];
    
    bfs();
    PII end = {0, 0};
    
    while(true)
    {
        cout << end.first <<" " << end.second << endl;
        if (end.first == n - 1 && end.second == n - 1) break;
        end = mem[end.first][end.second];
    }
    
    return 0;
    
}
```



#### 武士风度的牛

```c++
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

typedef pair<int, int> PII;
const int N = 160;
char g[N][N];
int dist[N][N];
PII q[N * N];
int n, m;
int bfs(int x, int y)
{
    memset(dist, -1, sizeof dist);
    int idx[] = {-2, -1, 1, 2, -2, -1, 2, 1};
    int idy[] = {1, 2, 2, 1, -1, -2, -1, -2};
    
    int hh = 0, tt = 0;
    q[tt++] = {x, y};
    dist[x][y] = 0;
    while(hh < tt)
    {
        auto ps = q[hh++];
        int x = ps.first, y = ps.second;
        if(g[x][y] == 'H') return dist[x][y];
        for(int i = 0; i < 8; i ++)
        {
            int a = x + idx[i], b = y + idy[i];
            if(a >= n || a < 0 || b >= m || b < 0) continue;
            if(dist[a][b] != -1 || g[a][b] == '*') continue;
            q[tt++] = {a, b};
            dist[a][b] = dist[x][y] + 1;
        }
    }
    
    return -1;
}
int main()
{
    scanf("%d%d", &m, &n);
    for(int i = 0 ; i < n; i ++) scanf("%s", &g[i]);
    
    int res;
    for(int i = 0; i < n; i ++)
        for(int j = 0;  j < m; j ++)
            if(g[i][j] == 'K')
                res = bfs(i, j);
    
    printf("%d\n", res);
    
    return 0;
}
```



####  抓住那头牛 



```c++
#include <iostream>
#include <cstring>
using namespace std;

const int N = 1e5 + 10;

int n, k;
int q[2 * N];

int dist[N * 2];

int bfs(int pos)
{
    memset(dist, -1, sizeof dist);
    
    dist[pos] = 0;
    
    int hh = 0, tt = 0;
    q[tt++] = pos;
    while(hh < tt)
    {
        int ps = q[hh++];
        while(ps == k) return dist[k];
        if(ps + 1 < 2 * N && dist[ps + 1] == -1)
        {
            dist[ps + 1] = dist[ps] + 1;
            q[tt++] = ps + 1;
        }
        if(ps - 1 >= 0 && dist[ps - 1] == -1)
        {
            dist[ps - 1] = dist[ps] + 1;
            q[tt++] = ps - 1;
        }
        if(ps * 2 < 2 * N && dist[ps * 2] == - 1)
        {
            dist[ps * 2] = dist[ps] + 1;
            q[tt++] = ps * 2;
        }
    }
    
    return -1;
}

int main()
{
    cin >> n >> k;
    cout << bfs(n) << endl;
    
    return 0;
}
```



## 多源BFS

与多源最短路不一样，多源最短路求的是没两个点之间的距离，而这里求的是每个点到各自最近起点的距离。

#### 矩阵距离

* 对所有起点入队列然后bfs。

```c++
#include <iostream>
#include <cstring>
#include <queue>
#include <algorithm>

using namespace std;

int n, m;
const int N = 1110;
char cg[N][N];
int g[N][N];
int dist[N][N];

int main()
{
    memset(dist, -1, sizeof dist);
    queue<pair<int, int>> q;
    
    scanf("%d%d", &n, &m);

    for(int i = 0; i < n; i ++)
    {
        scanf("%s", &cg[i]);
        for(int j = 0; j < m; j ++)
        {
            g[i][j] = cg[i][j] - '0';
            if (g[i][j] == 1)
            {
                dist[i][j] = 0;
                q.push(make_pair(i, j));
            }
        }
    }
    
    int idx[] = {0, 0, -1, 1}, idy[] = {1, -1, 0, 0};
    
    while(q.size())
    {
        auto pos = q.front(); q.pop();
        int x = pos.first, y = pos.second;
        for(int i = 0; i < 4; i ++)
        {
            int a = x + idx[i], b = y + idy[i];
            if(a >= n || a < 0 || b >= m || b < 0) continue;
            if(dist[a][b] != -1) continue;
            dist[a][b] = dist[x][y] + 1;
            q.push(make_pair(a, b));
        }
    }
    
    for(int i = 0; i < n; i ++)
    {
        for(int j = 0; j < m; j ++)
            cout << dist[i][j] << " ";
        cout <<endl;
    }
    
    return 0;
}

```



## 最小步数模型

即整个棋盘看做一个状态，对整个棋盘进行一些操作，成为另一个状态。难点：如何存储这个状态。（C：hash、康拓展开；C++：map、unordered_map）



#### 魔板

```c++
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <queue>
using namespace std;

unordered_map<string, pair<char,string>> pre;
unordered_map<string, int> dist;
char g[2][4];

string to()
{
    string s;
    for(int j = 0; j < 4; j ++)
        s += g[0][j];
    for(int j = 3; j >= 0; j--)
        s += g[1][j];
    return s;
}
void set(string s)
{
    for(int j = 0; j < 4; j ++)
        g[0][j] = s[j];
    for(int i = 0,j = 3; j >= 0; i++, j--)
        g[1][i] = s[4  + j];
}

string op1(string s)
{
    set(s);
    for(int i = 0; i < 4; i ++) swap(g[0][i], g[1][i]);
    s = to();
    return s;
}

string op2(string s)
{
    set(s);
    for(int i = 0 ; i < 2; i ++)
    {
        for(int j = 3; j > 0; j --)
        {
            swap(g[i][j], g[i][j - 1]);
        }
    }
    s = to();
    return s;
}

string op3(string s)
{
    set(s);
    swap(g[0][1], g[1][2]);
    swap(g[0][1], g[1][1]);
    swap(g[0][2], g[1][2]);
    s = to();
    return s;
}
int bfs(string start, string target)
{
    if(start == target) return 0;
    
    queue<string> q;
    q.push(start);
    dist[start] = 0;
    while(q.size())
    {
        
        string s = q.front(); q.pop();
        
        if(s == target) break;
        string m[3];
        
        m[0] = op1(s);
        m[1] = op2(s);
        m[2] = op3(s);
        for(int i = 0 ; i < 3; i ++)
        {
            if(!dist.count(m[i]))
            {
                dist[m[i]] = dist[s] + 1;
                pre[m[i]] = {'A' + i, s};
                if(m[i] == target) return dist[m[i]]; 
                q.push(m[i]);
            }
        }
     }
     
     return -1;
}
int main()
{
    string tg;
    for(int i = 1; i <= 8; i ++)
    {
        int a; 
        cin >> a;
        tg += char(a + '0');
    }
    string st = "12345678";
    int op_num = bfs(st, tg);
    if(!op_num) cout << 0 << endl;
    else
    {
        cout << op_num <<endl;
        string out;
        while(st != tg)
        {
            out += pre[tg].first;
            tg = pre[tg].second;
        }
        reverse(out.begin(), out.end());
        cout << out << endl;
    }
    
    return 0;
}
```



## 双端队列广搜



####  电路维修

* 考虑迪杰斯特拉算法，边权为1和0的最短路问题，利用双端队列处理这个问题可以，遇到0插入队头，1插入队尾。
* 特殊：不用记录某一个点是否被搜到，因为0、1会使得某位置会多次入队；出队的时候才判断该元素。

```c++
#include <iostream>
#include <deque>
#include <algorithm>
#include <cstring>


using namespace std;

typedef pair<int, int> PII;
const int N = 510;
char g[N][N];
int r, c;
int dist[N][N];


int ans;
bool Find;

void bfs()
{
    memset(dist, 0x3f, sizeof dist);
    deque<PII> dq;
    dq.push_front({0, 0});
    dist[0][0] = 0;

    /*
    如下三个所对应的位置必须对应:也就是说当前位置g[x][y]更新到下一位位置(g[a][b])且形状为pat[i]形状，
    那么需不需要旋转部件，取决于当前位置g[x][y]的目标更新位置g[ix][iy]是否和形状pat[i]一样，
    即我们在位置(i,j),是看周围的下一个位置是否需要旋转。
    pat:下一个可到达的四个方向；
    idx\idy://pat坐标：左上角 、右上角、右下角、左下角
    ix\iy:对于在g中下一个位置pat[i]的下标。
    */
    char pat[] = "\\/\\/"; 
    int idx[4] = {-1, -1, 1, 1}, idy[4] = {-1, 1, 1, -1};
    int ix[4] = {-1, -1, 0, 0}, iy[4] = {-1, 0, 0, -1};

    while(dq.size())
    {
        auto pos = dq.front(); dq.pop_front();
        int x = pos.first, y = pos.second;
        if (x == r &&  y == c)
        {
            Find = true;
            ans = dist[x][y];
            return;
        }
        for(int i = 0;  i < 4; i ++)
        {
            int a = x + idx[i], b = y + idy[i];
            if(a > r || a < 0 || b > c || b < 0) continue;
            int ca = x + ix[i], cb = y + iy[i];
            int w = g[ca][cb] != pat[i];
            if(dist[a][b] > dist[x][y] + w) //同dijkstra，达到同一点的距离可能更新多次
            {
                dist[a][b] = dist[x][y] + w;
                if (w) dq.push_back({a, b});
                else dq.push_front({a, b});
            }
        }
    }
}

int main()
{
    int t;
    scanf("%d",&t);
    while(t --)
    {
        Find = false;
        scanf("%d%d",&r, &c);
        for(int i = 0; i < r; i ++) scanf("%s", &g[i]);
        bfs();
        if (Find) printf("%d\n", ans);
        else puts("NO SOLUTION");
    }



    return 0;
}
```



## 双向广搜

一般用在最小步数模型里，因为这类问题搜索空间一般比较大，是指数级别。

实现方式：①两个方向扩展一步，每次选择当前队列中元素较少的一方进行扩展；



#### 字串变换



## A*

适合的环境：最小步数模型，即搜索空间非常大的时候；比较像dijkstra，可以看做所有估计函数都为0的算法。不能有负权回路。不能保证每一次出队的元素是最小值，只有终点出队才能保证是最小值。

* 估价函数：大于等于0，小于等于真实值。

* 必须保证有解的时候才能有，否则无解的时候效率低于BFS（队列O(1)）;

* 算法正确的条件：起点到当前状态的真实距离为`d(state)`，从`state`到终点的真实距离为`g(state)`，估计距离为`f(state)`,那么必须满足：`f(state) <= g(state)`。
* 证明：假设当终点出队的时候不是最小值，此时 `dist>d最优`，则队列中一定存在最优解。因此发现`d(u) + f(u) <= d(u) + g(u)`。







```pseudocode
while(!q.empty())
{
	t <---- 优先队列的队头 ~~~小根堆 //存储（从起点到当前点的真实距离）、（从当前点到终点的估计距离）
	当终点第一次出队的时候，break;
	for t的所有邻边
		将能更新距离的邻边入队
}
```





规律：
```pseudocode
1、堆入队：该初始状态的估计距离 + 该初始状态；
heap.push({mDis(start), start}); 
2、遍历周围状态，当该状态未记录距离，或者距离小于当前真实距离：
dist[state] = step + 1;
!dist.count(state) || dist[state] > step + 1
heap.push({dist[state] + mDis(state), state});
3、目标状态第一次出队即得到最优解；
if (state == end) break;
```
下：
```c++
#include <bits/stdc++.h>
using namespace std;


typedef pair<int, string> PIS;


char p[] = "lrud";
int ix[] = {0, 0, -1, 1}, iy[] = {-1, 1, 0, 0};



int mDis(string state)
{
    int cnt = 0;
    for(int i = 0; i < state.size(); i ++)
        if(state[i] != 'x')
        {
            int t = state[i] - '1';
            cnt += abs(t / 3 - i / 3) + abs(t % 3 - i % 3);
        }
    return cnt;
}

string AStar(string start)
{ 
    string end = "12345678x";
    unordered_map<string, pair<char, string>> pre;
    unordered_map<string , int> dist;
    priority_queue<PIS, vector<PIS>, greater<PIS>> heap;
    
    heap.push({mDis(start), start});
    dist[start] = 0;
    
    while(heap.size())
    {
        auto t = heap.top();
        heap.pop();
        
        string state = t.second; // state
        
        if (state == end) break;
        
        
        int step = dist[state]; //当前宽搜的真实距离
        
        int x, y;
        
        for(int i = 0; i < 9; i++)
            if(state[i] == 'x')
            {
                x = i / 3, y = i % 3;
                break;
            }
        
        
        string source = state;
        
        for(int i = 0; i < 4; i ++)
        {
            int a = x + ix[i], b = y + iy[i];
            if(a < 0 || a >=3 || b < 0 || b >= 3) continue;
            
            swap(state[3 * x + y], state[3 * a + b]); //next state
            if(!dist.count(state) || dist[state] > step + 1)
            {
                dist[state] = step + 1;
                
                pre[state] = {p[i], source};
                
                heap.push({dist[state] + mDis(state), state});
                //heap.push({mDis(state), dist[source] + 1});
            }
            
            swap(state[3 * x + y], state[3 * a + b]); //previous state
            
        }
    }
    
    string  res;
    
    while(end != start)
    {
        res += pre[end].first;
        end = pre[end].second;
    }
    
    reverse(res.begin(), res.end());
    return res;
    
}

int main()
{
    string state, seq;
    
    char c;
    
    for(int i = 0; i < 9; i ++)
    {
        cin >> c;
        state += c;
        if (c != 'x') seq += c;
    }
    
    int cnt;
    //逆序对数目
    for(int i = 0; i < seq.size(); i ++)
        for(int j = i + 1; j < seq.size(); j ++)
            if(seq[i] > seq[j]) 
                cnt++;
    if(cnt & 1) puts("unsolvable");
    
    else cout << AStar(state) << endl;
    
    return 0;
}
```