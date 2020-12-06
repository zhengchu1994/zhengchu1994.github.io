---

title: Python-Trick
mathjax: true
date: 2020-06-30 19:00:00
tags: Python
categories: Python
visible:

---





#### 可变对象与不可变对象

对象有两种,“可更改”（mutable）与“不可更改”（immutable）对象。在python中，strings, tuples, 和numbers是不可更改的对象，而 list, dict, set 等则是可以修改的对象。(这就是这个问题的重点)

所以，传入函数的是不可变对象，那函数会自动复制一份a的引用：

```python
In [1]: a =  1                                                                                            

In [2]: def func(a): 
   ...:     a = 2 
   ...:     print(id(a), id(2)) 
   ...:                                                                                                   

In [3]: func(a)                                                                                           
4312986784 4312986784

In [4]: id(a)                                                                                             
Out[4]: 4312986752

In [5]: a                                                                                                 
Out[5]: 1
```

####  staticmethod和classmethod

`classmethod`相当于c++中的重载构造函数，方便对类做自定义的初始化；

` staticmethod`相当于类方法，独立于对象，被所有类对象都适用的通用方法。

```python
class Date(object):
    def __init__(self, day=0, month=0, year=0):
        self.day = day
        self.month = month
        self.year = year
    
    def display(self):
        print("{0}-{1}-{2}".format(self.month, self.day, self.year))
    
    @classmethod
    def from_string(cls, date_as_string):
        day, month, year = map(int, date_as_string.split('-'))
        date1 = cls(day, month, year)
        return date1
    
    @staticmethod
    def is_date_valid(date_as_string):
        day, month, year = map(int, date_as_string.split('-'))
        return day <= 31 and month <= 12 and year <=2077

```



#### 类变量、实例变量

实例的作用域里把类变量的引用改变了，就变成了一个实例变量,self.name不再引用Person的类变量name.

```python
In [8]: class Person: 
   ...:     name = 'xx' 
   ...:                                                                                                   

In [9]: a1 = Person()                                                                                     

In [10]: a1.name                                                                                          
Out[10]: 'xx'

In [11]: a2 = Person()                                                                                    

In [12]: a1.name = "zc"                                                                                   

In [13]: a1.name                                                                                          
Out[13]: 'zc'

In [14]: a2.name                                                                                          
Out[14]: 'xx'
```



#### `*args` and `**kwargs`

 `**kwargs`传递的是字典，

```python
In [17]: def fruit(*args): 
    ...:     for count, thing in enumerate(args): 
    ...:         print( "{0}. {1}".format(count, thing)) 
    ...:                                                                                                  

In [18]:                                                                                                  

In [18]: fruit("banana", "apple", "watermelon")                                                           
0. banana
1. apple
2. watermelon
```



调用函数时解包：传递列表(或者元组)的每一项并把它们解包.注意必须与它们在函数里的参数相吻合

```python
In [19]: def de(a = 'a', b = 'b', c = 'c'): 
    ...:     print("{0} . {1}. {2}".format(a, b, c)) 
    ...:                          
    
In [20]: dic = {a:3, b:4,c:5} 


In [22]: de(**dic)                                                                                        
3 . 4. 5
```



#### Property

用途1: 访问对象时，产生特殊行为

下例得优势在于，设置其中一个属性，对应需要更改的属性也随之改变。

```python
class Resistor(object):
    def __init__(self, ohms):
        self.ohms = ohms
        self.voltage = 0
        self.current = 0
r1 = Resistor(50e3)
#r1.ohms = 10e3
print(r1.ohms)

50000.0

# 通过子类的私有属性_voltage改变继承的voltage属性
class VoltageResistance(Resistor):
    def __init__(self, ohms):
        super().__init__(ohms)
        self._voltage = 0
    @property
    def voltage(self):
        return self._voltage 
    @voltage.setter
    def voltage(self, voltage):
        self._voltage = voltage 
        self.current = self._voltage / self.ohms
        
r2 = VoltageResistance(1e3)
print("Before: %5r amps" % r2.current)
r2.voltage = 10
print("After: %5r amps" % r2.current)

Before:     0 amps
After:  0.01 amps
```



```python
# 1.
class C:
    def __init__(self):
        self._x = None

    def getx(self):
        return self._x

    def setx(self, value):
        self._x = value

    def delx(self):
        del self._x

    x = property(getx, setx, delx, "I'm the 'x' property.")
    
#等价于1
class C:
    def __init__(self):
        self._x = None

    @property
    def x(self):
        """I'm the 'x' property."""
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @x.deleter
    def x(self):
        del self._x
```



用途2：实时计算

```python
"""
漏桶算法：桶里加水，但须在进入下一个周期时清空，也就是加水前，若水过期，需要倒掉，再加水；
"""
class Bucket(object):
    def __init__(self, period):
        self.period_delta = timedelta(seconds=period)
        self.reset_time = datetime.now()
        self.max_quota = 0
        self.quota_consumed = 0

    def __repr__(self):
        return ("Bucket(max_quota=%d, quota_consumed=%d)" % (self.max_quota,
                                                             self.quota_consumed))
    @property
    def quota(self):
        return self.max_quota - self.quota_consumed
    @quota.setter
    def quota(self, amount):
        delta = self.max_quota - amount
        # 重置quota，新周期开始
        if amount == 0:
            self.quota_consumed = 0
            self.max_quota = 0
        # 刚开始，还没水，加水
        elif delta < 0:
            assert self.quota_consumed == 0 # 没有水，所以不能消耗水
            self.max_quota = amount
        else:
            # 水够用,当前周期就消耗掉amount，但是得在外面确保amount + quota_consumed <= max_quota
            assert self.max_quota >= self.quota_consumed
            self.quota_consumed += delta

            
def fill(bucket, amount):
    now = datetime.now()
    if now - bucket.reset_time > bucket.period_delta:
        bucket.quota = 0
        bucket.reset_time = now
    bucket.quota += amount
# 消耗水时，先确认桶里有足够多的水
def deduct(bucket, amount):
    now = datetime.now()
    # 水已经过期，等于没有水
    if now - bucket.reset_time > bucket.period_delta:
        print("++--____-____--++")
        return False
    #水不够
    if bucket.quota - amount < 0:
        return False
    bucket.quota -= amount
    return True

bucket = Bucket(60)
print("Initial", bucket)
fill(bucket, 100)
print("Filled", bucket)

if deduct(bucket, 99):
    print("Had 99 quota")
else:
    print("Not enough for 99 quota")

print("Now", bucket)

if deduct(bucket, 3):
    print("Had 3 quota")
else:
    print("Not enough for 3 quota")
print("Still", bucket)
```



####  动态属性

* 定义了`__getattr__`的类，动态加入对象查询时不存在的属性

```python
class lazyDB:
    def __init__(self):
        self.exists = 5
    def __getattr__(self, name):
        value = "Value for %s" % name
        setattr(self, name, value)
        return value
data = lazyDB()
print("Before: ", data.__dict__)
print("foo: ", data.foo)
print("After: ", data.__dict)

-------
Before:  {'exists': 5}
foo:  Value for foo
After:  Value for __dict
```



```python
class LoggingLazyDB(lazyDB):
    def __getattr__(self, name):
        print("Called __getattr__(%s)" % name)
        return super().__getattr__(name)
      
data = LoggingLazyDB()
print("Exists: ", data.exists)
print("foo: ", data.foo)
print("foo: ", data.foo)

------
Exists:  5
Called __getattr__(foo)
foo:  Value for foo
foo:  Value for foo
```

```python
# 会在搜索对象属性后，调用__getattr__
hasattr(data, 'zoo')

------
Called __getattr__(zoo)
True
```

```python
data.__dict__
----
{'exists': 5, 'foo': 'Value for foo', 'zoo': 'Value for zoo'}
```

```python
# 确保不会添加某些属性名
class MissingPropertyDB(object):
    def __getattr__(self, name):
        if name == "bad_name":
            raise AttributeError("%s is missing" % name)
data = MissingPropertyDB()
data.bad_name
```

* 定义了`__getattribute__`的类，每次访问对象属性时都会触发

```python
class ValidatingDB:
    def __init__(self):
        self.exists = 5
    def __getattribute__(self, name):
        print("Called __getattribute__(%s)" % name)
        try:
            return super().__getattribute__(name)
        except AttributeError:
            value = "Value for %s" % name
            setattr(self, name, value)
            return value
```

```python
data = ValidatingDB()
print("exsits: ", data.exists)
print("foo: ", data.foo)
print("foo: ", data.foo)

--------
data = ValidatingDB()
print("exsits: ", data.exists)
print("foo: ", data.foo)
print("foo: ", data.foo)
```



* 使用`__setattr__`时，每次对属性赋值都会触发。

```python
class SavingDB:
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
class LoggingSavingDB(SavingDB):
    def __setattr__(self, name, value):
        print("Called __setattr__(%s, %r) " % (name, value))
        super().__setattr__(name, value)
data = LoggingSavingDB()
print("Before: ", data.__dict__)
data.foo = 5
print("After: ", data.__dict__)
data.foo = 7
print("After: ", data.__dict__)

--------- 
Before:  {}
Called __setattr__(foo, 5) 
After:  {'foo': 5}
Called __setattr__(foo, 7) 
After:  {'foo': 7}
```



#### contextlib + with 实现可复用的try + finally

```python
def my_function():
    logging.debug('Some debug data')
    logging.error('Error log here')
    logging.debug('More debug data')
    
#系统默认的级别是WARNING，所以会打印Error级别的错误    
my_function()
-----
ERROR:root:Error log here
```





```python
@contextlib.contextmanager
def log_level(level, name):
    logger = logging.getLogger(name)
    old_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    try:
        yield logger #传递句柄
    finally:
        logger.setLevel(old_level)

#with语句内，logger的严重级别设为debug
with log_level(logging.DEBUG, 'my-log') as logger:
    logger.debug("This is my  message!")
#恢复为默认的warning
logger.debug("This will not print")

-----
DEBUG:my-log:This is my  message!
```



#### 自带函数

双端队列，在头部插入删除元素是常数时间复杂度，`list`是线性复杂度。

```python
from collections import deque

dq = deque()
dq.append(1)
x = dq.popleft()
```





### Wraps

* 用装饰器修饰函数，使得函数执行前后做一些事。



```python
## work: 打印函数调用时，接收到的参数以及函数返回值。
def trace(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print("%s(%r, %r) -> %r" % (func.__name__, args, kwargs, result))
        return result
    return wrapper

@trace
def fibonacci(n):
    """return the n-th fibonacci number"""
    if n in (0, 1):
        return n
    return (fibonacci(n - 2) + fibonacci(n - 1))
  
  
fibonacci(3)
-------
fibonacci((1,), {}) -> 1
fibonacci((0,), {}) -> 0
fibonacci((1,), {}) -> 1
fibonacci((2,), {}) -> 1
fibonacci((3,), {}) -> 2
```





#### partial

* 对函数上增加、替换额外参数

```python
from functools import partial
def pr(*arg, **kwargs):
    for i in arg:
        print(i, end=" ")
    for i, j in kwargs.items():
        print("%s:%s" % (i, j), end = " ")
        
pr(1,2,3,a=1,b=2)
------
1 2 3 a:1 b:2 


#
pr_partial = partial(pr, 2077, name="zhengchu",time="2077")
pr_partial(1, 2, 3, token="see you~")

-------
2077 1 2 3 name:zhengchu time:2077 token:see you~ 
```





#### cmp_to_key

Transform an old-style comparison function to a [key function](https://docs.python.org/3.8/glossary.html#term-key-function). 

Used with tools that accept key functions (such as [`sorted()`](https://docs.python.org/3.8/library/functions.html#sorted), [`min()`](https://docs.python.org/3.8/library/functions.html#min), [`max()`](https://docs.python.org/3.8/library/functions.html#max), [`heapq.nlargest()`](https://docs.python.org/3.8/library/heapq.html#heapq.nlargest), [`heapq.nsmallest()`](https://docs.python.org/3.8/library/heapq.html#heapq.nsmallest), [`itertools.groupby()`](https://docs.python.org/3.8/library/itertools.html#itertools.groupby)). 

This function is primarily used as a transition tool for programs being converted from Python2 which supported the use of comparison functions.

A comparison function is any callable that accept two arguments, compares them, and **returns a negative number for less-than, zero for equality, or a positive number for greater-than**. 

A key function is a callable that accepts one argument and returns another value to be used as the sort key.

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





#### format上格式控制!r

`{!r}`与`format()`配合使用，而`%r`与%配合使用，二者不可以混合使用，否则会报错！

在`str.format`中，`!s`选择用于`str`设置对象格式，而`!r`选择`repr`设置值格式。



