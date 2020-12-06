子进程可以并行，

```python
>>> import shlex, subprocess
>>> command_line = input()
/bin/vikings -input eggs.txt -output "spam spam.txt" -cmd "echo '$MONEY'"
>>> args = shlex.split(command_line)
>>> print(args)
['/bin/vikings', '-input', 'eggs.txt', '-output', 'spam spam.txt', '-cmd', "echo '$MONEY'"]
>>> p = subprocess.Popen(args) # Success!
```



参数 *shell* （默认为 `False`）指定是否使用 shell 执行程序。如果 *shell* 为 `True`，更推荐将 *args* 作为字符串传递而非序列。



[`Popen`](https://docs.python.org/zh-cn/3/library/subprocess.html#subprocess.Popen) 类的实例拥有以下方法：

- `Popen.``poll`()

  检查子进程是否已被终止。设置并返回 [`returncode`](https://docs.python.org/zh-cn/3/library/subprocess.html#subprocess.Popen.returncode) 属性。否则返回 `None`。

- `Popen.``wait`(*timeout=None*)

  等待子进程被终止。设置并返回 [`returncode`](https://docs.python.org/zh-cn/3/library/subprocess.html#subprocess.Popen.returncode) 属性。

* [`communicate()`](https://docs.python.org/zh-cn/3/library/subprocess.html#subprocess.Popen.communicate)： 返回一个 `(stdout_data, stderr_data)` 元组。如果文件以文本模式打开则为字符串；否则字节。

* `Popen.``send_signal`(*signal*) ：将信号 *signal* 发送给子进程。



### Popen 对象方法

- poll(): 检查进程是否终止，如果终止返回 returncode，否则返回 None。
- wait(timeout): 等待子进程终止。
- communicate(input,timeout): 和子进程交互，发送和读取数据。
- send_signal(singnal): 发送信号到子进程 。
- terminate(): 停止子进程,也就是发送SIGTERM信号到子进程。
- kill(): 杀死子进程。发送 SIGKILL 信号到子进程。

