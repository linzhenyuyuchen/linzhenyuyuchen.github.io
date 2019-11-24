---
layout:     post
title:      Python subprocess
subtitle:   from subprocess import *
date:       2019-11-03
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - subprocess
    - Python
---

[Reference](https://docs.python.org/2/library/subprocess.html)

# subprocess

`from subprocess import *`

# subprocess.call()

`retcode = call(["ls", "-l"])`


将程序名(ls)和所带的参数(-l)一起放在一个表中传递给subprocess.call()


shell默认为False，在Linux下，shell=False时, Popen调用os.execvp()执行args指定的程序；shell=True时，如果args是字符串，Popen直接调用系统的Shell来执行args指定的程序，如果args是一个序列，则args的第一项是定义程序命令字符串，其它项是调用系统Shell时的附加参数。

`retcode = call("ls -l",shell=True)`

# subprocess.Popen()

实际上，上面的几个函数都是基于Popen()的封装(wrapper)。这些封装的目的在于让我们容易使用子进程。当我们想要更个性化我们的需求的时候，就要转向Popen类，该类生成的对象用来代表子进程。

与上面的封装不同，Popen对象创建后，主程序不会自动等待子进程完成。我们必须调用对象的wait()方法，父进程才会等待 (也就是阻塞block)

```
child = Popen(['ping','-c','4','blog.linuxeye.com'])
child.wait()
print('parent process')
```

- child.poll() # 检查子进程状态
- child.kill() # 终止子进程
- child.send_signal() # 向子进程发送信号
- child.terminate() # 终止子进程

子进程的PID存储在child.pid


# 子进程的文本流控制

```
child.stdin
child.stdout
child.stderr
```

可以在Popen()建立子进程的时候改变标准输入、标准输出和标准错误，并可以利用subprocess.PIPE将多个子进程的输入和输出连接在一起，构成管道(pipe)

```
child1 = Popen(["ls","-l"], stdout=PIPE)

print(child1.stdout.read())
```

```
child1 = Popen(["cat","/etc/passwd"], stdout=PIPE)
child2 = Popen(["grep","0:0"],stdin=child1.stdout, stdout=PIPE)
out = child2.communicate()
```

subprocess.PIPE实际上为文本流提供一个缓存区。child1的stdout将文本输出到缓存区，随后child2的stdin从该PIPE中将文本读取走。child2的输出文本也被存放在PIPE中，直到communicate()方法从PIPE中读取出PIPE中的文本。

注意：communicate()是Popen对象的一个方法，该方法会阻塞父进程，直到子进程完成
