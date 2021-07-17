---
layout:     post
title:      multiprocessing
subtitle:   多进程线程
date:       2021-06-22
author:     LZY
header-img: img/bg-20210718.jpg
catalog: true
tags:
    - 多进程
    - 多线程
---

# 多进程线程

### multiprocessing

```python
# how many processes to use
# num_processes = multiprocessing.cpu_count()
# num_processes = min(cfg.NUM_WORKER, num_processes)

num_processes = cfg.NUM_WORKER
pool = multiprocessing.Pool(num_processes)

if image_num_list is not None:
num_train_images = len(image_num_list)
else:
num_train_images = slide.get_num_training_slides(cfg)
if num_processes > num_train_images:
num_processes = num_train_images
images_per_process = num_train_images / num_processes


# start tasks
results = []
# https://github.com/tqdm/tqdm/issues/484#issuecomment-352463240
pbar = tqdm(total=len(tasks))
def update(*a):
pbar.update()
for t in tasks:
if image_num_list is not None:
results.append(pool.apply_async(apply_filters_to_image_list, t, callback=update))

```





### 多线程卡住

[ref](https://pythonspeed.com/articles/python-multiprocessing/) | [zhihu](https://zhuanlan.zhihu.com/p/75207672)

>不幸的是，虽然Pool类是有用的，但它也充满了恶毒的鲨鱼，只是等待你犯错误。



**错误代码示例**



```python
import logging
from threading import Thread
from queue import Queue
from logging.handlers import QueueListener, QueueHandler
from multiprocessing import Pool

def setup_logging():
    # Logs get written to a queue, and then a thread reads
    # from that queue and writes messages to a file:
    _log_queue = Queue()
    QueueListener(
        _log_queue, logging.FileHandler("out.log")).start()
    logging.getLogger().addHandler(QueueHandler(_log_queue))

    # Our parent process is running a thread that
    # logs messages:
    def write_logs():
        while True:
            logging.error("hello, I just did something")
    Thread(target=write_logs).start()

def runs_in_subprocess():
    print("About to log...")
    logging.error("hello, I did something")
    print("...logged")

if __name__ == '__main__':
    setup_logging()

    # Meanwhile, we start a process pool that writes some
    # logs. We do this in a loop to make race condition more
    # likely to be triggered.
    while True:
        with Pool() as pool:
            pool.apply(runs_in_subprocess)
```



```shell
About to log...
...logged
About to log...
...logged
About to log...
<at this point the program freezes>
```



**分析**

1. `top`查看cpu没有子进程在运行
2. POSIX（Linux, BSDs, macOS, and so on）上的子进程，通过系统调用`fork()`来复制进程，调用`execve()`替换子进程
3. deadlock 的原因是：不断地调用`fork()`，却没有调用`execve()`，复制内存中的所有内容，包括死锁，由主程序控制的死锁永远无法解开



**解决**

1. 不从子进程继承模块状态，而是从头开始
2. **注意：其中Unix默认使用fork模式, windows 默认使用spawn。**

```
import multiprocessing as mp
mp.set_start_method("spawn")  # spawn
# mp.set_start_method("forkserver")   # 使用forkserver模式

"""
这里执行多进程代码
"""
```







