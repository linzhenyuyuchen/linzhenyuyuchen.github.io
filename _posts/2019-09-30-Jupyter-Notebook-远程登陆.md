---
layout:     post
title:      Jupyter Notebook 远程登陆
subtitle:   Jupyter 配置远程登陆
date:       2019-09-30
author:     LZY
header-img: img/lossfunction.jpeg
catalog: true
tags:
    - Jupyter
---

# 生成config文件。

终端输入：

jupyter notebook --generate-config

（如果是root用户请用：jupyter notebook --generate-config --allow-config）

执行成功应该会显示：

Writing default config to: /home/zylin/.jupyter/jupyter_notebook_config.py 

# 生成密码

jupyter notebook password

Enter password: **** Verify password: ****

执行成功应该会显示：

[NotebookPasswordApp] Wrote hashed password to /home/zylin/.jupyter/jupyter_notebook_config.json

 
# 修改config文件
在 jupyter_notebook_config.py 中找到下面的行，取消注释并修改。

c.NotebookApp.ip='*' #星号代表任意ip，这个跟mysql的权限设置一样，所以说知识是互通的

c.NotebookApp.password = u'sha123456' #就是把生成的密码json文件里面的一串密码放这里

c.NotebookApp.open_browser = False #不自动打开浏览器

c.NotebookApp.port =8888 #可自行指定一个端口, 访问时使用该端口

 

# 重启jupyter

(非root用户)

安装anaconda自带Jupyter Notebook运行python环境就是anaconda自带python

# 后台运行jupyter notebook

```
nohup jupyter notebook >jupyter.txt &
# docker容器是root用户所以加上--allow-root
nohup jupyter notebook --allow-root >jupyter.txt &
```

# 远程访问

在局域网内其他电脑浏览器输入 http://ip:8888 就可以远程访问jupyter

# 更改kernel(虚拟环境)

https://blog.csdn.net/weixin_40539892/article/details/80940885

**Anaconda + jupyter notebook**

- 方法1：(**推荐**)

```
#配置好了PyTorch的环境，进入
conda activate PyTorch
#安装
conda install ipython
conda install jupyter
#然后打开
Jupyter Notebook
```

- 方法2：

创建虚拟环境并进入

```
conda create -n myprojectname python=3.7.3
```

进入虚拟环境，这是以后进入虚拟环境的操作；第一次创建虚拟环境后会直接进入因此不需要执行以下命令

```
source activate myprojectname
```

安装支持虚拟环境的插件nb_conda

```
conda install nb_conda
```

- 方法3

如果所需版本并不是已有的环境，可以直接在创建环境时便为其预装ipykernel

```
conda create -n tfname python=3.7.3 ipykernel
```