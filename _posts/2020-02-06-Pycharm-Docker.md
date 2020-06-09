---
layout:     post
title:      Pycharm+Docker
subtitle:   Pycharm+Docker实现深度学习环境远程开发环境配置
date:       2020-02-06
author:     LZY
header-img: img/xiaohuangren.jpg
catalog: true
tags:
    - Linux
---

# 容器内部修改

打开容器

```
docker run --runtime=nvidia -p 10022:22 -it -v /home/houwentai/env/mask_rcnn/maskrcnn-benchmark/datasets/coco/:/home/coco lzy /bin/bash
```

修改root用户密码

```
passwd
```

首先检查容器内部是否以安装 openssh-server与openssh-client 若没安装执行一下命令安装

```
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install -y openssh-server
sudo apt-get install openssh-client
```

修改SSH配置文件以下选项

```
vim /etc/ssh/sshd_config


# PermitRootLogin prohibit-password # 默认打开 禁止root用户使用密码登陆，需要将其注释
RSAAuthentication yes #启用 RSA 认证
PubkeyAuthentication yes #启用公钥私钥配对认证方式
PermitRootLogin yes #允许root用户使用ssh登录
```

启动sshd服务

```
/etc/init.d/ssh restart
```


使用 docker commit 来提交容器副本

```
sudo docker commit 62139249ab41 lzy
```


# Pycharm 配置


端口号为10022，其他与连接远程主机配置相同


# 下次使用


打开容器

```
docker run --runtime=nvidia -p 10022:22 --shm-size 8G  -it -v /home/houwentai/env/mask_rcnn/maskrcnn-benchmark/datasets/coco/:/home/coco lzy /bin/bash

# or

docker attach d_name
```


启动sshd服务

```
/etc/init.d/ssh start
```

