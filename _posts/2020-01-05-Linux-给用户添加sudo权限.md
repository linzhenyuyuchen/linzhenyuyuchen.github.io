---
layout:     post
title:      Linux 给用户添加sudo权限
subtitle:   sudo
date:       2020-01-05
author:     LZY
header-img: img/whatisnext.jpg
catalog: true
tags:
    - Linux
---

# sudo

```
# 切换到root用户
su -
# 输入密码

# 编辑
vim /etc/sudoers

# 找到"root ALL=(ALL) ALL"在起下面添加"xxx ALL=(ALL) ALL"(xxx是用户名)

#保存
:wq!


```
