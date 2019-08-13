---
title: Unity用IL2Cpp打包64位版本aab文件，上线Google商店
date: 2019-08-06 20:07:02
tags:
- Unity

categories:
- Unity
- 安卓
---

## 正文

本文转载自https://www.cnblogs.com/INSIST-NLJY/p/11044558.html 部分内容，侵删

安卓Apk上线Google要求64位且打包成Android Asset Bundle（即aab格式），以下是相关设置

> 1. 在PlaySettings->other settings->Scriptiing Backend 选择IL2CPP（默认是Mono）, c++ Compiler Configuration 选择Release Target Architectures 里面的Arm64就可以勾选的了，勾选打包即可

> 2. Android Asset Bundle优化在Build Sttings 勾选Build App Bundle（Google Play）即可，打出的是aab包 打包运行成功。

> 注：要是测试的话就用Mono打包吧，毕竟IL2CPP打包要慢上几倍

---

## 参考链接

https://www.cnblogs.com/INSIST-NLJY/p/11044558.html


