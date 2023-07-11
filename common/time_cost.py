#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/11 14:31
# @Author  : qinbinbin
# @email   : qinbinbin@360.cn
# @File    : time_cost.py

import time


# 定义一个装饰器函数
def calculate_runtime(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录函数开始时间
        result = func(*args, **kwargs)  # 执行被装饰的函数
        end_time = time.time()  # 记录函数结束时间
        runtime = end_time - start_time  # 计算函数运行时长
        print(f"函数 {func.__name__} 运行时长：{runtime} 秒")
        return result

    return wrapper


# 使用装饰器装饰需要计算运行时长的函数
@calculate_runtime
def my_function():
    # 在这里编写您的函数代码
    time.sleep(2)  # 模拟函数执行时间


# 调用被装饰的函数
# my_function()
