#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Binbin Qin
# @Contact : qinbinbin@geotmt.com
# @Time    : 2023/7/22 23:55
import hashlib


def md5_encrypt(string):
    # 创建一个MD5对象
    md5 = hashlib.md5()

    # 更新MD5对象的内容
    md5.update(string.encode())

    # 获取加密后的结果
    encrypted_string = md5.hexdigest()

    print("src str: {}, encrypted_string: {}".format(string, encrypted_string))

    return encrypted_string


if __name__ == '__main__':
    pass
