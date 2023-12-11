#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/8 16:25
# @Author  : qinbinbin
# @email   : qinbinbin@360.cn
# @File    : response_handler.py
import json
import time
import werkzeug

from flask import jsonify, Response
from json import JSONDecodeError

from common.base_error import BaseError


def custom_response(data, status_code=200):
    res = {"context": {
        "message": "OK",
        "status": status_code,
        "timestamp": int(time.time())
    },
        "data": {}}

    # 解码字节数据为字符串
    if isinstance(data, bytes):
        data = data.decode('utf-8')

    try:
        data = json.loads(data)
    except JSONDecodeError:
        pass

    res["data"] = data
    response = jsonify(res)
    response.status_code = status_code
    return response


# 自定义处理函数，将异常增加上下文，并以 JSON 格式返回
def handle_exception(e):
    response = e

    context = {}
    context['timestamp'] = int(time.time())

    if isinstance(response, (Response, werkzeug.wrappers.Response)):
        return response
    if isinstance(response, Exception):
        if isinstance(response, BaseError):
            context['status'] = response.errno
            context['message'] = response.message
        elif isinstance(response, Exception):
            context['status'] = BaseError.INTERNAL_ERROR
            if hasattr(response, 'description'):
                context['message'] = response.description
            else:
                context['message'] = response.message
        else:
            context['status'] = BaseError.INTERNAL_ERROR
            context['message'] = 'Internal server error'

        response = {}
    else:
        context['status'] = BaseError.OK
        context['message'] = 'OK'

    if isinstance(response, Exception):
        response['context'] = context
    else:
        new_res = {}
        new_res['data'] = response
        new_res['context'] = context
        if 'cookies' in response:
            new_res['cookies'] = response['cookies']
            del response['cookies']
        response = new_res

    return response


# # 使用 @app.errorhandler 装饰器定义全局异常处理器
# @app.errorhandler(Exception)
# def handle_all_exceptions(e):
#     return handle_exception(e)
#
#
# @app.after_request
# def process_response(response):
#     res_json = response.get_json()
#     if res_json is not None and "data" in res_json.keys() and "context" in res_json.keys():
#         return response
#     if isinstance(response.get_data(), bytes):
#         return custom_response(response.get_data(), response.status_code)
#
#     return response
