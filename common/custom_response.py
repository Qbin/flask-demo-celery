#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/8 6:24 下午
# @Author  : qinbinbin
# @email   : qinbinbin@360.cn
# @File    : custom_response.py

import time
import werkzeug

from common.base_error import BaseError

from collections import Iterable
from flask import Response
from flask import jsonify


class ProphetResponse(Response):
    """
    we define a new response type, invoke jsonify converting response to json format.
    jsonify actually set response head: Content-Type = application/json
    """

    @classmethod
    def force_type(cls, response, environ=None):
        response = ProphetResponse.exception2context(response)
        if isinstance(response, Response):
            return super(Response, cls).force_type(response, environ)
        cookies = {}
        if isinstance(response, Iterable) and 'cookies' in response and isinstance(response['cookies'], dict):
            cookies = response['cookies']
            del response['cookies']
        if isinstance(response, (dict, list)):
            response = jsonify(response)

        for k, v in cookies.items():
            response.set_cookie(
                key=str(k), value=str(v[0]), path='/', domain=str(v[1]), expires=int(time.time()) + int(v[2]))
        return super(Response, cls).force_type(response, environ)

    @staticmethod
    def exception2context(response):
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
