#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/14 8:29 下午
# @Author  : qinbinbin
# @email   : qinbinbin@360.cn
# @File    : service_decorator.py
import copy
import logging

from flask import Response

from common.base_error import BaseError
from common.decorators import BaseDecorator


class ServiceDecorator(BaseDecorator):
    def __init__(self, func):
        super().__init__(func)
        self.func = func
        self.__name__ = self.func.__name__

    def __call__(self, *args, **kwargs):
        try:
            result = self.func(*args, **kwargs)
            if isinstance(result, Response):
                return result
            if isinstance(result, dict) and '_id' in result:
                del result['_id']
            if isinstance(result, BaseError):
                return result
            copy_result = copy.deepcopy(result)
            self.remove_instance(copy_result)
        except BaseError as e:
            logging.exception(e)
            logging.error(e.args)
            return e
        except UnicodeDecodeError as e:
            logging.exception(e)
            logging.error(e.args)
            return BaseError(
                BaseError.INTERNAL_ERROR,
                'Prophet only supports GBK and UTF-8 encoding format, please transform file format')
        except Exception as e:
            logging.exception(e)
            logging.error(e.args)
            return BaseError(BaseError.INTERNAL_ERROR, 'Prophet System error')

        if result is None:
            copy_result = result = {}
        return copy_result

    def remove_instance(self, root):
        if isinstance(root, dict):
            for key, value in root.items():
                root[key] = self.remove_instance(value)
        else:
            if isinstance(root, list):
                new_value = []
                for value in root:
                    n_value = self.remove_instance(value)
                    if n_value is not None:
                        new_value.append(n_value)
                return new_value
            elif hasattr(root, '__module__'):
                root = root.__class__.__name__
        return root


def route(self, rule, **options):
    def decorator(f):
        f = ServiceDecorator(f)
        endpoint = options.pop('endpoint', None)
        self.add_url_rule(rule, endpoint, f, **options)
        return f

    return decorator
