#!/usr/bin/env python
# -*- coding: utf-8 -*-

import functools
import logging


class BaseDecorator(object):
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, obj_type):
        return functools.partial(self.__call__, obj)

    def __call__(self, *args, **kwargs):
        try:
            return self.func(*args, **kwargs)
        except Exception as e:
            logging.exception(e)
            raise e


def singleton(cls):
    instances = {}

    def __singleton(*args, **kw):
        if cls not in instances:
            instances[cls] = cls(*args, **kw)
        return instances[cls]

    return __singleton
