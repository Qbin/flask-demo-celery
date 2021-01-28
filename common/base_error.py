#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/8 6:28 下午
# @Author  : qinbinbin
# @email   : qinbinbin@360.cn
# @File    : base_error.py


class BaseError(Exception):
    """ The base error(exception) class, all user defined error type should inherited this class"""
    OK = 0
    INTERNAL_ERROR = 80000
    PARAM_INVALID = 80100
    API_INVALID = 80200
    SYSTEM_ERROR = 80400
    DATABASE_ERROR = 80500
    FILESYSTEM_ERROR = 80600
    AUTH_INVALID = 80700

    CONFIG_ERROR = 80700
    CONFIG_FILE_INVALID = 80701
    CONFIG_TYPE_INVALID = 80702
    CONFIG_ENV_INVALID = 80711
    CONFIG_KEY_INVALID = 80712

    def __init__(self, *args):
        super(BaseError, self).__init__(*args)
        self.errno = args[0]
        self.message = args[1]
