#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/7 7:39 下午
# @Author  : qinbinbin
# @email   : qinbinbin@360.cn
# @File    : __init__.py


import types
from flask import Blueprint
from common.service_decorator import route

test_bp = Blueprint("test", __name__, url_prefix="/test")
test_bp.route = types.MethodType(route, test_bp)
