#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/7 7:39 下午
# @Author  : qinbinbin
# @email   : qinbinbin@360.cn
# @File    : __init__.py


import types
from flask import Blueprint
from common.service_decorator import route

text_cluster_bp = Blueprint("text_cluster", __name__, url_prefix="/text_cluster")
text_cluster_bp.route = types.MethodType(route, text_cluster_bp)
