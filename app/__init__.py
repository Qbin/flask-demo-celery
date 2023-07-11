#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/7 7:37 下午
# @Author  : qinbinbin
# @email   : qinbinbin@360.cn
# @File    : __init__.py.py

from app.test.view import test_bp
from app.text_cluster.view import text_cluster_bp


# register blueprint
def register(app):
    app.register_blueprint(test_bp)
    app.register_blueprint(text_cluster_bp)
