#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/7 7:38 下午
# @Author  : qinbinbin
# @email   : qinbinbin@360.cn
# @File    : xwz_view.py
import logging

from flask import current_app, request

from app.xwz import xwz_bp
from app.xwz.xwz_model import Xwz
from app.xwz.xwz_error import TestUserError


@xwz_bp.route('/', methods=['GET'])
def index():
    # celery 样例
    if current_app.config["DEBUG"] is True:
        logging.info("Hello Debug")
        # add.delay(1, 2)
        x = Xwz.get_by_key("春天")

        return x.to_dict()
    else:
        logging.info("Hello Test")
        return "Hello Test"


@xwz_bp.route('/<text>', methods=['GET'])
def get_by_text(text):
    # text = None
    info_list = Xwz.get_by_key(text)
    return {"result": [x.to_dict() for x in info_list]}


@xwz_bp.route('/text_key', methods=['POST'])
def get_by_key():
    params = request.form
    text = params.get("text_key", u"小丸子")
    info_list = Xwz.get_by_key(text)
    return {"result": [x.to_dict() for x in info_list]}
