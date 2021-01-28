#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/7 7:38 下午
# @Author  : qinbinbin
# @email   : qinbinbin@360.cn
# @File    : view.py
import logging

from flask import current_app, request

from app.test import test_bp
from app.test.test_model import TestUser
from app.celery_task import add
from app.test.test_user_error import TestUserError
from util.util import get_user_id


@test_bp.route('/', methods=['GET'])
def index():
    # celery 样例
    if current_app.config["DEBUG"] is True:
        logging.info("Hello Debug")
        add.delay(1, 2)
        return "Hello Debug"
    else:
        logging.info("Hello Test")
        return "Hello Test"


@test_bp.route('/redis/<set_num>', methods=['GET'])
def test_redis(set_num):
    # redis样例
    redis_client = current_app.extensions['redis']
    redis_client.set('test_key', set_num)
    return "test_key is {}".format(redis_client.get('test_key'))


@test_bp.route('/add', methods=['POST'])
def add_user():
    # 增
    params = request.form
    kwargs = dict()
    kwargs["username"] = params.get("username")
    kwargs["age"] = params.get("age")
    user = TestUser.add_user(**kwargs)
    return user


@test_bp.route('/update', methods=['POST'])
def update_user():
    # 改
    params = request.form
    user_id = params.get("user_id")
    username = params.get("username")
    age = params.get("age")
    try:
        user = TestUser.get_by_id(user_id)
    except Exception as e:
        logging.exception(e)
        raise TestUserError(TestUserError.USER_NOT_FOUND, "get user failed.")
    user.update_user(username, age)
    return user.to_dict()


@test_bp.route('/delete', methods=['DELETE'])
def delete_user():
    # 删
    params = request.form
    user_id = params.get("user_id")
    try:
        user = TestUser.get_by_id(user_id)
    except Exception as e:
        logging.exception(e)
        raise TestUserError(TestUserError.USER_NOT_FOUND, "get user failed.")
    return {"user_id": user.delete_user()}


@test_bp.route('/list', methods=['GET'])
# 用户登录校验装饰器
@get_user_id
def users_list():
    # 查
    params = request.args
    page = params.get('page', 1, int)
    per_page = params.get('rows', 10, int)

    offset = (page - 1) * per_page
    kwargs = {
        'offset': offset,
        'per_page': per_page,
        'is_delete': False,
    }
    total, rows = TestUser.get_users(kwargs)
    return {
        'total': total,
        'rows': rows
    }
