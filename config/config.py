#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/7 7:35 下午
# @Author  : qinbinbin
# @email   : qinbinbin@360.cn
# @File    : config.py

import os


# TODO 使用prophet 和 pioneer的线上环境
class BaseConfig:
    SECRET_KEY = os.environ.get('SECRET_KEY') or os.urandom(16)
    DEBUG = False
    TESTING = False

    # Mysql
    # SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://root:123456@localhost/test'
    # SQLALCHEMY_COMMIT_ON_TEARDOWN = True
    # SQLALCHEMY_TRACK_MODIFICATIONS = True

    CELERY_BROKER_URL = "redis://:123456@127.0.0.1:8723/10"
    result_backend = "redis://:123456@127.0.0.1:8723/10"

    MONGODB_SETTINGS = {
        # "host": "mongodb://mongo:73c74b9461f00453@10.220.139.135:7830/prophet_k8s_test?authSource=admin",
        "host": "mongodb://127.0.0.1:27017/prophet_k8s_test",
    }

    # redis
    REDIS_URL = "redis://:123456@127.0.0.1:8723/0"


class ProductionConfig(BaseConfig):
    pass


class PreConfig(BaseConfig):
    pass


class DevelopmentConfig(BaseConfig):
    pass


class TestingConfig(BaseConfig):
    pass


class LocalConfig(BaseConfig):
    pass


registered_app = [
    'app'
]

config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'pre': PreConfig,
    'testing': TestingConfig,
    'local': LocalConfig
}
