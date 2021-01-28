#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/3 11:53 上午
# @Author  : qinbinbin
# @email   : qinbinbin@360.cn
# @File    : celery_extensions.py

import os

from celery import Celery
from celery.signals import setup_logging

from config.config import config_map


@setup_logging.connect
def setup_logger(*args, **kwargs):
    from config.logger import config_logger
    log_level = os.environ.get('LOG_LEVEL') or 'INFO'
    log_file = os.environ.get('LOG_FILE') or 'logs/celery.log'
    config_logger(
        enable_console_handler=True,
        enable_file_handler=True,
        log_level=log_level,
        log_file=log_file
    )


def make_celery(app_name):
    conf = config_map.get(os.getenv('FLASK_ENV') or 'development')
    celery = Celery(
        app_name,
        backend=conf.result_backend,
        broker=conf.CELERY_BROKER_URL
    )
    return celery


my_celery = make_celery("FLASK")
