#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/3 12:06 下午
# @Author  : qinbinbin
# @email   : qinbinbin@360.cn
# @File    : celery_task.py
import logging

from celery_extensions import my_celery as celery

# from flask_log_request_id.extras.celery import enable_request_id_propagation
from flask_log_request_id import current_request_id

# enable_request_id_propagation(celery)


@celery.task()
def add(a, b):
    """Simple function to add two numbers that is not aware of the request id"""

    logging.info('Called generic_add({}, {}) from request_id: {}'.format(a, b, current_request_id()))
    return a + b
