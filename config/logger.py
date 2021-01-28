#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/7 7:36 下午
# @Author  : qinbinbin
# @email   : qinbinbin@360.cn
# @File    : logger.py
from logging.config import dictConfig


def config_logger(enable_console_handler=True, enable_file_handler=True, log_file='logs/app.log', log_level='ERROR',
                  log_file_max_bytes=5000000, log_file_max_count=5, disable_existing_loggers=False):
    console_handler = {
        'class': 'logging.StreamHandler',
        'formatter': 'default',
        'level': log_level,
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'filters': ['request_id']
    }
    file_handler = {
        'class': 'logging.handlers.RotatingFileHandler',
        'formatter': 'detail',
        'filename': log_file,
        'level': log_level,
        'maxBytes': log_file_max_bytes,
        'backupCount': log_file_max_count,
        'filters': ['request_id']
    }
    default_formatter = {
        'format': "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] [request_id: %(request_id)s] %(message)s"
    }
    detail_formatter = {
        'format': "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] [request_id: %(request_id)s] %(message)s"
    }
    handlers = []
    if enable_console_handler:
        handlers.append('console')
    if enable_file_handler:
        handlers.append('file')
    d = {
        'version': 1,
        'filters': {
            'request_id': {
                '()': 'flask_log_request_id.RequestIDLogFilter'
            }
        },
        'formatters': {
            'default': default_formatter,
            'detail': detail_formatter
        },
        'handlers': {
            'console': console_handler,
            'file': file_handler
        },
        'root': {
            'level': log_level,
            'handlers': handlers,
        },
        'disable_existing_loggers': disable_existing_loggers
    }
    dictConfig(d)
