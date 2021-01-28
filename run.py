#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/7 7:43 下午
# @Author  : qinbinbin
# @email   : qinbinbin@360.cn
# @File    : run.py

import click
from envparse import env
from application import create_app
from celery_extensions import my_celery


@click.command()
@click.option('-h', '--host', help='Bind host', default='localhost', show_default=True)
@click.option('-p', '--port', help='Bind port', default=8000, type=int, show_default=True)
@click.option('-e', '--env', help='Running env, override environment FLASK_ENV.', default='development',
              show_default=True)
@click.option('-f', '--env-file', help='Environment from file', type=click.Path(exists=True))
def main(**kwargs):
    if kwargs['env_file']:
        env.read_envfile(kwargs['env_file'])
    app = create_app(kwargs['env'], celery=my_celery)
    app.run(host=kwargs['host'], port=kwargs['port'])


if __name__ == '__main__':
    main()
