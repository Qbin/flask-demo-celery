#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/14 16:55
# @Author  : qinbinbin
# @email   : qinbinbin@360.cn
# @File    : models_model.py
from mongoengine import Document, StringField, DictField, ListField

from common.time_cost import calculate_runtime


class ClusterModels(Document):
    data_indexes = ListField(required=True)
    model_params = DictField(required=True)
    model_type = StringField(required=True)

    @property
    def str_id(self):
        if self.id:
            return str(self.id)

    @classmethod
    def add_model(cls, **kwargs):
        """
        data_indexes:list
        model_params: dict
        model_name: str
        """
        obj = cls(**kwargs)
        obj.save()
        return obj

    @classmethod
    def get_by_id(cls, model_id):
        return cls.objects.get(id=model_id)
