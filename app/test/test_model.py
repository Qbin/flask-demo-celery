#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/8 8:43 下午
# @Author  : qinbinbin
# @email   : qinbinbin@360.cn
# @File    : test_model.py

from mongoengine import Document, StringField, IntField, BooleanField


class TestUser(Document):
    username = StringField()
    age = IntField()
    is_delete = BooleanField(default=False)

    @property
    def str_id(self):
        return str(self.id) if self.id else None

    @classmethod
    def get_by_id(cls, model_id):
        return cls.objects.get(id=model_id)

    @classmethod
    def add_user(cls, **kwargs):
        obj = cls(**kwargs)
        obj.save()
        return obj.to_dict()

    def update_user(self, username, age):
        if username:
            self.username = username
        if age:
            self.age = age
        self.save()
        return self

    def delete_user(self):
        self.is_delete = True
        self.save()
        return self.str_id

    def to_dict(self):
        return {
            "id": self.str_id,
            "username": self.username,
            "age": self.age
        }

    @classmethod
    def get_users(cls, kwargs):
        offset = kwargs.pop("offset")
        per_page = kwargs.pop("per_page")

        objs = [obj.to_dict() for obj in cls.objects(**kwargs).order_by("-create_time").skip(offset).limit(per_page)]
        return cls.objects(**kwargs).count(), objs
