#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/8 8:43 下午
# @Author  : qinbinbin
# @email   : qinbinbin@360.cn
# @File    : xwz_model.py

from mongoengine import Document, StringField, IntField, BooleanField, ListField


class Xwz(Document):
    episode = StringField(db_field="话数")
    jp_title = StringField(db_field="日文标题")
    cn_title = StringField(db_field="台湾标题")
    series = IntField(db_field="季")
    tags = ListField()
    tags_src = StringField()
    part = IntField(db_field="上下集")

    original_date = StringField(db_field="首播日期")
    animation_script = StringField(db_field="脚本")
    animation_script_supervisor = StringField(db_field="脚本监督")
    animation_supervisor = StringField(db_field="作画监督")
    animation_drawing = StringField(db_field="絵コンテ")
    animation_performances = StringField(db_field="演出")

    # cn_title = StringField()
    # age = IntField()
    # is_delete = BooleanField(default=False)

    @property
    def str_id(self):
        return str(self.id) if self.id else None

    @classmethod
    def get_by_id(cls, model_id):
        return cls.objects.get(id=model_id)

    @classmethod
    def get_by_key(cls, text):
        print("text")
        return cls.objects.search_text(text=text).limit(10)

    def to_dict(self):
        return {
            "id": self.str_id,
            "话数": self.episode,
            "季": self.series,
            "标题": self.cn_title,
            "上下集": "上" if self.part == 0 else "下",
        }

    @classmethod
    def get_users(cls, kwargs):
        offset = kwargs.pop("offset")
        per_page = kwargs.pop("per_page")

        objs = [obj.to_dict() for obj in cls.objects(**kwargs).order_by("-create_time").skip(offset).limit(per_page)]
        return cls.objects(**kwargs).count(), objs
