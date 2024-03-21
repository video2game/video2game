#!/usr/bin/env python3
# -*- coding:utf-8-*-
import torch

class global_var():
    @staticmethod
    def _init():
        global _global_dict
        _global_dict = {}
        _global_dict['log_dL_dsigmas'] = torch.zeros([1])
        _global_dict['log_dL_dnormals_diff'] = torch.zeros([1])
        _global_dict['log_dL_dnormals_ori'] = torch.zeros([1])
    @staticmethod
    def set_value(key, value):
        _global_dict[key] = value
    @staticmethod
    def get_value(key):
        return _global_dict[key]