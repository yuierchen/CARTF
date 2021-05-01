#!/usr/bin/env python 
# -*- coding:utf-8 -*-
class Record:
    def __init__(self,package_name,class_name,class_description,method_name,method_annotation,method_returntype,method_paratype,method_block,method_block_flat,method_api_sequence,decompose_methodname):
        self.package_name=package_name
        self.class_name=class_name
        self.class_description=class_description
        self.class_description_words=None
        self.class_description_matrix=None
        self.class_description_idf_vector=None
        self.method_name=method_name
        self.method_annotation=method_annotation
        self.method_annotation_words = None
        self.method_annotation_matrix = None
        self.method_annotation_idf_vector = None
        self.method_returntype=method_returntype
        self.method_paratype=method_paratype
        self.method_block=method_block
        self.method_block_flat=method_block_flat
        self.method_api_sequence=method_api_sequence
        self.decompose_methodname=decompose_methodname
        self.decompose_methodname_words=None
        self.decompose_methodname_matrix=None
        self.decompose_methodname_idf_vector=None
        self.fullmethod_annotation=method_annotation+" "+decompose_methodname
        self.fullmethod_annotation_words = None
        self.fullmethod_annotation_matrix = None
        self.fullmethod_annotation_idf_vector = None



class Experiment:
    def __init__(self, method_annotation, now_method_flat, true_api,decompose_methodname,now_api):
        self.method_annotation=method_annotation
        self.now_method_flat=now_method_flat
        self.true_api=true_api
        self.decompose_methodname=decompose_methodname
        self.now_api=now_api
