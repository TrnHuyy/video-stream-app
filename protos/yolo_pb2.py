# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: yolo.proto
# Protobuf Python Version: 5.26.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\nyolo.proto\x12\x04yolo\"\x15\n\x05Image\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\"+\n\x08Metadata\x12\x0e\n\x06status\x18\x01 \x01(\t\x12\x0f\n\x07message\x18\x02 \x01(\t\"?\n\rImageResponse\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\x12 \n\x08metadata\x18\x02 \x01(\x0b\x32\x0e.yolo.Metadata21\n\x04Yolo\x12)\n\x05Track\x12\x0b.yolo.Image\x1a\x13.yolo.ImageResponseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'yolo_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_IMAGE']._serialized_start=20
  _globals['_IMAGE']._serialized_end=41
  _globals['_METADATA']._serialized_start=43
  _globals['_METADATA']._serialized_end=86
  _globals['_IMAGERESPONSE']._serialized_start=88
  _globals['_IMAGERESPONSE']._serialized_end=151
  _globals['_YOLO']._serialized_start=153
  _globals['_YOLO']._serialized_end=202
# @@protoc_insertion_point(module_scope)
