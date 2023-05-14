#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The Google Research Authors and The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for constructing PyTrees of PartitionSpecs."""

# utils adapted from https://github.com/google-research/google-research/blob/master/flax_models/t5x/partitions.py

import re

from flax.core.frozen_dict import freeze
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.experimental import PartitionSpec as P


# Sentinels
_unmatched = object()

# For specifying empty leaf dict `{}`
empty_dict = object()


# def _match(qs, ks):
#     """Return True if regexes in qs match any window of strings in tuple ks."""
#     # compile regexes and force complete match
#     qts = tuple(map(lambda x: re.compile(x + "$"), qs))
#     for i in range(len(ks) - len(qs) + 1):
#         matches = [x.match(y) for x, y in zip(qts, ks[i:])]
#         if matches and all(matches):
#             return True
#     return False
def _match(joinks, v):
    shape = v
    """Return True if regexes in qs match any window of strings in tuple ks."""
    # compile regexes and force complete match
    if len(shape) == 1:
      if shape[0] % 4 == 0:
        return P("dp")
      else:
        return P("mp")
    if len(shape) == 2:
      if shape[0] % 4 == 0 and shape[1] % 2 == 0:
        return P("dp","mp")
      if shape[0] % 2 == 0 and shape[1] % 4 == 0:
        return P("mp","dp")
      if shape[0] % 4 == 0:# and shape[1] % 2 == 0:
        return P("dp",None)
      if shape[1] % 4 == 0:# and shape[1] % 2 == 0:
        return P(None,"dp")
      if shape[0] % 2 == 0 and shape[1] % 2 == 0:
        return P("mp",None)
    if len(shape) == 4:
      if shape[-2] % 4 == 0 and shape[-1] % 2 == 0:
        return P(None,None,"dp","mp")
      if shape[-2] % 2 == 0 and shape[-1] % 4 == 0:
        return P(None,None,"mp","dp")
      if shape[-2] % 4 == 0:# and shape[1] % 2 == 0:
        return P(None,None,"dp",None)
      if shape[-1] % 4 == 0:# and shape[1] % 2 == 0:
        return P(None,None,None,"dp")
      if shape[-1] % 2 == 0 and shape[-2] % 2 == 0:
        return P(None,None,"mp",None)

    print("fail")
    return object()

def _replacement_rules(rules):
    def replace(key, val):
#         for rule, replacement in rules:
#             if _match(rule, key):
#                 return replacement
        return _match(key,val)

    return replace


# PartitionSpec for GPTNeo
# replicate the hidden dim and shard feed-forward and head dim
def _get_partition_rules():
    return [
        # embeddings
        (("transformer", "wpe", "embedding"), P("mp", None)),
        (("transformer", "wte", "embedding"), P("mp", None)),
        # atention
        (("attention", "(q_proj|k_proj|v_proj)", "kernel"), P(None, "mp")),
        (("attention", "out_proj", "kernel"), P("mp", None)),
        (("attention", "out_proj", "bias"), None),
        # mlp
        (("mlp", "c_fc", "kernel"), P(None, "mp")),
        (("mlp", "c_fc", "bias"), P("mp")),
        (("mlp", "c_proj", "kernel"), P("mp", None)),
        (("mlp", "c_proj", "bias"), None),
        # layer norms
        ((r"ln_\d+", "bias"), None),
        ((r"\d+", r"ln_\d+", "scale"), None),
        (("ln_f", "bias"), None),
        (("ln_f", "scale"), None),
    ]


def set_partitions(in_dict):
    rules = _get_partition_rules()
    replace = _replacement_rules(rules)
#     initd = {k: _unmatched for k in flatten_dict(in_dict)}
    result = {k: replace(k, v.shape) for k, v in flatten_dict(in_dict).items()}
    assert _unmatched not in result.values(), "Incomplete partition spec."
    return freeze(unflatten_dict(result))