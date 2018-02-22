# -*- coding: utf-8 -*-
"""
pandomics

====

Provides
--------
FIXME: ADD COPY

How to use the documentation
----------------------------
FIXME: ADD COPY

Available subpackages
---------------------
FIXME: ADD COPY

Utilities
---------
FIXME: ADD COPY
"""

# LICENSE
# -------
# Copyright 2018 James Draper

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files, (the software)), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions: The above copyright
# notice and this permission notice shall be included in all copies or
# substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS",
# WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
# TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM. OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
# THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import os
from .utils import pandas


def find_path():
    """Find the location of omin package in any given file system."""
    __dir_path__ = os.path.dirname(os.path.realpath(__file__))
    return __dir_path__

with open(os.path.join(find_path(), '__version__')) as f:
    __version__ = f.read().strip()
