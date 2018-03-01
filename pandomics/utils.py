# -*- coding: utf-8 -*-
"""Custom pandas tools.

Hotpatched functions for pandas.DataFrame
-------------------------------------
Hotpatching aka monkey-patching functions to an existing class is suggested
over subclassing functions. I prefer to use this method instead of pandas
pandas.DataFrame.pipe command because I think that it yields a simpler syntax for
for the lab rats.
"""

# LICENSE
# -------

# Copyright 2018 James Draper
#
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


import pandas
# from pandas import Series
# from pandas import DataFrame
import numpy as np


def first_member(self, delim=';'):
    """Return the first member of split on delim.

    Intended for pandas.Series that has lists in the form of strings
    with a foreign delimiter.

    WARNING: NaNs are cast as strings and cast back to floats. So if
    your first member happens to be 'nan' it will be cast as a float.

    Parameters
    ----------
    delim : str
        The delimeter to to perform the split on.

    Returns
    -------
    results : pandas.Series
        A Series of the first members of the list.
    """
    result = self.astype(str).apply(lambda x: x.split(delim)[0])

    result = result.replace('nan', float('nan'))

    return result


setattr(pandas.Series, 'first_member', first_member)


def log2(self):
    """Take the Log2 of all values in a DataFrame."""
    return self.apply(np.log2)


setattr(pandas.DataFrame, "log2", log2)


def log2_normalize(self):
    """Return a Log2 normalized DataFrame Log2(values)-mean(Log2(values))."""
    return self.log2().sub(self.log2().mean(axis=1), axis=0)


setattr(pandas.DataFrame, "log2_normalize", log2_normalize)


def compare_to(self, *args, **kwargs):
    """Subtract all columns of a DataFrame by specified column."""
    result = self.subtract(self.filter(*args, **kwargs).iloc[:, 0], axis=0)
    return result


setattr(pandas.DataFrame, "compare_to", compare_to)


def normalization_factors(self):
    """Compute normalization factors for a DataFrame."""
    norm_factors = self.sum() / self.sum().mean()
    return norm_factors


setattr(pandas.DataFrame, "normalization_factors", normalization_factors)


def normalize_to(self, normal):
    """Normalize a DataFrame to another DataFrame."""
    # Sort by columns.
    self_sort = self.columns.tolist()
    self_sort.sort()
    self = self[self_sort]
    # Sort the normal dataframe by it's columns.
    normal_sort = normal.columns.tolist()
    normal_sort.sort()
    normal = normal[normal_sort]
    # Divide by the normalization factors.
    normalized = self / normal.normalization_factors().as_matrix()
    normalized.columns = self.columns + ": Normalized to: " + normal.columns
    return normalized


setattr(pandas.DataFrame, "normalize_to", normalize_to)


@property
def label_pos(self):
    """Return a dict with column labels as keys and positions as values."""
    return dict(i[::-1] for i in enumerate(self.columns))


setattr(pandas.DataFrame, 'label_pos', label_pos)


def label_insert(self, label, how='Right', *args, **kwargs):
    """Insert an array-like into a DataFrame before that label.

    Parameters
    ----------
    label : str
        The label of th ecolumn to insert by.
    how : str
        Right or Left.
    value : Series
        New column.
    column : str
        New column name.
    """
    if how == 'Right':
        self.insert(int(self.label_pos[label]+1), *args, **kwargs)
    if how == 'Left':
        self.insert(int(self.label_pos[label]), *args, **kwargs)


setattr(pandas.DataFrame, 'label_insert', label_insert)


def filter_type(self, col, desired):
    """Filter a DataFrame by the type stored in a given column."""
    assert(type(self) is pandas.DataFrame)
    assert(type(col) is str)
    return self[self[col].apply(lambda x:type(x) == desired)]


setattr(pandas.DataFrame, 'filter_type', filter_type)


def filter_out(self, *args, **kwargs):
    """Filter like filter but the opposite."""

    filter_result = self.filter(*args, **kwargs)

    result = list(filter(lambda x:x not in set(filter_result.columns), self.columns))

    result = self[result]

    return result


setattr(pandas.DataFrame, 'filter_out', filter_out)


def subtract_by_matrix(self, other_dataframe=None, prepend_cols=None, append_cols=None, *args, **kwargs):
    """Returns DataFrame that has been subtracted by another DataFrame using the as_matrix method.
    """
    result = pandas.DataFrame()

    try:
        result = self.as_matrix() - other_dataframe.as_matrix()
        result = pandas.DataFrame(result)
        result.index = self.index
        result.columns = self.columns

        if prepend_cols:
            result.columns = prepend_cols + result.columns

        if append_cols:
            result.columns = append_cols + result.columns

    except Exception as err:
        print("Could not subtract by matrix.", err)

    return result


setattr(pandas.DataFrame, 'subtract_by_matrix', subtract_by_matrix)
