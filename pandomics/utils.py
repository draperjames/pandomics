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

import functools
import pandas
from pandas.core.dtypes.common import is_integer, is_hashable
import numpy as np


try:
    from scipy.stats import ttest_ind
except ImportError as err:
    print(err)

try:
    from statsmodels.sandbox.stats.multicomp import multipletests
except ImportError as err:
    print(err)


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


def log2_normalize(self, prepend_cols=None, append_cols=None, *args, **kwargs):
    """Return a Log2 normalized DataFrame Log2(values)-mean(Log2(values))."""

    result = self.log2().sub(self.log2().mean(axis=1), axis=0)

    if prepend_cols:
        result.columns = prepend_cols + result.columns

    if append_cols:
        result.columns = result.columns + append_cols

    return result


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


def fold_change(self, right=None, numerator=None, denominator=None, filter_out_numerator=False,
                filter_out_denominator=False, column_name="FC", axis=1):

    """Return the fold change of two groups in this DataFrame or this DataFrame and another(right).

    When making comparisons between this DataFrame and another, set the `right`
    kwarg to the other DataFrame, do not use the numerator and denominator
    kwargs.

    Parameters
    ----------
    numerator: str

    denominator: str

    right: pandas.DataFrame

    column_name: str

    axis: int

    Returns
    -------
    result: pandas.DataFrame
    """

    if right is not None:
        left = self

    else:
        if numerator is not None:
            if filter_out_numerator:
                left = self.filter_out(regex=numerator)
            else:
                left = self.filter(regex=numerator)

        if denominator is not None:
            if filter_out_denominator:
                right = self.filter_out(regex=denominator)
            else:
                right = self.filter(regex=denominator)

    result = left.mean(axis=axis) - right.mean(axis=axis)
    result = pandas.DataFrame(result, index=self.index, columns=[column_name])

    return result


setattr(pandas.DataFrame, "fold_change", fold_change)


def ttest(self, right=None, numerator=None, denominator=None, filter_out_numerator=False,
          filter_out_denominator=False, column_name="pvalue", axis=1):

    """Return the p-value of two groups in this DataFrame or this DataFrame and another(right).

    The numerator and denominator must be defined to make a comparison inside of
    a DataFrame.

    When making comparisons between this DataFrame and another, set the `right`
    kwarg to the other DataFrame, do not use the numerator and denominator
    kwargs.

    Parameters
    ----------
    numerator: str

    denominator: str

    right: pandas.DataFrame

    column_name: str

    axis: int

    Returns
    -------
    result: pandas.DataFrame
    """

    if right is not None:
        left = self

    else:
        if numerator is not None:
            if filter_out_numerator:
                left = self.filter_out(regex=numerator)
            else:
                left = self.filter(regex=numerator)

        if denominator is not None:
            if filter_out_denominator:
                right = self.filter_out(regex=denominator)
            else:
                right = self.filter(regex=denominator)

    # The loop below suppresses an irrelevent error message.
    # For more details on this see:
    # http://stackoverflow.com/questions/40452765/invalid-value-in-less-when-comparing-np-nan-in-an-array
    with np.errstate(invalid='ignore'):
        np.less([np.nan, 0], 1)
        # ttest_ind implemented
        result = ttest_ind(left, right, axis=axis).pvalue

    result = pandas.DataFrame(result, columns=[column_name], index=self.index)

    return result


setattr(pandas.DataFrame, "ttest", ttest)


def ttest_fdr(self, numerator=None, denominator=None, filter_out_numerator=False,
              filter_out_denominator=False, right=None, column_name="pvalue",
              alpha=None, method="fdr_bh", axis=1):
    """Return the p-value and p-adjusted of this dataframe and another or two groups inside of this dataframe.
    """
    alpha = alpha or .05
    result = self.ttest(right=right,
                        numerator=numerator,
                        denominator=denominator,
                        filter_out_numerator=filter_out_numerator,
                        filter_out_denominator=filter_out_denominator,
                        column_name=column_name,
                        axis=axis)

    fdr_funct = functools.partial(multipletests, method=method, alpha=alpha)

    p_adj = fdr_funct(pvals=result.dropna().values.T[0])

    p_adj = pandas.DataFrame(p_adj[1])

    p_adj.index = result.dropna().index

    p_adj = p_adj.reindex(index=result.index)

    p_adj.columns = ["p_adjusted"]

    result = pandas.concat([result, p_adj], axis=axis)

    return result


setattr(pandas.DataFrame, "ttest_fdr", ttest_fdr)


def fold_change_with_ttest(self, numerator=None, denominator=None, right=None,
                           filter_out_numerator=False, filter_out_denominator=False,
                           fdr_alpha=None, fdr_method="fdr_bh", axis=1):
    """Return the fold change, p-values, and p-adjusted for a comparison.
    """

    fold_change = self.fold_change(numerator=numerator,
                                   denominator=denominator,
                                   filter_out_numerator=filter_out_numerator,
                                   filter_out_denominator=filter_out_denominator,
                                   right=right,
                                   axis=axis)

    pvalue = self.ttest_fdr(numerator=numerator,
                            denominator=denominator,
                            filter_out_numerator=filter_out_numerator,
                            filter_out_denominator=filter_out_denominator,
                            right=right,
                            alpha=fdr_alpha,
                            method=fdr_method,
                            axis=axis)

    result = pandas.concat([fold_change, pvalue], axis=axis)
    return result


setattr(pandas.DataFrame, "fold_change_with_ttest", fold_change_with_ttest)


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


def filter_rows(self, on=None, term=None):
    """Return DataFrame that contains a given term(str) on specific column.
    """
    filtered = self
    try:
        filtered = self[self[on].str.contains(term)]

    except Exception as err:
        print("filter_rows failed:", err)

    return filtered


setattr(pandas.DataFrame, 'filter_rows', filter_rows)


def subtract_by_matrix(self, other_dataframe=None, prepend_cols=None, append_cols=None, *args, **kwargs):
    """Returns DataFrame that has been subtracted by another DataFrame using the as_matrix method.

    This method can be invoked instead of the pandas.DataFrame.subtract method.
    Which does not subtract matrices in the usual way.
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
            result.columns = result.columns + append_cols

    except Exception as err:
        print("Could not subtract by matrix.", err)

    return result


setattr(pandas.DataFrame, 'subtract_by_matrix', subtract_by_matrix)


# -----------------
# VOLCANOPLOT CLASS
#------------------


class VolcanoPlot(pandas.plotting._core.PlanePlot):
    _kind = 'volcano'

    def __init__(self, data, x=None, y=None, s=None, c=None, **kwargs):

        if x is None:
            if "FC" in data:
                x="FC"

            elif "LFC" in data:
                x="LFC"

        if y is None:
            if "negative_log10_pvalue" in data:
                y="negative_log10_pvalue"

            elif "pvalue" in data:
                data["negative_log10_pvalue"] = -np.log10(data["pvalue"])
                y="negative_log10_pvalue"

        if s is None:
            # hide the matplotlib default for size, in case we want to change
            # the handling of this argument later
            s = 20

        super(VolcanoPlot, self).__init__(data, x, y, s=s, **kwargs)

        if is_integer(c) and not self.data.columns.holds_integer():
            c = self.data.columns[c]
        self.c = c

    def _make_plot(self):
        x, y, c, data = self.x, self.y, self.c, self.data
        ax = self.axes[0]

        c_is_column = is_hashable(c) and c in self.data.columns

        # plot a colorbar only if a colormap is provided or necessary
        cb = self.kwds.pop('colorbar', self.colormap or c_is_column)

        # pandas uses colormap, matplotlib uses cmap.
        cmap = self.colormap or 'Greys'
        cmap = self.plt.cm.get_cmap(cmap)
        color = self.kwds.pop("color", None)

        if c is not None and color is not None:
            raise TypeError('Specify exactly one of `c` and `color`')

        elif c is None and color is None:
            c_values = self.plt.rcParams['patch.facecolor']

        elif color is not None:
            c_values = color

        elif c_is_column:
            c_values = self.data[c].values

        else:
            c_values = c

        if self.legend and hasattr(self, 'label'):
            label = self.label

        else:
            label = None

        # Scatter plot called
        scatter = ax.scatter(data[x].values, data[y].values, c=c_values,
                             label=label, cmap=cmap, **self.kwds)

        if cb:
            img = ax.collections[0]
            kws = dict(ax=ax)
            if self.mpl_ge_1_3_1():
                kws['label'] = c if c_is_column else ''
            self.fig.colorbar(img, **kws)

        if label is not None:
            self._add_legend_handle(scatter, label)

        else:
            self.legend = False

        errors_x = self._get_errorbars(label=x, index=0, yerr=False)
        errors_y = self._get_errorbars(label=y, index=0, xerr=False)

        if len(errors_x) > 0 or len(errors_y) > 0:
            err_kwds = dict(errors_x, **errors_y)
            err_kwds['ecolor'] = scatter.get_facecolor()[0]
            ax.errorbar(data[x].values, data[y].values,
                        linestyle='none', **err_kwds)

# Patch VolcanoPlot pandas.plotting._core

# Set VolcanoPlot as an attribute of pandas.plotting._core
setattr(pandas.plotting._core, "VolcanoPlot", VolcanoPlot)

# Create the volcano helper function
def volcano(self, x=None, y=None, s=None, c=None, **kwds):
    """
    Create a volcano scatter plot with varying marker point size and color.

    The coordinates of each point are defined by two dataframe columns and
    filled circles are used to represent each point. This kind of plot is
    useful to see complex correlations between two variables. Points could
    be for instance natural 2D coordinates like longitude and latitude in
    a map or, in general, any pair of metrics that can be plotted against
    each other.

    Parameters
    ----------
    x : int or str
        The column name or column position to be used as horizontal
        coordinates for each point. Defaults to FC or LFC if present.

    y : int or str
        The column name or column position to be used as vertical
        coordinates for each point. Defaults to negative_log10_pvalue or will
        take the -np.log10 of the pvalue if it is present.

    s : scalar or array_like, optional
        The size of each point. Possible values are:

        - A single scalar so all points have the same size.

        - A sequence of scalars, which will be used for each point's size
          recursively. For instance, when passing [2,14] all points size
          will be either 2 or 14, alternatively.

    c : str, int or array_like, optional
        The color of each point. Possible values are:

        - A single color string referred to by name, RGB or RGBA code,
          for instance 'red' or '#a98d19'.

        - A sequence of color strings referred to by name, RGB or RGBA
          code, which will be used for each point's color recursively. For
          intance ['green','yellow'] all points will be filled in green or
          yellow, alternatively.

        - A column name or position whose values will be used to color the
          marker points according to a colormap.

    **kwds
        Keyword arguments to pass on to :meth:`pandas.DataFrame.plot`.

    Returns
    -------
    axes : :class:`matplotlib.axes.Axes` or numpy.ndarray of them

    See Also
    --------
    matplotlib.pyplot.scatter : scatter plot using multiple input data
        formats.

    Examples
    --------
    Let's see how to draw a scatter plot using coordinates from the values
    in a DataFrame's columns.

    .. plot::
        :context: close-figs

        >>> df = pd.DataFrame([[5.1, 3.5, 0], [4.9, 3.0, 0]],
        ...                   columns=['FC', 'negative_log10_pvalue'])
        ...
        >>> ax1 = df.plot.volcano()

    And now with the color determined by a column as well.

    .. plot::
        :context: close-figs

        >>> ax2 = df.plot.volcano(colormap='viridis')
    """
    return self(kind='volcano', x=x, y=y, c=c, s=s, **kwds)


# Set the helper function
setattr(pandas.plotting._core.FramePlotMethods, "volcano", volcano)
# Append the class to pandas.plotting._core._klasses
pandas.plotting._core._klasses.append(pandas.plotting._core.VolcanoPlot)
# Append to dataframe kinds.
pandas.plotting._core._dataframe_kinds.append("volcano")
# Append to all kinds.
pandas.plotting._core._all_kinds.append("volcano")
# Add the class to the pandas.plotting._core._plot_klass dict
pandas.plotting._core._plot_klass[VolcanoPlot._kind] = pandas.plotting._core.VolcanoPlot
