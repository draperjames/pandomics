# -*- coding: utf-8 -*-

import os
import inspect


JPY = bool([i for i in os.environ if i[:3] == "JPY"])
SPY = bool([i for i in os.environ if i[:3] == "SPY"])


def bokeh_nb(**kwargs):
    """Has the same args as bokeh.io.output_notebook
    """
    if JPY:
        from bokeh.io import output_notebook

        if "hide_banner" in kwargs:
            pass
        else:
            kwargs["hide_banner"] = True

        output_notebook(**kwargs)

    else:
        pass

try:
    from bokeh.io import output_notebook
    # Apply the signature to bokeh_nb
    bokeh_nb.__signature__ = inspect.signature(output_notebook)
    del output_notebook

except Exception as err:
    print(err)
