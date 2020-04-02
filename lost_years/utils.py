# -*- coding: utf-8 -*-

import sys
import requests


def isstring(s):
    # if we use Python 3
    if (sys.version_info[0] >= 3):
        return isinstance(s, str)
    # we use Python 2
    return isinstance(s, basestring)


def column_exists(df, col):
    """Check the column name exists in the DataFrame.

    Args:
        df (:obj:`DataFrame`): Pandas DataFrame.
        col (str): Column name.

    Returns:
        bool: True if exists, False if not exists.

    """
    if col and (col not in df.columns):
        print("The specify column `{0!s}` not found in the input file"
              .format(col))
        return False
    else:
        return True


def fixup_columns(cols):
    """Replace index location column to name with `col` prefix

    Args:
        cols (list): List of original columns

    Returns:
        list: List of column names

    """
    out_cols = []
    for col in cols:
        if type(col) == int:
            out_cols.append('col{:d}'.format(col))
        else:
            out_cols.append(col)
    return out_cols


def closest(lst, c):
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i] - c))]


def download_file(url, local_path=None):
    if local_path is None:
        local_path= url.split('/')[-1]
    r = requests.get(url)
    with open(local_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=512 * 1024):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)