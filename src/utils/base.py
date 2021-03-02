import datetime
import time
from typing import Iterable
import re


def get_timestamp():
    r""" A function to get the current time as a formatted string.
    Returns
    -------
    timestamp : str
        The current time in the format YearMonthDay_HourMinutesSeconds.
    Examples
    --------
    If executed at the 28th of April, 2020 at 10h:17m:20s the function would return 20200428_101720.
    """
    date = datetime.datetime.strptime(time.ctime(), "%a %b %d %H:%M:%S %Y")
    timestamp = datetime.datetime.strftime(date, "%Y%m%d_%H%M%S")

    return timestamp


def key_in_dict(keys, dictionary):
    """Check if keys appear in dictionary or are None.
    Parameters
    ----------
    keys : str, list(str)
        Keys to be checked.
    dictionary : dict
        The dictionary which should be searched for the keys.
    Returns
    -------
    bool
        True f all keys appear in the dictionary and are not None and False otherwise.
    Raises
    ------
    AssertionError
        If keys is neither a string nor a list.
    """
    assert type(keys) in [str, list]
    if type(keys) == str:
        keys = [keys]
    for key in keys:
        if key not in dictionary.keys() or dictionary[key] is None:
            return False

    return True


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def sorted_nicely(l: Iterable):
    """ Sorts the given iterable in the way that is expected.

    Required arguments:
    l -- The iterable to be sorted.

    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)
