import functools
import operator


def reduce_lists(list_of_lists):
    return functools.reduce(operator.iconcat, list_of_lists, [])
