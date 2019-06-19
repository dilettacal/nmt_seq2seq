import os, sys
sys.path += [os.path.abspath('.')]

from os.path import dirname, join

PROJECT_ROOT = dirname(dirname(__file__))


def get_full_path(*path):
    return join(PROJECT_ROOT, *path)