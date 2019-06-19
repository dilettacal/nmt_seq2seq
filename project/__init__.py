import os
import sys
sys.path.append(os.path.join(".", "TMX2Corpus"))



from os.path import dirname, join

PROJECT_ROOT = dirname(dirname(__file__))


def get_full_path(*path):
    return join(PROJECT_ROOT, *path)