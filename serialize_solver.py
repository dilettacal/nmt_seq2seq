import os

from settings import RESULTS_DIR
from translate import translate

dirs = os.listdir(os.path.join(RESULTS_DIR,"no_cat/s/4/uni/"))
print(dirs)


for dir in dirs:
    try:
        translate(root=os.path.join(RESULTS_DIR, "no_cat/s/4/uni/"), path=dir)
    except Exception as e:
        print(e)