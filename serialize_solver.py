import os
from settings import DATA_DIR_PREPRO

print(DATA_DIR_PREPRO)

path_to_bitext = os.path.join(DATA_DIR_PREPRO, "europarl", "de")
files = sorted([file for file in os.listdir(path_to_bitext) if file.endswith("tok.en") or file.endswith("tok.de")])
print(files)

de_lines = open(os.path.join(path_to_bitext, files[0]), mode="r").read().split("\n")
print(len(de_lines))


en_lines = open(os.path.join(path_to_bitext, files[1]), mode="r").read().split("\n")
print(len(en_lines))

longest_en = max(en_lines, key=lambda x: len(x))
longest_de = max(de_lines, key=lambda x: len(x))

shortest_en = min(en_lines, key=lambda x: len(x))
shortest_de = min(de_lines, key=lambda x: len(x))


print(len(list(longest_de))) # 3800
print(len(list(longest_en))) # 4113


en_sent_mapped = list(map(lambda x: len(list(x)), en_lines))
de_sent_mapped = list(map(lambda x: len(list(x)), de_lines))

from collections import Counter

EN_C = Counter(en_sent_mapped)
sorted_en = sorted(EN_C, key=EN_C.get, reverse=True)

print(EN_C.most_common(10))

DE_C = Counter(de_sent_mapped)
sorted_de = sorted(DE_C, key=DE_C.get, reverse=True)
print(DE_C.most_common(10))

import numpy as np
import matplotlib.pyplot as plt


labels, values = zip(*EN_C.items())

indexes = np.arange(len(labels))
width = 5

"""
plt.title("English Chars")
plt.bar(indexes, values, width)
plt.xticks(indexes + width * 0.5, labels)
#plt.show()
plt.close()
plt.savefig("en_chars.png")

labels, values = zip(*DE_C.items())

indexes = np.arange(len(labels))
width = 5

plt.title("German Chars")
plt.bar(indexes, values, width)
plt.xticks(indexes + width * 0.5, labels)
#plt.show()
plt.close()
plt.savefig("de_chars.png")
"""

n, bins, patches = plt.hist(x = labels, bins='auto', alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)

plt.xlabel('Sequence lengths')
plt.ylabel('Frequency')
plt.title('English sequence lengths')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.show()

