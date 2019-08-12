"""
This is the main script to preprocess the Europarl dataset.
The script automatically downloads the dataset from the Opus Platform: http://opus.nlpl.eu/Europarl.php
All TMX files are stored in the section "Statistics and TMX/Moses Downloads".
The upper right triangle contains the tmx files. The lower left triangle the corresponding text files.

Raw files should be extracted to: data/raw/europarl/<lang_code>
"""

from project.utils.parsers.get_preprocess_parser import data_prepro_parser
from project.utils.preprocess import raw_preprocess

if __name__ == '__main__':
    raw_preprocess(data_prepro_parser().parse_args())
