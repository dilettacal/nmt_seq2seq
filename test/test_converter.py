import os
import unittest
from project.utils.external.tmx_to_text import FileOutput, Converter

data_dir = os.path.join(".", "test_data")
DE = "Hallo, Welt!"
EN = "Hello, world!"

class TestTMX2Corpus(unittest.TestCase):

    def test_tmx_extraction(self):

        output = FileOutput(path=data_dir)
        converter = Converter(output=output)
        file = os.path.join(data_dir, "test.tmx")
        success = converter.convert(files=[file])
        self.assertIs(success, True)
        # check files
        files = sorted([file for file in os.listdir(data_dir) if file.startswith("bitext")])
        self.assertIs(len(files), 2)

    def test_content(self):
        files = sorted([file for file in os.listdir(data_dir) if file.startswith("bitext")])
        de_line = open(os.path.join(data_dir, files[0])).read().strip("\n")
        set1 = set(de_line.split(' '))
        set2 = set(DE.split(' '))
        self.assertEqual(set1, set2)

        en_line = open(os.path.join(data_dir, files[1])).read().strip("\n")
        set1 = set(en_line.split(' '))
        set2 = set(EN.split(' '))
        self.assertEqual(set1, set2)
if __name__ == '__main__':
    unittest.main(verbosity=2)