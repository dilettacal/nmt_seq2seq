'''
This file contains classes and functions readapted from:
https://github.com/amake/tmx2corpus

The following is the License Text from that repository:

The MIT License (MIT)

Copyright (c) 2015 Aaron Madlon-Kay

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

##############################################################################
Script description:

A script to convert TMXs into parallel corpuses for machine
translation (e.g. Moses: http://www.statmt.org/moses/) training.

Pass in either paths to TMX files, or directories containing TMX files.
The script will recursively traverse directories and process all TMXs.

To perform tokenization or to filter the output, use the convert() method
with subclasses of the Tokenizer or Filter objects.
@author: aaron.madlon-kay

###############################################################################
'''

import os
import codecs
import logging
logging.basicConfig(filename='converter.log',level=logging.DEBUG)
from xml.etree import ElementTree

try:
    from HTMLParser import HTMLParser
    unescape = HTMLParser().unescape
except ModuleNotFoundError:
    import html
    unescape = html.unescape


class FileOutput(object):
    def __init__(self, path=os.getcwd()):
        self.files = {}
        self.path = path
        logging.debug('Output path: %s', self.path)

    def init(self, language):
        if language not in self.files:
            self.files[language] = codecs.open(os.path.join(
                self.path, 'bitext.' + language), 'w', encoding='utf-8')

    def write(self, language, content):
        out_file = self.files[language]
        out_file.write(content)
        out_file.write('\n')

    def cleanup(self):
        for out_file in self.files.values():
            out_file.close()
        self.files.clear()

class Converter(object):
    """
    This object converts a bitext to plain text.
    Slightly modified from the original version.
    No filtering and tokenizing function, only simple conversion task.
    """
    def __init__(self, output:FileOutput):
        self.suppress_count = 0
        self.output = output
        self.output_lines = 0

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.output.cleanup()

    def convert(self, files):
        self.suppress_count = 0
        self.output_lines = 0
        for tmx in files:
            print('Extracting %s' % os.path.basename(tmx))
            for bitext in extract_tmx(tmx):
                self.__output(bitext)
        logging.debug('Output %d pairs', self.output_lines)
        if self.suppress_count:
            logging.debug('Suppressed %d pairs', self.suppress_count)
        return True

    def __output(self, bitext):

        for lang in bitext.keys():
            self.output.init(lang)

        for lang, text in bitext.items():
            self.output.write(lang, text)

        self.output_lines += 1


def get_files(path, ext):
    for root, dirs, files in os.walk(path):
        for a_file in files:
            if a_file.endswith(ext):
                yield os.path.join(root, a_file)


def extract_tmx(tmx):
    for event, elem in ElementTree.iterparse(tmx):
        if elem.tag == 'tu':
            bitext = extract_tu(elem)
            if bitext:
                yield bitext


def extract_tu(tu):
    bitext = {}
    for tuv in tu.findall('tuv'):
        lang, text = extract_tuv(tuv)
        if None not in (lang, text):
            bitext[lang] = text
    if len(bitext) != 2:
        logging.debug('TU had %d TUV(s). Skipping.', len(bitext))
        logging.debug('\t' + ElementTree.tostring(tu))
        return {}
    return bitext


def extract_tuv(tuv):
    lang = tuv.attrib.get('lang', None)
    if lang == None:
        lang = tuv.attrib.get(
            '{http://www.w3.org/XML/1998/namespace}lang', None)
    if lang == None:
        logging.debug('TUV missing lang. Skipping.')
        return None, None
    lang = normalize_lang(lang)
    segs = tuv.findall('seg')
    if len(segs) > 1:
        logging.debug('Multiple segs found in TUV. Skipping.')
        return None, None
    text = extract_seg(segs[0])
    if text is None:
        logging.debug('TUV missing seg. Skipping.')
        return None, None
    text = clean_text(text)
    if not text:
        logging.debug('TUV had blank seg. Skipping.')
        return None, None
    return lang, text


def extract_seg(seg):
    buffer = [seg.text]
    for child in seg:
        buffer.append(child.text)
        buffer.append(child.tail)
    return ''.join([piece for piece in buffer if piece != None])


def clean_text(text):
    text = text.strip().replace('\n', '').replace('\r', '')
    return unescape(text)


def normalize_lang(lang):
    result = lang.lower()
    if len(result) > 2 and result[2] in ('-', '_'):
        result = result[:2]
    return result


def convert(paths,output=None):
    """
    Converts tmx files from the given paths
    Slightly modified from the original version:
    - Filters and Tokenizers removed
    :param paths:
    :param output:
    :return:
    """
    files = set()
    for path in sorted(set(os.path.abspath(p) for p in paths)):
        if os.path.isdir(path):
            tmxs = set(get_files(path, '.tmx'))
            logging.info('Queuing %d TMX(s) in %s', len(tmxs), path)
            files |= tmxs
        elif os.path.isfile(path) and path.endswith('.tmx'):
            files.add(path)
    if files:
        with Converter(output or FileOutput()) as converter:
            converter.convert(sorted(files))
    else:
        logging.error('Please specify input files or paths.')
        return 1
    return 0


def glom_urls(tokens):
    result = []
    in_url = False
    url = None
    tokens.reverse()
    while len(tokens):
        tok = tokens.pop()
        if in_url:
            if tok[0] == '<' or tok[0] > u'\u007F':
                result.append(url)
                result.append(tok)
                in_url = False
            else:
                url += tok
                if not len(tokens):
                    result.append(url)
        else:
            if tok in ('://', '@'):
                url = (result.pop() if len(result) else '') + tok
                in_url = True
            else:
                result.append(tok)
    return result