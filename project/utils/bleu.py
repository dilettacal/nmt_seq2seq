# coding=utf-8

# Modifications copyright (C) 2019 dltcls
# The download part has been removed. The path to the perl script has been adapted to the project requirements.
#
# Copyright 2017 The Tensor2Tensor Authors.
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import re
import subprocess
import tempfile
import logging

import numpy as np

from settings import ROOT

logger = logging.getLogger(__name__)


def get_moses_multi_bleu(hypotheses, references, lowercase=False):

    """
    Computes the Bleu score for the given hypothesis and references.
    This method uses the perl script from mosesdecoder, available at:
     https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/multi-bleu.perl

    The python computation is taken from the Google seq2seq implementation.
    The download part has been removed, as it is not needed within this project.

    See original code at: https://github.com/google/seq2seq/blob/master/seq2seq/metrics/bleu.py

    :param hypotheses:
    :param references:
    :param lowercase:
    :return:
    """

    if isinstance(hypotheses, list):
        hypotheses = np.array(hypotheses)
    if isinstance(references, list):
        references = np.array(references)

    if np.size(hypotheses) == 0:
        return np.float32(0.0)


    multi_bleu_path = os.path.join(os.path.abspath(ROOT), "scripts")
    os.makedirs(multi_bleu_path, exist_ok=True)
    multi_bleu_path = os.path.join(multi_bleu_path, "multi-bleu.perl")
    # Dump hypotheses and references to tempfiles
    hypothesis_file = tempfile.NamedTemporaryFile()
    hypothesis_file.write("\n".join(hypotheses).encode("utf-8"))
    hypothesis_file.write(b"\n")
    hypothesis_file.flush()
    reference_file = tempfile.NamedTemporaryFile()
    reference_file.write("\n".join(references).encode("utf-8"))
    reference_file.write(b"\n")
    reference_file.flush()

    # Calculate BLEU using multi-bleu script
    with open(hypothesis_file.name, "r") as read_pred:
        bleu_cmd = [multi_bleu_path]
        if lowercase:
            bleu_cmd += ["-lc"]
        bleu_cmd += [reference_file.name]
        try:
            bleu_out = subprocess.check_output(bleu_cmd, stdin=read_pred, stderr=subprocess.STDOUT)
            bleu_out = bleu_out.decode("utf-8")
            bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
            bleu_score = float(bleu_score)
            bleu_score = np.float32(bleu_score)
        except subprocess.CalledProcessError as error:
            if error.output is not None:
                logger.warning("multi-bleu.perl script returned non-zero exit code")
                logger.warning(error.output)
            bleu_score = None

    # Close temp files
    hypothesis_file.close()
    reference_file.close()

    return bleu_score