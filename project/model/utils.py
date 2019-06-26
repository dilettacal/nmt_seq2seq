import torch
from torch import nn


def init_weights_uniform(m, lowerbound, upperbound):
    for name, param in m.named_parameters():
        ### sutskever: -0.08, 0.08
        nn.init.uniform_(param.data,lowerbound, upperbound)

def init_weights_normal(m, mean, std):
    for name, param in m.named_parameters():
        ### cho: mean=0, std=0.01
        nn.init.normal_(param.data, mean=mean, std=std)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def init_weights(m):
    if m.__class__.__name__ == "ContextSeq2Seq":
        init_weights_normal(m, mean=0, std=0.01)
    elif m.__class__.__name__ == "StandardSeq2Seq":
        init_weights_uniform(m, -0.08, 0.08)


# Based on https://github.com/MaximumEntropy/Seq2Seq-PyTorch/
class Beam(object):
    """Ordered beam of candidate outputs."""

    def __init__(self, size, pad=1, bos=2, eos=3, cuda=False):
        """Initialize params."""
        self.size = size
        self.done = False
        self.pad = pad
        self.bos = bos
        self.eos = eos
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size).fill_(self.pad)]
        self.nextYs[0][0] = self.bos

    # Get the outputs for the current timestep.
    def get_current_state(self):
        """Get state of beam."""
        return self.nextYs[-1]

    # Get the backpointers for the current timestep.
    def get_current_origin(self):
        """Get the backpointer to the beam at this step."""
        return self.prevKs[-1]

    def advance(self, workd_lk):
        print(workd_lk.shape)
        """Advance the beam."""
        num_words = workd_lk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beam_lk = workd_lk + self.scores.unsqueeze(1).expand_as(workd_lk)
        else:
            beam_lk = workd_lk[0]

        flat_beam_lk = beam_lk.view(-1)

        bestScores, bestScoresId = flat_beam_lk.topk(self.size, 0,
                                                     True, True)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = bestScoresId / num_words
        self.prevKs.append(prev_k)
        self.nextYs.append(bestScoresId - prev_k * num_words)
        # End condition is when top-of-beam is EOS.
        if self.nextYs[-1][0] == self.eos:
            self.done = True
        return self.done

    def get_hyp(self, k):
        """Get hypotheses."""
        hyp = []
        # print(len(self.prevKs), len(self.nextYs), len(self.attn))
        ### right to left
        for j in range(len(self.prevKs) - 1, -1, -1):
            #### navigate nextYs till the second element (index 1) in the list
            hyp.append(self.nextYs[j + 1][k])
            k = self.prevKs[j][k]
        return hyp[::-1]

