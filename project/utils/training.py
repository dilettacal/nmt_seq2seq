"""
This script contains methods to train the model and to modify gradients.

Code inspirations:
- Ben Trevett 2018: https://github.com/bentrevett/pytorch-seq2seq/

"""
import os
import time
from typing import Any, Optional

import torch
import numpy as np
import math

from project.utils.constants import UNK_TOKEN, EOS_TOKEN, SOS_TOKEN, PAD_TOKEN
from project.utils.utils import convert_time_unit, AverageMeter
from settings import DEFAULT_DEVICE, SEED
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pandas as pd

import random
random.seed(SEED)


class CustomReduceLROnPlateau(ReduceLROnPlateau):

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 verbose=False, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, min_lr=0, eps=1e-8):
        super().__init__(optimizer=optimizer,mode=mode,factor=factor,
                                                      patience=patience,verbose=verbose,threshold=threshold,
                                                      threshold_mode=threshold_mode, cooldown=cooldown,min_lr=min_lr, eps=eps)

        self.TOTAL_LR_DECAYS = 0

    def step(self, metrics, epoch=None):
        #Source: https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#ReduceLROnPlateau
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            ### keep track of decays ####
            self.TOTAL_LR_DECAYS +=1
            ##########################
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def get_total_decays(self):
        return self.TOTAL_LR_DECAYS


def train_model(train_iter, val_iter, model, criterion, optimizer, scheduler, epochs, SRC, TRG, logger=None,
                device=DEFAULT_DEVICE, tr_logger=None, samples_iter=None, check_translations_every=5, beam_size=5,
                char_level=False):
    best_bleu_score = 0
    metrics = dict()
    train_losses = []
    train_ppls = []
    nltk_bleus =[]
    bleus = dict()
    last_avg_loss = 100
    check_transl_every = check_translations_every if epochs <= 80 else check_translations_every*2
    mini_samples = [batch for i, batch in enumerate(samples_iter) if i < 3]
    CHECKPOINT = 20
    TOLERANCE = 30
    TOLERATE_DECAYS = 3
    no_metric_improvements = 0
    print("Validation Beam: ", beam_size)

    for epoch in range(epochs):
        start_time = time.time()
        avg_train_loss = train(train_iter=train_iter, model=model, criterion=criterion, optimizer=optimizer,
                               device=device)
        avg_bleu_val = validate(val_iter=val_iter, model=model, device=device, TRG=TRG, beam_size=beam_size)

        train_losses.append(avg_train_loss)
        train_ppl = math.exp(avg_train_loss)
        train_ppls.append(train_ppl)
        nltk_bleus.append(avg_bleu_val)
        bleu = avg_bleu_val
        ### scheduler monitors val loss value
        scheduler.step(bleu)  # input bleu score
        if bleu > best_bleu_score:
            best_bleu_score = bleu
            logger.save_model(model.state_dict())
            logger.log('New best BLEU: {:.3f}'.format(best_bleu_score))
            no_metric_improvements = 0
        else:
            if scheduler.get_total_decays() >= TOLERATE_DECAYS:
                no_metric_improvements +=1
            if avg_train_loss < last_avg_loss:
                if epoch % CHECKPOINT == 0:
                    logger.save_model(model.state_dict())
                    logger.log('Training Checkpoint - BLEU: {:.3f}'.format(bleu))

        last_avg_loss = avg_train_loss #update checkpoint loss to last avg loss

        if epoch % check_transl_every == 0:
            #### checking translations
            if samples_iter:
                tr_logger.log("Translation check. Epoch {}".format(epoch + 1))
                check_translation(mini_samples, model, SRC, TRG, tr_logger, char_level=char_level)


        end_epoch_time = time.time()

        total_epoch = convert_time_unit(end_epoch_time - start_time)

        logger.log('Epoch: {} | Time: {}'.format(epoch + 1, total_epoch))
        logger.log(f'\tTrain Loss: {avg_train_loss:.3f} | Train PPL: {train_ppl:7.3f} | Val. BLEU: {bleu:.3f}')

        metrics.update({"loss": train_losses, "ppl": train_ppls})
        bleus.update({'nltk': nltk_bleus})

        if no_metric_improvements >= TOLERANCE:
            logger.log("No training improvements in the last {} epochs. Training stopped.".format(TOLERANCE))
            break

    return bleus, metrics


def train(train_iter, model, criterion, optimizer, device="cuda"):

    model.train()
    losses = AverageMeter()
    train_iter.init_epoch()

    for i, batch in enumerate(train_iter):

        #print(device)
        # Use GPU
        src = batch.src.to(device)
        trg = batch.trg.to(device)

        # Forward, backprop, optimizer
        model.zero_grad()
        scores = model(src, trg) #teacher forcing during training.

        scores = scores[:-1]
        trg = trg[1:]

        # Reshape for loss function
        scores = scores.view(scores.size(0) * scores.size(1), scores.size(2))
        #print(scores.requires_grad)
        trg = trg.view(scores.size(0))

        # Pass through loss function
        loss = criterion(scores, trg)
        loss.backward()
        losses.update(loss.item())
        # Clip gradient norms and step optimizer
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
    return losses.avg


def validate(val_iter, model, device, TRG, beam_size=5):
    model.eval()
    val_iter.init_epoch()

    # Iterate over words in validation batch.
    bleu = AverageMeter()
    sent_candidates = []  # list of sentences from decoder
    sent_references = []  # list of target sentences

    clean_tokens = [TRG.vocab.stoi[PAD_TOKEN], TRG.vocab.stoi[SOS_TOKEN], TRG.vocab.stoi[EOS_TOKEN]]
    with torch.no_grad():
        for i, batch in enumerate(val_iter):
            # Use GPU
            src = batch.src.to(device)
            trg = batch.trg.to(device)
            # Get model prediction (from beam search)
            out = model.predict(src, max_len=trg.size(0), beam_size=beam_size)  ### the beam value is the best value from the baseline study
            # print(out.size())
            ref = list(trg.data.squeeze())
            # Prepare sentence for bleu script
            out = [w for w in out if w not in clean_tokens]
            ref = [w for w in ref if w not in clean_tokens]
            sent_out = ' '.join(TRG.vocab.itos[j] for j in out)
            sent_ref = ' '.join(TRG.vocab.itos[j] for j in ref)
            sent_candidates.append(sent_out)
            sent_references.append(sent_ref)

        # smoothing technique for any cases
        smooth = SmoothingFunction()
        nlkt_bleu = corpus_bleu(list_of_references=[[sent.split()] for sent in sent_references],
                                hypotheses=[hyp.split() for hyp in sent_candidates],
                                smoothing_function=smooth.method4) * 100
        bleu.update(nlkt_bleu)

    return bleu.val

def beam_predict(model, data_iter, device, beam_size, TRG, max_len=30, char_level=False):
    model.eval()
    sent_candidates = []
    sent_references = []
    with torch.no_grad():
        for i, batch in enumerate(data_iter):
            src = batch.src.to(device)
            tgt = batch.trg.to(device)
            #### BLEU
            # compute scores with greedy search
            out = model.predict(src, beam_size=beam_size, max_len=max_len)  # out is a list

            ## Prepare sentences for BLEU
            ref = list(tgt.data.squeeze())
            # Prepare sentence for bleu script
            remove_tokens = [TRG.vocab.stoi[PAD_TOKEN], TRG.vocab.stoi[SOS_TOKEN], TRG.vocab.stoi[EOS_TOKEN]]
            out = [w for w in out if w not in remove_tokens]
            ref = [w for w in ref if w not in remove_tokens]
            if not char_level:
                sent_out = ' '.join(TRG.vocab.itos[j] for j in out)
                sent_ref = ' '.join(TRG.vocab.itos[j] for j in ref)
            else:
                sent_out = ''.join(TRG.vocab.itos[j] for j in out)
                sent_ref = ''.join(TRG.vocab.itos[j] for j in ref)
            sent_candidates.append(sent_out)
            sent_references.append(sent_ref)

    smooth = SmoothingFunction()  # if there are less than 4 ngrams
    nlkt_bleu = corpus_bleu(list_of_references=[[sent.split()] for sent in sent_references],
                            hypotheses=[hyp.split() for hyp in sent_candidates],
                            smoothing_function=smooth.method4) * 100
    # print("BLEU", batch_bleu)
    return nlkt_bleu

def check_translation(samples, model, SRC, TRG, logger,persist=False, char_level=False):
    """
    Readapted from Luke Melas Machine-Translation project:
    https://github.com/lukemelas/Machine-Translation/blob/master/training/train.py#L50
    :param src:
    :param trg:
    :param model:
    :param SRC:
    :param TRG:
    :param logger:
    :return:
    """
    if not samples:
        return
    logger.log("*" * 100,stdout=False)


    all_src, all_trg, all_beam1, all_beam2, all_beam5, all_beam10 = [],[],[],[], [],[]
    final_translations = None

    for i, batch in enumerate(samples):
        logger.log("Batch {}".format(str(i)), stdout=False)
        src = batch.src.to(model.device)
        trg = batch.trg.to(model.device)
        for k in range(src.size(1)): #actually src.size(1) is always set to 1
            src_bs1 = src.select(1, k).unsqueeze(1)
            trg_bs1 = trg.select(1, k).unsqueeze(1)
            model.eval()  # predict mode
            predictions = model.predict(src_bs1, beam_size=1, max_len=trg_bs1.size(0))
            predictions_beam = model.predict(src_bs1, beam_size=2, max_len=trg_bs1.size(0))
            predictions_beam5 = model.predict(src_bs1, beam_size=5,max_len=trg_bs1.size(0))
            predictions_beam10 = model.predict(src_bs1, beam_size=10,max_len=trg_bs1.size(0))

            #model.train()  # test mode
            #probs, maxwords = torch.max(scores.data.select(1, k), dim=1)  # training mode
            if not char_level:
                src_sent = ' '.join(SRC.vocab.itos[x] for x in src_bs1.squeeze().data)
                trg_sent = ' '.join(TRG.vocab.itos[x] for x in trg_bs1.squeeze().data)
                beam1 = ' '.join(TRG.vocab.itos[x] for x in predictions)
                beam2 = ' '.join(TRG.vocab.itos[x] for x in predictions_beam)
                beam5 = ' '.join(TRG.vocab.itos[x] for x in predictions_beam5)
                beam10 = ' '.join(TRG.vocab.itos[x] for x in predictions_beam10)
            else:
                src_sent = ''.join(SRC.vocab.itos[x] for x in src_bs1.squeeze().data)
                trg_sent = ''.join(TRG.vocab.itos[x] for x in trg_bs1.squeeze().data)
                beam1 = ''.join(TRG.vocab.itos[x] for x in predictions)
                beam2 = ''.join(TRG.vocab.itos[x] for x in predictions_beam)
                beam5 = ''.join(TRG.vocab.itos[x] for x in predictions_beam5)
                beam10 = ''.join(TRG.vocab.itos[x] for x in predictions_beam10)

            logger.log('Source: {}'.format(src_sent), stdout=False)
            logger.log('Target: {}'.format(trg_sent), stdout=False)
            logger.log('Validation Greedy Pred: {}'.format(beam1),stdout=False)
            logger.log('Validation Beam (2) Pred: {}'.format(beam2),stdout=False)
            logger.log('Validation Beam (5) Pred: {}'.format(beam5),stdout=False)
            logger.log('Validation Beam (10) Pred: {}'.format(beam10),stdout=False)
            logger.log("",stdout=False)

            if persist:
                all_src.append(src_sent)
                all_trg.append(trg_sent)
                all_beam1.append(beam1)
                all_beam2.append(beam2)
                all_beam5.append(beam5)
                all_beam10.append(beam10)
        logger.log("*"*100, stdout=False)
    if persist:
        logger.log("Total checks: {}".format(len(all_trg)))
        final_translations = dict({"SRC": all_src, "TRG":all_trg, "BEAM1": all_beam1, "BEAM2":all_beam2, "BEAM5": all_beam5, "BEAM10": all_beam10})
        filename = os.path.join(logger.path, "final.csv")
        df = pd.DataFrame(final_translations, columns=final_translations.keys())
        df.to_csv(filename, sep=",", columns=final_translations.keys(), encoding="utf-8")



def predict_from_input(model, input_sentence, SRC, TRG, logger, device="cuda", stdout=False, beam_size = 5, max_len=30):

    #### Changed from original ###
    sent_indices = [SRC.vocab.stoi[word] if word in SRC.vocab.stoi else SRC.vocab.stoi[UNK_TOKEN] for word in input_sentence]
    sent = torch.LongTensor([sent_indices])
    sent = sent.to(device)
    sent = sent.view(-1,1) # reshape to sl x bs
    logger.log('SRC  >>> ' + ' '.join([SRC.vocab.itos[index] for index in sent_indices]), stdout=stdout)
    ### predict sentences with beam search 5
    pred = model.predict(sent, beam_size=beam_size, max_len=max_len)
    pred = [index for index in pred if index not in [TRG.vocab.stoi[SOS_TOKEN], TRG.vocab.stoi[EOS_TOKEN]]]
    out = ' '.join(TRG.vocab.itos[idx] for idx in pred)
    logger.log('PRED >>> ' + out, stdout=stdout)
    return out


#################### gradient utility methods #################

def get_gradient_norm(m):
    total_norm = 0
    for p in m.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def customized_clip_value(parameters, norm_value):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    for p in filter(lambda p: p.grad is not None, parameters):
        p.grad = (5*p.grad)/norm_value


def get_gradient_statistics(model):
    parameters = list(filter(lambda p: p.grad is not None, model.parameters()))
    min_stats = min(p.grad.data.min() for p in parameters)
    max_stats = max(p.grad.data.max() for p in parameters)
    mean_stats = np.mean([p.grad.mean().cpu().numpy() for p in parameters])

    return {"min": min_stats, "max": max_stats, "mean": mean_stats}


###### other functions #######

### validate with teacher forcing (not used in any experiment)
def validate_scores_tf(val_iter, model, criterion, logger):
    '''
    Computes standard teacher forcing on the validation set
    :param val_iter:
    :param model:
    :param criterion:
    :param logger:
    :return:
    '''
    model.eval()
    losses = AverageMeter()
    for i, batch in enumerate(val_iter):
        src = batch.src.to(model.device)
        trg = batch.trg.to(model.device)
        # Forward
        scores = model(src, trg)
        scores = scores[:-1]
        trg = trg[1:]
        # Reshape for loss function
        scores = scores.view(scores.size(0) * scores.size(1), scores.size(2))
        trg = trg.view(scores.size(0))
        num_words = (trg != 0).float().sum()
        # Calculate loss
        loss = criterion(scores, trg)
        losses.update(loss.data[0])
    logger.log('Average loss on validation: {:.3f}'.format(losses.avg))
    return losses.avg

