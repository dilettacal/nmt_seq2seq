"""
This script contains methods to train the model and to modify gradients.

Code inspirations:
- Ben Trevett 2018: https://github.com/bentrevett/pytorch-seq2seq/

"""

import time

import torch
import numpy as np
import math

from project.utils.bleu import get_moses_multi_bleu
from project.utils.constants import UNK_TOKEN, EOS_TOKEN, SOS_TOKEN, PAD_TOKEN
from project.utils.utils import convert, AverageMeter
from settings import DEFAULT_DEVICE, SEED
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

import random
random.seed(SEED)


def train_model(train_iter, val_iter, model, criterion, optimizer, scheduler, epochs, SRC, TRG, logger=None,
                device=DEFAULT_DEVICE, tr_logger = None, samples_iter = None, log_every=5):
    best_bleu_score = 0

    metrics = dict()
    train_losses = []
    train_ppls = []

    nltk_bleus, perl_bleus = [], []
    bleus = dict()

    valid_every = log_every

    for epoch in range(epochs):

        start_time = time.time()
        if epoch == (epochs-1):
            samples = [batch for i, batch in enumerate(samples_iter)]
        else:
            samples = [batch for i, batch in enumerate(samples_iter) if i < 3]

        if epoch % valid_every == 0:
            tr_logger.log("Translation check. Epoch {}".format(epoch+1))
            avg_train_loss = train(train_iter=train_iter, model=model, criterion=criterion,
                                   optimizer=optimizer, device=device, logger=logger,
                                   SRC=SRC, TRG=TRG, samples=samples, tr_logger=tr_logger)

            train_losses.append(avg_train_loss)
            train_ppl = math.exp(avg_train_loss)
            train_ppls.append(train_ppl)

            avg_bleu_val = validate(val_iter=val_iter, model=model, device=device, TRG=TRG)
            nltk_bleus.append(avg_bleu_val[0])
            perl_bleus.append(avg_bleu_val[1])
            bleu = avg_bleu_val[0]
            perl_b = avg_bleu_val[1]

            ### scheduler monitors val loss value
            scheduler.step(bleu)  # input bleu score

            if bleu > best_bleu_score:
                best_bleu_score = bleu
                logger.save_model(model.state_dict())
                logger.log('New best BLEU: {:.3f}'.format(best_bleu_score))

            end_epoch_time = time.time()

            total_epoch = convert(end_epoch_time - start_time)

            logger.log('Epoch: {} | Time: {}'.format(epoch + 1, total_epoch))
            logger.log(f'\tTrain Loss: {avg_train_loss:.3f} | Train PPL: {train_ppl:7.3f} | Val. BLEU: {bleu:.3f} | Val. (perl) BLEU: {perl_b:.3f}')

            metrics.update({"loss": train_losses, "ppl": train_ppls})
            bleus.update({'nltk': nltk_bleus, 'perl': perl_bleus})

        else:
            end_epoch_time = time.time()

            avg_train_loss = train(train_iter=train_iter, model=model, criterion=criterion,
                                   optimizer=optimizer, device=device, logger=logger,
                                   SRC=SRC, TRG=TRG, samples=None, tr_logger=tr_logger)

            train_losses.append(avg_train_loss)
            train_ppl = math.exp(avg_train_loss)
            train_ppls.append(train_ppl)
            total_epoch = convert(end_epoch_time - start_time)

            logger.log('Epoch: {} | Time: {}'.format(epoch + 1, total_epoch))
            logger.log(f'\tTrain Loss: {avg_train_loss:.3f} | Train PPL: {train_ppl:7.3f}')

            metrics.update({"loss": train_losses, "ppl": train_ppls})

    return bleus, metrics



def train(train_iter, model, criterion, optimizer, SRC, TRG, device="cuda", logger=None, samples = None, tr_logger=None):
   # print(device)
    norm_changes = 0
    # Train model
    model.train()
    losses = AverageMeter()

    for i, batch in enumerate(train_iter):

        condition = i == len(train_iter)-1
        #print(device)
        # Use GPU
        src = batch.src.to(device)
        trg = batch.trg.to(device)

        # Forward, backprop, optimizer
        model.zero_grad()
        scores = model(src, trg) #teacher forcing during training.

        if samples:
            raw_scores = scores.clone()
        else: raw_scores = None
        # Remove <s> from trg and </s> from scores
        scores = scores[:-1]
        trg = trg[1:]

        # Reshape for loss function
        scores = scores.view(scores.size(0) * scores.size(1), scores.size(2))
        trg = trg.view(scores.size(0))

        # Pass through loss function
        loss = criterion(scores, trg)
        loss.backward()
        losses.update(loss.item())
        # Clip gradient norms and step optimizer
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        if condition:
            check_translation(samples, raw_scores, model,SRC=SRC, TRG=TRG, logger=tr_logger)

    return losses.avg


def validate(val_iter, model, device, TRG):
    model.eval()

    # Iterate over words in validation batch.
    bleu = AverageMeter()
    perl = AverageMeter()
    sent_candidates = []  # list of sentences from decoder
    sent_references = []  # list of target sentences

    clean_tokens = [TRG.vocab.stoi[PAD_TOKEN], TRG.vocab.stoi[SOS_TOKEN], TRG.vocab.stoi[EOS_TOKEN]]
    with torch.no_grad():
        for i, batch in enumerate(val_iter):
            # Use GPU
            src = batch.src.to(device)
            trg = batch.trg.to(device)


            # Get model prediction (from beam search)
            out = model.predict(src, max_len=trg.size(0),
                                beam_size=1)  # list of ints (word indices) from greedy search
            # print(out.size())
            ref = list(trg.data.squeeze())
            # Prepare sentence for bleu script
            out = [w for w in out if w not in clean_tokens]
            ref = [w for w in ref if w not in clean_tokens]
            sent_out = ' '.join(TRG.vocab.itos[j] for j in out)
            sent_ref = ' '.join(TRG.vocab.itos[j] for j in ref)
            sent_candidates.append(sent_out)
            sent_references.append(sent_ref)

            # Get model prediction (from beam search)
            out = model.predict(src, beam_size=1)  # list of ints (word indices) from greedy search
            ref = list(trg.data.squeeze())
            out = [w for w in out if w not in clean_tokens]
            ref = [w for w in ref if w not in clean_tokens]
            sent_out = ' '.join(TRG.vocab.itos[j] for j in out)
            sent_ref = ' '.join(TRG.vocab.itos[j] for j in ref)
            sent_candidates.append(sent_out)
            sent_references.append(sent_ref)

        smooth = SmoothingFunction()  # if there are less than 4 ngrams
        nlkt_bleu = corpus_bleu(list_of_references=[[sent.split()] for sent in sent_references],
                                hypotheses=[hyp.split() for hyp in sent_candidates],
                                smoothing_function=smooth.method4) * 100
        bleu.update(nlkt_bleu)
        try:
            perl_bleu = get_moses_multi_bleu(sent_candidates, sent_references)
            perl.update(perl_bleu)
        except TypeError or Exception as e:
            print("Perl BLEU score set to 0. \tException in perl script: {}".format(e))
            perl_bleu = 0
            perl.update(perl_bleu)

    return [bleu.val, perl.val]


def validate_test_set(val_iter, model, criterion, device, TRG, beam_size = 1, max_len=30):
    model.eval()
    losses = AverageMeter()
    sent_candidates = []
    sent_references = []
    with torch.no_grad():
        for i, batch in enumerate(val_iter):
            src = batch.src.to(device)
            tgt = batch.trg.to(device)
            ### compute normal scores
            scores = model(src, tgt)

            scores = scores[:-1]
            tgt = tgt[1:]

            # Reshape for loss function
            scores = scores.view(scores.size(0) * scores.size(1), scores.size(2))


            ##### top1 = output.max(1)[1]
            tgt = tgt.view(scores.size(0))

            # Calculate loss
            loss = criterion(scores, tgt)
            losses.update(loss.item())

            #### BLEU
            # compute scores with greedy search
            out = model.predict(src, beam_size=beam_size, max_len=max_len)  # out is a list

            ## Prepare sentences for BLEU
            ref = list(tgt.data.squeeze())
            # Prepare sentence for bleu script
            remove_tokens = [TRG.vocab.stoi[PAD_TOKEN], TRG.vocab.stoi[SOS_TOKEN], TRG.vocab.stoi[EOS_TOKEN]]
            out = [w for w in out if w not in remove_tokens]
            ref = [w for w in ref if w not in remove_tokens]
            sent_out = ' '.join(TRG.vocab.itos[j] for j in out)
            sent_ref = ' '.join(TRG.vocab.itos[j] for j in ref)
            sent_candidates.append(sent_out)
            sent_references.append(sent_ref)

    smooth = SmoothingFunction() #if there are less than 4 ngrams
    nlkt_bleu = corpus_bleu(list_of_references=[[sent.split()] for sent in sent_references],
                            hypotheses=[hyp.split() for hyp in sent_candidates],
                            smoothing_function=smooth.method4)*100
    try:
        perl_bleu = get_moses_multi_bleu(sent_candidates, sent_references)
    except TypeError or Exception as e:
        print("Perl BLEU score set to 0. \tException in perl script: {}".format(e))
        perl_bleu = 0

    # print("BLEU", batch_bleu)
    return losses.avg, [nlkt_bleu, perl_bleu]


def beam_predict(model, data_iter, device, beam_size, TRG, max_len=30):
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
            sent_out = ' '.join(TRG.vocab.itos[j] for j in out)
            sent_ref = ' '.join(TRG.vocab.itos[j] for j in ref)
            sent_candidates.append(sent_out)
            sent_references.append(sent_ref)

    smooth = SmoothingFunction()  # if there are less than 4 ngrams
    nlkt_bleu = corpus_bleu(list_of_references=[[sent.split()] for sent in sent_references],
                            hypotheses=[hyp.split() for hyp in sent_candidates],
                            smoothing_function=smooth.method4) * 100
    try:
        perl_bleu = get_moses_multi_bleu(sent_candidates, sent_references)
    except TypeError or Exception as e:
        print("Perl BLEU score set to 0. \tException in perl script: {}".format(e))
        perl_bleu = 0

    # print("BLEU", batch_bleu)
    return [nlkt_bleu, perl_bleu]

def check_translation(samples, scores, model, SRC, TRG, logger):
    """
    Readapted from Luke Melas Machine-Translation project:
    https://github.com/lukemelas/Machine-Translation/blob/master/training/train.py#L50
    :param src:
    :param trg:
    :param scores:
    :param model:
    :param SRC:
    :param TRG:
    :param logger:
    :return:
    """
    if not samples:
        return
    logger.log("*" * 100,stdout=False)

    for i, batch in enumerate(samples):
        logger.log("Batch {}".format(str(i)), stdout=False)
        src = batch.src.to(model.device)
        trg = batch.trg.to(model.device)
        for k in range(src.size(1)): #actually src.size(1) is always set to 1
            src_bs1 = src.select(1, k).unsqueeze(1)
            trg_bs1 = trg.select(1, k).unsqueeze(1)
            model.eval()  # predict mode
            predictions = model.predict(src_bs1, beam_size=1)
            predictions_beam = model.predict(src_bs1, beam_size=2)
            predictions_beam5 = model.predict(src_bs1, beam_size=5)
            predictions_beam12 = model.predict(src_bs1, beam_size=12)

            model.train()  # test mode
            probs, maxwords = torch.max(scores.data.select(1, k), dim=1)  # training mode
            src_sent = ' '.join(SRC.vocab.itos[x] for x in src_bs1.squeeze().data)

            logger.log('Source: {}'.format(src_sent), stdout=False)
            logger.log('Target: {}'.format(' '.join(TRG.vocab.itos[x] for x in trg_bs1.squeeze().data)), stdout=False)
            logger.log('Training Pred (Greedy): {}'.format(' '.join(TRG.vocab.itos[x] for x in maxwords)), stdout=False)
            logger.log('Validation Greedy Pred: {}'.format(' '.join(TRG.vocab.itos[x] for x in predictions)),stdout=False)
            logger.log('Validation Beam (2) Pred: {}'.format(' '.join(TRG.vocab.itos[x] for x in predictions_beam)),stdout=False)
            logger.log('Validation Beam (5) Pred: {}'.format(' '.join(TRG.vocab.itos[x] for x in predictions_beam5)),stdout=False)
            logger.log('Validation Beam (12) Pred: {}'.format(' '.join(TRG.vocab.itos[x] for x in predictions_beam12)),stdout=False)
            logger.log("",stdout=False)
        logger.log("*"*100, stdout=False)


def predict_from_input(model, input_sentence, SRC, TRG, logger, device="cuda"):
    input_sent = input_sentence.split(' ') # sentence --> list of words
    sent_indices = [SRC.vocab.stoi[word] if word in SRC.vocab.stoi else SRC.vocab.stoi[UNK_TOKEN] for word in input_sent]
    sent = torch.LongTensor([sent_indices])
    sent = sent.to(device)
    sent = sent.view(-1,1) # reshape to sl x bs
    logger.log('Source: ' + ' '.join([SRC.vocab.itos[index] for index in sent_indices]))
    # Predict five sentences with beam search
    preds = model.predict_k(sent, 5) # returns list of 5 lists of word indices
    for i, pred in enumerate(preds): # loop over top 5 sentences
        pred = [index for index in pred if index not in [TRG.vocab.stoi[SOS_TOKEN], TRG.vocab.stoi[EOS_TOKEN]]]
        out = str(i+1) + ': ' + ' '.join([TRG.vocab.itos[index] for index in pred])
        logger.log(out)
    return


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
        #p.grad.data.clamp_(min=clip_value[0], max=clip_value[1])
        p.grad = (5*p.grad)/norm_value


def get_gradient_statistics(model):
    parameters = list(filter(lambda p: p.grad is not None, model.parameters()))
    min_stats = min(p.grad.data.min() for p in parameters)
    max_stats = max(p.grad.data.max() for p in parameters)
    mean_stats = np.mean([p.grad.mean().cpu().numpy() for p in parameters])

    return {"min": min_stats, "max": max_stats, "mean": mean_stats}

