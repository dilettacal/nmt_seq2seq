"""
This script contains methods to train the model and to modify gradients.

Code inspirations:
- Luke Melas: https://lukemelas.github.io/machine-translation.html
- Ben Trevett 2018: https://github.com/bentrevett/pytorch-seq2seq/

"""

import time

import torch
import numpy as np
import math

from project.utils.constants import UNK_TOKEN, EOS_TOKEN, SOS_TOKEN, PAD_TOKEN
from project.utils.utils import convert, AverageMeter
from settings import DEFAULT_DEVICE
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


## Truncated backpropagation
def detach_states(states):
    # https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/language_model/main.py#L59
    # a lighter implementation of 'repackage_hidden' from: https://github.com/pytorch/examples/blob/master/word_language_model/main.py#L103
    if states is None:
        return states
    return [state.detach() for state in states]


def train_model(train_iter, val_iter, model, criterion, optimizer, scheduler, epochs, TRG, logger=None, device=DEFAULT_DEVICE, model_type="custom", max_len=30):
    best_valid_loss = float('inf')
    best_bleu_value = 0

    bleus = []
    losses = dict()
    train_losses = []
    val_losses = []
    ppl = dict()
    train_ppls = []
    val_ppls = []

    for epoch in range(epochs):
        start_time = time.time()
        avg_train_loss = train(train_iter=train_iter, model=model, criterion=criterion, optimizer=optimizer,device=device, model_type=model_type, logger=logger)
        #val_iter, model, criterion, device, TRG,
        avg_val_loss, avg_bleu_loss = validate(val_iter, model, criterion, device, TRG, max_len=max_len)

        val_losses.append(avg_val_loss)
        train_losses.append(avg_train_loss)

        bleus.append(avg_bleu_loss)

        ### scheduler monitors BLEU value
        scheduler.step(avg_bleu_loss)  # input bleu score

        if avg_bleu_loss > best_bleu_value:
            best_bleu_value = avg_bleu_loss
            logger.save_model(model.state_dict())
            logger.log('New best BLEU value: {:.3f}'.format(best_bleu_value))

        if avg_bleu_loss < best_valid_loss:
            best_valid_loss = avg_val_loss
            logger.log('New best validation value: {:.3f}'.format(best_valid_loss))

        end_epoch_time = time.time()
        total_epoch = convert(end_epoch_time-start_time)

        val_ppl = math.exp(avg_val_loss)
        train_ppl = math.exp(avg_train_loss)

        val_ppls.append(val_ppl)
        train_ppls.append(train_ppl)

        logger.log('Epoch: {} | Time: {}'.format(epoch+1, total_epoch))
        logger.log(f'\tTrain Loss: {avg_train_loss:.3f} | Train PPL: {train_ppl:7.3f}')
        logger.log(f'\t Val. Loss: {avg_val_loss:.3f} |  Val. PPL: {val_ppl:7.3f} | Val. BLEU: {avg_bleu_loss:.3f}')

    losses.update({"train": train_losses, "val": val_losses})
    ppl.update({"train": train_ppls, "val": val_ppls})

    return bleus, losses, ppl



def train(train_iter, model, criterion, optimizer, device="cuda", model_type="custom", logger=None):
   # print(device)
    norm_changes = 0
    # Train model
    model.train()
    losses = AverageMeter()
    for i, batch in enumerate(train_iter):
        #print(device)
        # Use GPU
        src = batch.src.to(device)
        trg = batch.trg.to(device)

        # Forward, backprop, optimizer
        model.zero_grad()
        scores = model(src, trg.detach())
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

        ### Sutskever gradient clipping type
        if model_type == "s":
            ### this should clip the norm to the range [10, 25] as in the paper
            grad_norm = get_gradient_norm(model)
            if grad_norm > 5:
                norm_changes += 1
                customized_clip_value(model.parameters(), grad_norm)

        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


    logger.log("Gradient Norm Changes: {}".format(norm_changes))
    return losses.avg



def validate(val_iter, model, criterion, device, TRG, beam_size = 2, max_len=30):
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

    smooth = SmoothingFunction()
    nlkt_bleu = corpus_bleu(list_of_references=[[sent.split()] for sent in sent_references], hypotheses=[hyp.split() for hyp in sent_candidates], smoothing_function=smooth.method4) *100
    return losses.avg, nlkt_bleu


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

def flatten_tensor(t):
    t = t.reshape(1, -1)
    t = t.squeeze()
    return t
