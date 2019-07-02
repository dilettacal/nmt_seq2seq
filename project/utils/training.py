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
from settings import DEFAULT_DEVICE, SEED, TEACHER_RATIO
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

import random
random.seed(SEED)


def train_model(train_iter, val_iter, model, criterion, optimizer, scheduler, epochs, SRC, TRG, logger=None, device=DEFAULT_DEVICE, model_type="custom", max_len=30):
    best_valid_loss = float('inf')
    best_ppl_value = float('inf')

    losses = dict()
    train_losses = []
    val_losses = []
    ppl = dict()
    train_ppls = []
    val_ppls = []

    nltk_bleus, perl_bleus = [], []
    bleus = dict()

    for epoch in range(epochs):
        start_time = time.time()
        compute_bleu = True if epoch % 5 == 0 else False
        check_tr = True if epoch % 10 == 0 else False
        avg_train_loss = train(train_iter=train_iter, model=model, criterion=criterion,
                               optimizer=optimizer,device=device, model_type=model_type, logger=logger, check_trans=check_tr, SRC=SRC, TRG=TRG,)
        avg_val_loss,  avg_bleu_val = validate(val_iter, model, criterion, device, TRG, bleu=compute_bleu)

        val_losses.append(avg_val_loss)
        train_losses.append(avg_train_loss)
        if compute_bleu:
            nltk_bleus.append(avg_bleu_val[0])
            perl_bleus.append(avg_bleu_val[1])

        val_ppl = math.exp(avg_val_loss)

        if val_ppl < best_ppl_value:
            best_ppl_value = val_ppl
            logger.save_model(model.state_dict())
            logger.log('New best perplexity value: {:.3f}'.format(best_ppl_value))

        end_epoch_time = time.time()
        total_epoch = convert(end_epoch_time-start_time)


        train_ppl = math.exp(avg_train_loss)

        ### scheduler monitors val loss value
        scheduler.step(val_ppl)  # input bleu score

        val_ppls.append(val_ppl)
        train_ppls.append(train_ppl)

        logger.log('Epoch: {} | Time: {}'.format(epoch+1, total_epoch))
        logger.log(f'\tTrain Loss: {avg_train_loss:.3f} | Train PPL: {train_ppl:7.3f}')
        if compute_bleu:
            nlkt_b = avg_bleu_val[0]
            perl_b = avg_bleu_val[1]
            #perl_b = 0
            logger.log(f'\t Val. Loss: {avg_val_loss:.3f} |  Val. PPL: {val_ppl:7.3f} | Val. (nlkt) BLEU: {nlkt_b:.3f} | Val. (perl) BLEU: {perl_b:.3f} |')
        else:
            logger.log(f'\t Val. Loss: {avg_val_loss:.3f} |  Val. PPL: {val_ppl:7.3f}')

    losses.update({"train": train_losses, "val": val_losses})
    ppl.update({"train": train_ppls, "val": val_ppls})
    bleus.update({'nltk': nltk_bleus, 'perl': perl_bleus})

    return bleus, losses, ppl



def train(train_iter, model, criterion, optimizer, SRC, TRG, device="cuda", model_type="custom", logger=None, check_trans=False):
   # print(device)
    norm_changes = 0
    # Train model
    model.train()
    losses = AverageMeter()

    for i, batch in enumerate(train_iter):

        condition = i == 0 or i == len(train_iter)//2 or i == len(train_iter)
        #print(device)
        # Use GPU
        src = batch.src.to(device)
        trg = batch.trg.to(device)
        if check_trans and condition:
            src_copy = src.clone()
            trg_copy = trg.clone()

        # Forward, backprop, optimizer
        model.zero_grad()
        scores = model(src, trg) #teacher forcing during training.

        if check_trans and condition:
            raw_scores = scores.clone()
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
        if check_trans and condition:
            num_translation = 5
            src_to_translate = src_copy[:, :num_translation]
            trg_to_translate = trg_copy[:, :num_translation]
            check_translation(src_to_translate, trg_to_translate, raw_scores,model,SRC=SRC, TRG=TRG, logger=logger)

    logger.log("Gradient Norm Changes: {}".format(norm_changes))
    return losses.avg



def validate(val_iter, model, criterion, device, TRG, bleu=False):
    model.eval()
    losses = AverageMeter()
    bleus = AverageMeter()
    perl_bleus = AverageMeter()
    sent_candidates = []
    sent_references = []

    with torch.no_grad():
        for i, batch in enumerate(val_iter):
            src = batch.src.to(device)
            trg = batch.trg.to(device)

            batch_size = trg.size(1)
            #### compute model scores
            ### These are still raw computations from the linear output layer from the model
            ### As CrossEntropyLoss wrapper was used for loss computation
            ### the log softmax operation is done by this object internally
            scores = model(src, trg)
            scores = scores[:-1]
            trg = trg[1:]

            raw_scores = scores.clone() ### for BLEU
            raw_trg = trg.clone() ### for BLEU

            # Reshape for loss function
            scores = scores.view(scores.size(0) * scores.size(1), scores.size(2))

            trg = trg.view(scores.size(0))

            # Calculate loss
            loss = criterion(scores, trg)
            # save loss
            losses.update(loss.item())

            #### check the BLEU value for the batch ####

            if bleu:
                #print("Computing BLEU score for batch {}...".format(i))
                for seq_idx in range(batch_size):

                    ### raw_trg = [seq_len, batch_size]
                    ### with select(1, seq_idx), we get at each step the sequence at seq_idx in the batch, thus trg_seq = [seq_len]
                    trg_seq = raw_trg.select(1, seq_idx)  ### adding a dimension at index 1

                    #### Greedy search - the max value for each candidate is chosen #####

                    ### For each sequence in the batch, the model delivers a probability distribution over the trg vocabulary
                    ### We select the probabilities for each sequence (seq_idx) and computes the max, thus we get a sequence of word idx (as they are stored in the voc)
                    ### max_idx = [seq_len]

                    probs, max_idx = torch.max(raw_scores.data.select(1, seq_idx), dim=1)

                    ### Convert each index to a string
                    candidate = [TRG.vocab.itos[x] for x in max_idx]

                    #### BLEU ####

                    ### References are the real target sequences
                    ### To pass them the NLTK function "corpus_bleu" they need to be bidimensional, thus we add a dimension to trg_seq and convert them to a list

                    reference = list(trg_seq.data)

                    ### For the BLEU computation we do not want to consider special tokens (SOS, EOS and PAD)
                    clean_tokens = [TRG.vocab.stoi[PAD_TOKEN], TRG.vocab.stoi[SOS_TOKEN], TRG.vocab.stoi[EOS_TOKEN]]

                    ### cleaning the prediction
                    candidate = [w for w in candidate if w not in clean_tokens]

                    ### cleaning here the reference
                    reference = [w for w in reference if w not in clean_tokens]

                    sent_out = ' '.join(candidate)
                    sent_ref = ' '.join(TRG.vocab.itos[j] for j in reference)
                    sent_candidates.append(sent_out)
                    sent_references.append(sent_ref)

                ### smoothing functions allow to avoid the problem of missing n-gram overlappings
                smooth = SmoothingFunction()

                ### Computing corpus bleu for this batch
                batch_bleu = corpus_bleu(list_of_references=[[sent.split()] for sent in sent_references],
                                            hypotheses=[hyp.split() for hyp in sent_candidates],
                                            smoothing_function=smooth.method4) * 100

                try:
                    perl_bleu = get_moses_multi_bleu(sent_candidates, sent_references)
                except TypeError or Exception as e:
                    print("Perl BLEU score set to 0. \tException in perl script: {}".format(e))
                    perl_bleu = 0
                bleus.update(batch_bleu)
                perl_bleus.update(perl_bleu)


        if bleu:
           return losses.avg, [bleus.avg, perl_bleus.avg]

        else:
           return losses.avg, -1




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

def check_translation(src, trg, scores, model, SRC, TRG, logger):
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
    remove_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]
    logger.log("*" * 100,stdout=False)

    samples = src.size(1)
    for k in range(src.size(1)):
        logger.log("Sequence {}".format(str(k)),stdout=False)
        src_bs1 = src.select(1, k).unsqueeze(1)  # bs1 means batch size 1
        trg_bs1 = trg.select(1, k).unsqueeze(1)
        model.eval()  # predict mode
        predictions = model.predict(src_bs1, beam_size=1)
        predictions_beam = model.predict(src_bs1, beam_size=2)
        predictions_beam5 = model.predict(src_bs1, beam_size=5)
        predictions_beam12 = model.predict(src_bs1, beam_size=12)

        model.train()  # test mode
        probs, maxwords = torch.max(scores.data.select(1, k), dim=1)  # training mode

        logger.log('Source: {}'.format(' '.join(SRC.vocab.itos[x] for x in src_bs1.squeeze().data)), stdout=False)
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

