import time

import torch
import numpy as np
import math

from project.utils.bleu import get_moses_multi_bleu
from project.utils.constants import UNK_TOKEN, EOS_TOKEN, SOS_TOKEN, PAD_TOKEN
from project.utils.utils import convert, AverageMeter
from settings import DEFAULT_DEVICE
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def train_model(train_iter, val_iter, model, criterion, optimizer, scheduler, epochs, TRG, logger=None, device=DEFAULT_DEVICE):
    best_valid_loss = float('inf')

    for epoch in range(epochs):
        start_time = time.time()
        avg_train_loss = train(train_iter=train_iter, model=model, criterion=criterion, optimizer=optimizer,device=device)
        #val_iter, model, criterion, device, TRG,
        avg_val_loss, avg_bleu_loss = validate(val_iter, model, criterion, device, TRG)
        #print(avg_bleu_loss)
        scheduler.step(avg_val_loss)  # input bleu score

        if avg_val_loss < best_valid_loss:
            best_valid_loss = avg_val_loss
            logger.save_model(model.state_dict(), epoch)
            logger.log('New best loss: {:.3f}'.format(best_valid_loss))

        end_epoch_time = time.time()

        logger.log('Epoch: {} | Time: {}'.format(epoch+1, convert(start_time-end_epoch_time)))
        logger.log(f'\tTrain Loss: {avg_train_loss:.3f} | Train PPL: {math.exp(avg_train_loss):7.3f}')
        logger.log(f'\t Val. Loss: {avg_val_loss:.3f} |  Val. PPL: {math.exp(avg_val_loss):7.3f} | Val. BLEU: {avg_bleu_loss:.3f}')


def train(train_iter, model, criterion, optimizer, device="cuda"):

    # Train model
    model.train()
    losses = AverageMeter()
    for i, batch in enumerate(train_iter):
        # Use GPU
        src = batch.src.to(device)
        trg = batch.trg.to(device)

        # Forward, backprop, optimizer
        model.zero_grad()
        scores = model(src, trg)

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

    return losses.avg



def validate(val_iter, model, criterion, device, TRG, beam_size = 1):
    model.eval()
    losses = AverageMeter()
    sent_candidates = []
    sent_references = []
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
        out = model.predict(src, beam_size=beam_size)  # out is a list

        ## Prepare sentences for BLEU
        ref = list(tgt.data.squeeze())
        # Prepare sentence for bleu script
        remove_tokens = [TRG.vocab.stoi['<pad>'], TRG.vocab.stoi['<s>'], TRG.vocab.stoi['</s>']]
        out = [w for w in out if w not in remove_tokens]
        ref = [w for w in ref if w not in remove_tokens]
        sent_out = ' '.join(TRG.vocab.itos[j] for j in out)
        sent_ref = ' '.join(TRG.vocab.itos[j] for j in ref)
        sent_candidates.append(sent_out)
        sent_references.append(sent_ref)

    bleu = get_moses_multi_bleu(references=sent_references, hypotheses=sent_candidates, lowercase=True)
    print("Bleu Moses Script:", bleu)
#    sacrebl = sacre_corpus_bleu(sys_stream=sent_references, ref_streams=sent_candidates, lowercase=True)
    smooth = SmoothingFunction()
    nlkt_bleu = corpus_bleu(list_of_references=[[sent.split()] for sent in sent_references], hypotheses=[hyp.split() for hyp in sent_candidates], smoothing_function=smooth.method4) *100
    print("NLTK:", nlkt_bleu)
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
