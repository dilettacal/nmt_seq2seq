import time

import torch
import numpy as np
import math

from project.utils.constants import UNK_TOKEN, EOS_TOKEN, SOS_TOKEN, PAD_TOKEN
from project.utils.utils import epoch_time, AverageMeter, moses_multi_bleu


def train(train_iter, val_iter, model, criterion, optimizer, scheduler, SRC, TRG, num_epochs, logger=None, device="cuda"):
    # Iterate through epochs
    print("Training on device: {}".format(device))
    bleu_best = -1
    for epoch in range(num_epochs):

        # Validate model with BLEU
        start_time = time.time()  # timer
        bleu_val = validate(val_iter, model, TRG, logger, device)
        if bleu_val > bleu_best:
            bleu_best = bleu_val
            logger.save_model(model.state_dict(), epoch)
            logger.log('New best: {:.3f}'.format(bleu_best))
        val_time = time.time()
        logger.log('Validation time: {:.3f}'.format(val_time - start_time))

        # Validate model with teacher forcing (for PPL)
        val_loss = 0  # validate_losses(val_iter, model, criterion, logger)
        logger.log('PPL: {:.3f}'.format(torch.FloatTensor([val_loss]).exp()[0]))

        # Step learning rate scheduler
        scheduler.step(bleu_val)  # input bleu score

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

            # Log within epoch
            if i % 1000 == 10:
                logger.log('''Epoch [{e}/{num_e}]\t Batch [{b}/{num_b}]\t Loss: {l:.3f}'''.format(e=epoch + 1,
                                                                                                  num_e=num_epochs, b=i,
                                                                                                  num_b=len(train_iter),
                                                                                                  l=losses.avg))

        # Log after each epoch
        logger.log(
            '''Epoch [{e}/{num_e}] complete. Loss: {l:.3f}'''.format(e=epoch + 1, num_e=num_epochs, l=losses.avg))


        logger.log('Training time: {:.3f}'.format(time.time() - val_time))


def validate(data_iter, model, TRG, logger, device, test_set=False, beam_size=1):
    model.eval()

    # Iterate over words in validation batch.
    sents_out = []  # list of sentences from decoder
    sents_ref = []  # list of target sentences
    for i, batch in enumerate(data_iter):
        # Use GPU
        src = batch.src.to(device)
        trg = batch.trg.to(device)
        # Get model prediction (from beam search)
        out = model.predict(src, beam_size=beam_size)  # list of ints (word indices) from greedy search
        ref = list(trg.data.squeeze())
        # Prepare sentence for bleu script
        remove_tokens = [TRG.vocab.stoi[PAD_TOKEN], TRG.vocab.stoi[SOS_TOKEN], TRG.vocab.stoi[EOS_TOKEN]]
        out = [w for w in out if w not in remove_tokens]
        ref = [w for w in ref if w not in remove_tokens]
        sent_out = ' '.join(TRG.vocab.itos[j] for j in out)
        sent_ref = ' '.join(TRG.vocab.itos[j] for j in ref)
        sents_out.append(sent_out)
        sents_ref.append(sent_ref)
    # Run moses bleu script
    bleu = moses_multi_bleu(sents_out, sents_ref)
    # Log information after validation
    if not test_set: logger.log('Validation on validation set complete. BLEU: {bleu:.3f}'.format(bleu=bleu))
    else: logger.log('Validation on test set complete. BLEU: {bleu:.3f}'.format(bleu=bleu))

    return bleu

def translate_test_set(model, data_iter, SRC, TRG, logger, device, beam_size=1):
    logger.log("Translation on test set. Beam size {}".format(beam_size))
    model.eval()

    for i, batch in enumerate(data_iter):
        if i == 50: break
        # Use GPU
        src = batch.src.to(device)
        trg = batch.trg.to(device)
        # Get model prediction (from beam search)
        out = model.predict(src, beam_size=beam_size)  # list of ints (word indices) from greedy search
        ref = list(trg.data.squeeze())
        # Prepare sentence for bleu script
        remove_tokens = [TRG.vocab.stoi[PAD_TOKEN], TRG.vocab.stoi[SOS_TOKEN], TRG.vocab.stoi[EOS_TOKEN]]
        out = [w for w in out if w not in remove_tokens]
        ref = [w for w in ref if w not in remove_tokens]
        src = [w for w in src if w not in remove_tokens]
        sent_out = ' '.join(TRG.vocab.itos[j] for j in out)
        sent_ref = ' '.join(TRG.vocab.itos[j] for j in ref)
        src_sent = ' '.join(SRC.vocab.itos[j] for j in src)

        logger.log("Source > {}".format(src_sent), stdout=False)
        logger.log("Target > {}".format(sent_ref), stdout=False)
        logger.log("Pred   > {}".format(sent_out), stdout=False)
        logger.log("")



# called from main.py: predict.predict(model, args.predict, args.predict_outfile, SRC, TRG, logger)
def predict(model, infile, outfile, SRC, TRG, logger, device="cuda"):
    model.eval()
    with open(infile, 'r') as in_f, open(outfile, 'w') as out_f:
        print('id,word', file=out_f) # for Kaggle
        for i, line in enumerate(in_f):
            input_sent = line.split(' ') # next turn sentence into ints
            input_sent[-1] = input_sent[-1][:-1] # remove '\n' from last word
            sent_indices = [SRC.vocab.stoi[word] if word in SRC.vocab.stoi else SRC.vocab.stoi['<unk>'] for word in input_sent]
            sent = torch.LongTensor([sent_indices])
            sent = sent.to(device)
            sent = sent.view(-1,1) # reshape to sl x bs
            # Predict with beam search
            final_preds = '{},'.format(i+1) # string of the form id,word1|word2|word3 word1|word2|word3 ...
            remove_tokens = [TRG.vocab.stoi[EOS_TOKEN], TRG.vocab.stoi[UNK_TOKEN]] # block predictions of <eos> and <unk>
            preds = model.predict_k(sent, 100, max_len=3, remove_tokens=remove_tokens) # predicts list of 100 lists of size 3
            for pred in preds: # pred is list of size 3
                pred = pred[1:] # remove '<s>' from start of sentence
                pred = [TRG.vocab.itos[index] for index in pred] # convert indices to strings
                pred = [word.replace("\"", "<quote>").replace(",", "<comma>") for word in pred] # for Kaggle
                if len(pred) != 3: print('TOO SHORT: ', pred); continue # should not occur; just in case
                final_preds = final_preds + '{p[0]}|{p[1]}|{p[2]} '.format(p=pred) # for Kaggle
            print(final_preds, file=out_f) # add to output file
            if i % 25 == 0: # log first 100 chars of each 10th prediction
                logger.log('Source: {}\nTarget: {}\n'.format(input_sent, final_preds[0:100]))
        logger.log('Finished predicting')
    return




def predict_from_input(model, input_sentence, SRC, TRG, logger, device="cuda"):
    input_sent = input_sentence.split(' ') # sentence --> list of words
    sent_indices = [SRC.vocab.stoi[word] if word in SRC.vocab.stoi else SRC.vocab.stoi['<unk>'] for word in input_sent]
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
