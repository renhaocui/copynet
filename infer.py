import random

import torch
from torch.autograd import Variable

from dataset import SequencePairDataset
from utils import seq_to_string, to_np, trim_seqs
from model.encoder_decoder import EncoderDecoder
from torch.utils.data import DataLoader
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def evaluate(encoder_decoder: EncoderDecoder, data_loader):

    loss_function = torch.nn.NLLLoss(ignore_index=0, reduce=False) # what does this return for ignored idxs? same length output?

    losses = []
    all_output_seqs = []
    all_target_seqs = []

    for batch_idx, (input_idxs, target_idxs, _, _) in enumerate(tqdm(data_loader)):
        input_lengths = (input_idxs != 0).long().sum(dim=1)

        sorted_lengths, order = torch.sort(input_lengths, descending=True)

        input_variable = Variable(input_idxs[order, :][:, :max(input_lengths)], volatile=True)
        target_variable = Variable(target_idxs[order, :], volatile=True)
        batch_size = input_variable.shape[0]

        output_log_probs, output_seqs = encoder_decoder(input_variable, list(sorted_lengths))
        all_output_seqs.extend(trim_seqs(output_seqs))
        all_target_seqs.extend([list(seq[seq > 0])] for seq in to_np(target_variable))

        flattened_log_probs = output_log_probs.view(batch_size * encoder_decoder.decoder.max_length, -1)
        batch_losses = loss_function(flattened_log_probs, target_variable.contiguous().view(-1))
        losses.extend(list(to_np(batch_losses)))

    mean_loss = len(losses) / sum(losses)

    bleu_score = corpus_bleu(all_target_seqs, all_output_seqs, smoothing_function=SmoothingFunction().method1)

    return mean_loss, bleu_score


def print_output(input_seq, encoder_decoder: EncoderDecoder, input_tokens=None, target_tokens=None, target_seq=None):
    idx_to_tok = encoder_decoder.lang.idx_to_tok
    if input_tokens is not None:
        input_string = ' '.join(input_tokens)
    else:
        input_string = seq_to_string(input_seq, idx_to_tok)

    lengths = list((input_seq != 0).long().sum(dim=0))
    input_variable = Variable(input_seq).view(1, -1)
    target_variable = Variable(target_seq).view(1, -1)

    if target_tokens is not None:
        target_string = ' '.join(target_tokens)
    elif target_seq is not None:
        target_string = seq_to_string(target_seq, idx_to_tok, input_tokens=input_tokens)
    else:
        target_string = ''

    if target_seq is not None:
        target_eos_idx = list(target_seq).index(2) if 2 in list(target_seq) else len(target_seq)
        target_outputs, _ = encoder_decoder(input_variable, lengths, targets=target_variable, teacher_forcing=1.0)
        target_log_prob = sum([target_outputs[0, step_idx, target_idx] for step_idx, target_idx in enumerate(target_seq[:target_eos_idx+1])])

    outputs, idxs = encoder_decoder(input_variable, lengths)
    idxs = idxs.data.view(-1)
    eos_idx = list(idxs).index(2) if 2 in list(idxs) else len(idxs)
    string = seq_to_string(idxs[:eos_idx+1], idx_to_tok, input_tokens=input_tokens)
    log_prob = sum([outputs[0, step_idx, idx] for step_idx, idx in enumerate(idxs[:eos_idx+1])])

    print('>', input_string, '\n',flush=True)

    if target_seq is not None:
        print('=', target_string, flush=True)
    print('<', string, flush=True)

    print('\n')

    if target_seq is not None:
        print('target log prob:', float(target_log_prob))
    print('output log prob:', float(log_prob))

    print('-' * 100, '\n')

    return idxs


def main(model_path, data_path, use_cuda, n_print, batch_size):
    if use_cuda:
        encoder_decoder = torch.load(model_path)
    else:
        encoder_decoder = torch.load(model_path, map_location=lambda storage, loc: storage)

    if use_cuda:
        encoder_decoder = encoder_decoder.cuda()
    else:
        encoder_decoder = encoder_decoder.cpu()

    dataset = SequencePairDataset(data_path=data_path, lang=encoder_decoder.lang, use_cuda=use_cuda, is_val=True, val_size=1)

    data_loader = DataLoader(dataset, batch_size=batch_size)

    if n_print is not None:
        for _ in range(n_print):
            i_seq, t_seq, i_str, t_str = random.choice(dataset)
            i_length = (i_seq > 0).sum()
            t_length = (i_seq > 0).sum()
            i_seq = i_seq[:i_length]
            t_seq = t_seq[:t_length]
            i_tokens = i_str.split()
            t_tokens = t_str.split()

            print_output(i_seq, encoder_decoder, input_tokens=i_tokens, target_tokens=t_tokens, target_seq=t_seq)
    else:
        evaluate(encoder_decoder, data_loader)


if __name__ == '__main__':
    random = random.Random(42)
    model_path = 'model/copynet_sampled_6.pt'
    data_path = 'data/commTweets.NNP.tokenized.sampled.original.data'
    use_cuda = False
    n_print = 5
    batch_size = 100
    try:
        main(model_path, data_path, use_cuda, n_print, batch_size)
    except KeyboardInterrupt:
        pass
