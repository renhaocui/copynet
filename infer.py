import random
import torch
from torch.autograd import Variable
from dataset import SequencePairDataset
from utils import seq_to_string, to_np, trim_seqs
from model.encoder_decoder import EncoderDecoder
from torch.utils.data import DataLoader
from tqdm import tqdm

@torch.no_grad()
def generateResults(encoder_decoder: EncoderDecoder, data_loader, resultFilename, input_tokens_list):
    idx_to_tok = encoder_decoder.lang.idx_to_tok
    all_output_seqs = []
    all_target_seqs = []

    for batch_idx, (input_idxs, target_idxs, _, _) in enumerate(tqdm(data_loader)):
        input_lengths = (input_idxs != 0).long().sum(dim=1)

        sorted_lengths, order = torch.sort(input_lengths, descending=True)
        input_variable = Variable(input_idxs[order, :][:, :max(input_lengths)])
        target_variable = Variable(target_idxs[order, :])

        output_log_probs, output_seqs = encoder_decoder(input_variable, list(sorted_lengths))
        all_output_seqs.extend(trim_seqs(output_seqs))
        all_target_seqs.extend([list(seq[seq > 0])] for seq in to_np(target_variable))

    with open(resultFilename, 'w') as fo:
        for seq, input_tokens in zip(all_output_seqs, input_tokens_list):
            string = seq_to_string(seq, idx_to_tok, input_tokens=input_tokens)
            fo.write(string + '\n')

    return None



def print_output(input_seq, encoder_decoder: EncoderDecoder, input_tokens=None, target_tokens=None, target_seq=None):
    idx_to_tok = encoder_decoder.lang.idx_to_tok
    if input_tokens is not None:
        input_string = ' '.join(input_tokens)
    else:
        input_string = seq_to_string(input_seq, idx_to_tok)
    print((input_seq != 0).long().sum(dim=0))
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


def main(model_path, outputFilename, data_path, use_cuda, n_print, batch_size):
    if use_cuda:
        encoder_decoder = torch.load(model_path)
        encoder_decoder = encoder_decoder.cuda()
    else:
        encoder_decoder = torch.load(model_path, map_location=lambda storage, loc: storage)
        encoder_decoder = encoder_decoder.cpu()

    dataset = SequencePairDataset(data_path=data_path, lang=encoder_decoder.lang, use_cuda=use_cuda, is_val=True, val_size=1)
    data_loader = DataLoader(dataset, batch_size=batch_size)

    input_tokens = [item[2].split() for item in dataset]
    generateResults(encoder_decoder, data_loader, outputFilename, input_tokens)

    return None


if __name__ == '__main__':
    random = random.Random(42)
    model_path = 'model/copynet_full_20000_0.pt'
    data_path = 'data/commTweets.NNP.tokenized.sampled.original.data'
    result_path = 'data/commTweets.NNP.tokenized.sampled.original.copynet'
    use_cuda = False
    n_print = None
    batch_size = 100
    try:
        main(model_path, result_path, data_path, use_cuda, n_print, batch_size)
    except KeyboardInterrupt:
        pass
