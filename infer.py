import torch
import random
from torch.autograd import Variable
from dataset import SequencePairDataset
from utils import seq_to_string, to_np, trim_seqs
from encoder_decoder import EncoderDecoder
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
        print(output_seqs.size())
        all_output_seqs.extend(trim_seqs(output_seqs))
        all_target_seqs.extend([list(seq[seq > 0])] for seq in to_np(target_variable))

    with open(resultFilename, 'w') as fo:
        for seq, input_tokens in zip(all_output_seqs, input_tokens_list):
            print(type(seq))
            #seq = seq.data.view(-1)
            eos_idx = seq.index(2) if 2 in seq else seq
            string = seq_to_string(seq[:eos_idx+1], idx_to_tok, input_tokens=None)
            fo.write(string + '\n')

    return None


def singleOutput(input_seq, encoder_decoder: EncoderDecoder, input_tokens=None):
    idx_to_tok = encoder_decoder.lang.idx_to_tok
    #if input_tokens is not None:
    #    input_string = ' '.join(input_tokens)
    #else:
    #    input_string = seq_to_string(input_seq, idx_to_tok)
    lengths = ((input_seq != 0).long().sum(dim=0)).unsqueeze(0)
    input_variable = Variable(input_seq).view(1, -1)

    outputs, idxs = encoder_decoder(input_variable, lengths)
    idxs = idxs.data.view(-1)
    eos_idx = list(idxs).index(2) if 2 in list(idxs) else len(idxs)

    string = seq_to_string(idxs[:eos_idx+1], idx_to_tok, input_tokens=input_tokens)

    #print('>', input_string, flush=True)
    #print('<', string, '\n', flush=True)

    return string.strip()



def main(model_path, outputFilename, data_path, use_cuda, batch_size):
    if use_cuda:
        encoder_decoder = torch.load(model_path)
        encoder_decoder = encoder_decoder.cuda()
    else:
        encoder_decoder = torch.load(model_path, map_location=lambda storage, loc: storage)
        encoder_decoder = encoder_decoder.cpu()

    dataset = SequencePairDataset(data_path=data_path, lang=encoder_decoder.lang, use_cuda=use_cuda, is_val=True, val_size=1, shuffle=False)
    #data_loader = DataLoader(dataset, batch_size=batch_size)
    #input_tokens = [item[2].split() for item in dataset]
    #generateResults(encoder_decoder, data_loader, outputFilename, input_tokens)

    #i_seq, _, i_str, _ = random.choice(dataset)
    totalCount = len(dataset)
    outputFile = open(outputFilename, 'w')
    for index, data in enumerate(dataset):
        if index % 100 == 0:
            print(str(index) + '/' + str(totalCount))
        i_seq, _, i_str, _ = data
        i_length = (i_seq > 0).sum()
        i_seq = i_seq[:i_length]
        i_tokens = i_str.split()
        output = singleOutput(i_seq, encoder_decoder, input_tokens=i_tokens)
        outputFile.write(output + '\n')

    outputFile.close()

    return None


if __name__ == '__main__':
    model_path = 'model/copynet_full_20000_3.pt'
    data_path = 'data/commTweets.NNP.tokenized.sampled.original'
    result_path = 'data/commTweets.NNP.tokenized.sampled.original.copynet'
    use_cuda = False
    batch_size = 100
    try:
        main(model_path, result_path, data_path, use_cuda, batch_size)
    except KeyboardInterrupt:
        pass
