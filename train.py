import argparse
import os
import time
import numpy as np

import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import SequencePairDataset
from model.encoder_decoder import EncoderDecoder
from evaluate import evaluate
from utils import to_np, trim_seqs

from tensorboardX import SummaryWriter
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def train(encoder_decoder: EncoderDecoder, train_data_loader: DataLoader, model_name, model_path,
          val_data_loader: DataLoader, keep_prob, teacher_forcing_schedule, lr, max_length):

    global_step = 0
    loss_function = torch.nn.NLLLoss(ignore_index=0)
    optimizer = optim.Adam(encoder_decoder.parameters(), lr=lr)

    for epoch, teacher_forcing in enumerate(teacher_forcing_schedule):
        print('epoch %i' % epoch, flush=True)

        for batch_idx, (input_idxs, target_idxs, input_tokens, target_tokens) in enumerate(tqdm(train_data_loader)):
            # input_idxs and target_idxs have dim (batch_size x max_len)
            # they are NOT sorted by length

            lengths = (input_idxs != 0).long().sum(dim=1)
            sorted_lengths, order = torch.sort(lengths, descending=True)

            input_variable = Variable(input_idxs[order, :][:, :max(lengths)])
            target_variable = Variable(target_idxs[order, :])

            optimizer.zero_grad()
            output_log_probs, output_seqs = encoder_decoder(input_variable, list(sorted_lengths), targets=target_variable,
                                                            keep_prob=keep_prob, teacher_forcing=teacher_forcing)

            batch_size = input_variable.shape[0]

            flattened_outputs = output_log_probs.view(batch_size * max_length, -1)

            batch_loss = loss_function(flattened_outputs, target_variable.contiguous().view(-1))
            batch_loss.backward()
            optimizer.step()

            batch_outputs = trim_seqs(output_seqs)

            batch_targets = [[list(seq[seq > 0])] for seq in list(to_np(target_variable))]

            batch_bleu_score = corpus_bleu(batch_targets, batch_outputs, smoothing_function=SmoothingFunction().method1)

            if global_step < 10 or (global_step % 10 == 0 and global_step < 100) or (global_step % 100 == 0 and epoch < 2):
                input_string = "Amy, Please schedule a meeting with Marcos on Tuesday April 3rd. Adam Kleczewski"
                output_string = encoder_decoder.get_response(input_string)
                writer.add_text('schedule', output_string, global_step=global_step)

                input_string = "Amy, Please cancel this meeting. Adam Kleczewski"
                output_string = encoder_decoder.get_response(input_string)
                writer.add_text('cancel', output_string, global_step=global_step)

            if global_step % 100 == 0:

                writer.add_scalar('train_batch_loss', batch_loss, global_step)
                writer.add_scalar('train_batch_bleu_score', batch_bleu_score, global_step)

                for tag, value in encoder_decoder.named_parameters():
                    tag = tag.replace('.', '/')
                    writer.add_histogram('weights/' + tag, value, global_step, bins='doane')
                    writer.add_histogram('grads/' + tag, to_np(value.grad), global_step, bins='doane')

            global_step += 1

        val_loss, val_bleu_score = evaluate(encoder_decoder, val_data_loader)

        writer.add_scalar('val_loss', val_loss, global_step=global_step)
        writer.add_scalar('val_bleu_score', val_bleu_score, global_step=global_step)

        encoder_embeddings = encoder_decoder.encoder.embedding.weight.data
        encoder_vocab = encoder_decoder.lang.tok_to_idx.keys()
        writer.add_embedding(encoder_embeddings, metadata=encoder_vocab, global_step=0, tag='encoder_embeddings')

        decoder_embeddings = encoder_decoder.decoder.embedding.weight.data
        decoder_vocab = encoder_decoder.lang.tok_to_idx.keys()
        writer.add_embedding(decoder_embeddings, metadata=decoder_vocab, global_step=0, tag='decoder_embeddings')

        input_string = "Amy, Please schedule a meeting with Marcos on Tuesday April 3rd. Adam Kleczewski"
        output_string = encoder_decoder.get_response(input_string)
        writer.add_text('schedule', output_string, global_step=global_step)

        input_string = "Amy, Please cancel this meeting. Adam Kleczewski"
        output_string = encoder_decoder.get_response(input_string)
        writer.add_text('cancel', output_string, global_step=global_step)

        print('val loss: %.5f, val BLEU score: %.5f' % (val_loss, val_bleu_score), flush=True)
        torch.save(encoder_decoder, "%s%s_%i.pt" % (model_path, model_name, epoch))

        print('-' * 100, flush=True)


def main(data_path, model_path, model_name, use_cuda, batch_size, teacher_forcing_schedule, keep_prob, val_size, lr, decoder_type, vocab_limit, hidden_size, embedding_size, max_length, seed=42):
    #print("training %s with use_cuda=%s, batch_size=%i"% (model_name, use_cuda, batch_size), flush=True)
    #print("teacher_forcing_schedule=", teacher_forcing_schedule, flush=True)
    #print("keep_prob=%f, val_size=%f, lr=%f, decoder_type=%s, vocab_limit=%i, hidden_size=%i, embedding_size=%i, max_length=%i, seed=%i" % (keep_prob, val_size, lr, decoder_type, vocab_limit, hidden_size, embedding_size, max_length, seed), flush=True)

    if os.path.isdir(model_path):
        print("loading encoder and decoder from model_path", flush=True)
        encoder_decoder = torch.load(model_path + model_name + '.pt')

        print("creating training and validation datasets with saved languages", flush=True)
        train_dataset = SequencePairDataset(data_path=data_path, lang=encoder_decoder.lang,
                                            use_cuda=use_cuda, is_val=False, val_size=val_size,
                                            use_extended_vocab=(encoder_decoder.decoder_type == 'copy'))

        val_dataset = SequencePairDataset(data_path=data_path, lang=encoder_decoder.lang,
                                          use_cuda=use_cuda, is_val=True, val_size=val_size,
                                          use_extended_vocab=(encoder_decoder.decoder_type == 'copy'))
    else:
        os.mkdir(model_path)

        print("creating training and validation datasets", flush=True)
        train_dataset = SequencePairDataset(data_path=data_path, vocab_limit=vocab_limit, use_cuda=use_cuda,
                                            is_val=False, val_size=val_size, seed=seed,
                                            use_extended_vocab=(decoder_type == 'copy'))

        val_dataset = SequencePairDataset(data_path=data_path, lang=train_dataset.lang, use_cuda=use_cuda,
                                          is_val=True, val_size=val_size, seed=seed,
                                          use_extended_vocab=(decoder_type == 'copy'))

        print("creating encoder-decoder model", flush=True)
        encoder_decoder = EncoderDecoder(train_dataset.lang, max_length, embedding_size, hidden_size, decoder_type)

        torch.save(encoder_decoder, model_path + '/%s.pt' % model_name)

    if use_cuda:
        encoder_decoder = encoder_decoder.cuda()
    else:
        encoder_decoder = encoder_decoder.cpu()

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size)

    train(encoder_decoder, train_data_loader, model_name, model_path,
          val_data_loader, keep_prob, teacher_forcing_schedule, lr,
          encoder_decoder.decoder.max_length)


if __name__ == '__main__':
    epochs = 5
    model_name = 'copynet'
    model_path = '/content/drive/My Drive/Cui_workspace/CopyNet/model/' + model_name + '/'
    data_path = '/content/drive/My Drive/Cui_workspace/CopyNet/data/pmt.sample.line.data'
    scheduled_teacher_forcing = 'store_true'
    teacher_forcing_fraction = 0.5
    writer = SummaryWriter('./logs/%s_%s' % (model_name, str(int(time.time()))))
    if scheduled_teacher_forcing:
        schedule = np.arange(1.0, 0.0, -1.0/epochs)
    else:
        schedule = np.ones(epochs) * teacher_forcing_fraction

    #main(model_name, use_cuda, batch_size, teacher_forcing_schedule, keep_prob, val_size, lr, decoder_type, vocab_limit, hidden_size, embedding_size, max_length, seed=42)
    main(data_path, model_path, model_name, True, 128, schedule, 1.0, 0.1, 0.001, 'copy', 20000, 256, 256, 100)
