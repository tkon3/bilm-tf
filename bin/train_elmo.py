
import argparse

import numpy as np

from bilm.training import train, load_options_latest_checkpoint, load_vocab
from bilm.data import BidirectionalLMDataset


def main(args):
    
    # load the vocab
    if args.vocab_file != "":
        vocab = load_vocab(args.vocab_file, 50)
    else:
        pass
    
    # number of tokens in training data 
    if args.n_tokens != 0:
        n_train_tokens = args.n_tokens
    

    # define the options
    batch_size = args.batch_size  # batch size for each GPU
    n_gpus = 3

    options = {
     'bidirectional': True,

     'char_cnn': {'activation': 'relu',
      'embedding': {'dim': 16},
      'filters': [[1, 32],
       [2, 32],
       [3, 64],
       [4, 128],
       [5, 256],
       [6, 512],
       [7, 1024]],
      'max_characters_per_token': 50,
      'n_characters': 261,
      'n_highway': 2},
    
     'dropout': 0.1,
    
     'lstm': {
      'cell_clip': 3,
      'dim': 4096,
      'n_layers': 2,
      'proj_clip': 3,
      'projection_dim': args.output_dim/2,
      'use_skip_connections': True},
    
     'all_clip_norm_val': 10.0,
    
     'n_epochs': args.epochs,
     'n_train_tokens': n_train_tokens,
     'batch_size': batch_size,
     'n_tokens_vocab': vocab.size,
     'unroll_steps': 20,
     'n_negative_samples_batch': 8192,
    }

    prefix = args.train_prefix
    data = BidirectionalLMDataset(prefix, vocab, test=False,
                                      shuffle_on_load=True)

    tf_save_dir = args.save_dir
    tf_log_dir = args.save_dir
    train(options, data, n_gpus, tf_save_dir, tf_log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--vocab_file', help='Vocabulary file', type=string, default="")
    parser.add_argument('--train_prefix', help='Prefix for train files')
    parser.add_argument('--output_dim', help='Output dim', type=int, default=256)
    parser.add_argument('--epochs', help='Prefix for train files', type=int, default=10)
    parser.add_argument('--n_tokens', help='Prefix for train files', type=int, default=0)
    parser.add_argument('--batch_size', help='Prefix for train files', type=int, default=128)

    args = parser.parse_args()
    main(args)

