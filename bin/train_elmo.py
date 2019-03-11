
import argparse
import numpy as np
from bilm.training import train, load_options_latest_checkpoint, load_vocab
from bilm.data import BidirectionalLMDataset
import glob
from collections import Counter

def write_to_file(lexique, output_path):
    
    counter = 0
    with open(output_path+"vocab.txt", "w", encoding='utf-8') as f:
        f.write("%s\n" % "<S>")
        f.write("%s\n" % "</S>")
        f.write("%s\n" % "<UNK>")
        for key, value in lexique.items():
            f.write("%s\n" % key)
            count += value
            
    return output_path+"vocab.txt", count

def vocab_builder(train_path, output_path):
    
    file_list = glob.glob(train_path+"/*.txt",recursive=True)
    word_counter = Counter()
    base_list = []
    
    for i,file in enumerate(file_list):
        if i % 20000 == 0:
            word_counter.update(base_list)
            base_list = []
            
        with open(file, encoding='utf-8') as f:
            new_list = f.read().split()
            
        base_list = base_list + new_list  
        
    word_counter.update(base_list)
    lexique = dict(sorted(word_counter.items(), key=operator.itemgetter(1), reverse=True))
    
    return write_to_file(lexique, output_path)


def main(args):
    
    prefix = args.train_prefix
    tf_save_dir = args.save_dir
    tf_log_dir = args.save_dir
    
    # load the vocab
    if args.vocab_file != "" and args.build_vocab == False:
        vocab = load_vocab(args.vocab_file, 50)
   
    # number of tokens in training data 
    if args.n_tokens != 0 and args.build_vocab == False:
        n_train_tokens = args.n_tokens
        
    if args.build_vocab == True:
        print("Building vocabulary...")
        vocab_file, n_train_tokens = vocab_builder(prefix, tf_save_dir)
        vocab = load_vocab(vocab_file, 50)
    

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

    
    data = BidirectionalLMDataset(prefix, vocab, test=False,
                                      shuffle_on_load=True)
    
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
    parser.add_argument('--build_vocab', help='Prefix for train files', type=bool, default=False)

    args = parser.parse_args()
    main(args)

