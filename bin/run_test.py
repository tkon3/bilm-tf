
import argparse

from bilm.training import test, load_options_latest_checkpoint, load_vocab
from bilm.data import LMDataset, BidirectionalLMDataset

import glob
from collections import Counter

def write_to_file(lexique, output_path):
    
    counter = 0
    prefix = "vocab_test.txt"
    with open(output_path+prefix, "w", encoding='utf-8') as f:
        f.write("%s\n" % "<S>")
        f.write("%s\n" % "</S>")
        f.write("%s\n" % "<UNK>")
        for key, value in lexique.items():
            f.write("%s\n" % key)
            count += value
            
    return output_path+prefix, count

def vocab_builder(test_path, output_path):
    
    file_list = glob.glob(test_path+"/*.txt",recursive=True)
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
    options, ckpt_file = load_options_latest_checkpoint(args.save_dir)

    # load the vocab
    if 'char_cnn' in options:
        max_word_length = options['char_cnn']['max_characters_per_token']
    else:
        max_word_length = None
        
    if args.vocab_file != "" and args.build_vocab == False:
        vocab = load_vocab(args.vocab_file, max_word_length)
    else:
        vocab_file, count = vocab_builder(args.test_prefix, args.save_dir)
        vocab = load_vocab(vocab_file, max_word_length)

    test_prefix = args.test_prefix

    kwargs = {
        'test': True,
        'shuffle_on_load': False,
    }

    if options.get('bidirectional'):
        data = BidirectionalLMDataset(test_prefix, vocab, **kwargs)
    else:
        data = LMDataset(test_prefix, vocab, **kwargs)

    test(options, ckpt_file, data, batch_size=args.batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute test perplexity')
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--vocab_file', help='Vocabulary file', type=string, default="")
    parser.add_argument('--test_prefix', help='Prefix for test files')
    parser.add_argument('--batch_size',
        type=int, default=256,
        help='Batch size')
    parser.add_argument('--build_vocab', help='Build test vocab', type=bool, default=False)

    args = parser.parse_args()
    main(args)

