# bilm-tf
Fork of elmo implementation, see https://github.com/allenai/bilm-tf

# Installation
```
pip install tensorflow-gpu==1.2 h5py
```
Clone and run setup.py

# Preparing

To train and evaluate a biLM, you need to provide:

    a vocabulary file
    a set of training files
    a set of heldout files

The vocabulary file is a a text file with one token per line. It must also include the special tokens <S>, </S> and <UNK> (case sensitive) in the file.

IMPORTANT: the vocabulary file should be sorted in descending order by token count in your training data. The first three lines should be the special tokens (\<S>, \</S> and \<UNK>), then the most common token in the training data, ending with the least common token.

NOTE: the vocabulary file used in training may differ from the one use for prediction.

The training data should be randomly split into many training files, each containing one slice of the data. Each file contains pre-tokenized and white space separated text, one sentence per line. Don't include the \<S> or \</S> tokens in your training data.

All tokenization/normalization is done before training a model, so both the vocabulary file and training files should include normalized tokens. As the default settings use a fully character based token representation, in general we do not recommend any normalization other then tokenization.

Finally, reserve a small amount of the training data as heldout data for evaluating the trained biLM.


# Start training
```
export CUDA_VISIBLE_DEVICES=0,1,2
```
Then run ```bin/train_elmo.py``` with the following arguments :
```
    --train_prefix='/path/to/dataset/**'  # ** allow to get files in directory and subdirectories
    --vocab_file='/path/to/vocab-2016-09-10.txt'  #optional if --build_vocab=True
    --save_dir='/output_path/to/checkpoint' 
    --output_dim=256  #default 256
    --epochs=10  #default 10
    --n_tokens=20000000  #optional if --build_vocab=True 
    --batch_size=128  #default 128
    --build_vocab=False  #default False
```
# Evaluate

Run ```bin/test_elmo.py``` with the following arguments :
```
    --test_prefix='/path/to/test_dataset/**' # ** allow to get files in directory and subdirectories
    --vocab_file='/path/to/vocab-2016-09-10.txt' #optional if --build_vocab=True
    --save_dir='/output_path/to/checkpoint'
    --build_vocab=False  #default False
```

# Dump weights
First, edit options.json file for the newly trained model in the save_dir directory. 

**Important**: always set n_characters to 262 after training (see below).

Then Run:
```
python bin/dump_weights.py \
    --save_dir='/output_path/to/checkpoint'
    --outfile='/output_path/to/weights.hdf5'
```

# Cite :

```
@inproceedings{Peters:2018,
  author={Peters, Matthew E. and  Neumann, Mark and Iyyer, Mohit and Gardner, Matt and Clark, Christopher and Lee, Kenton and Zettlemoyer, Luke},
  title={Deep contextualized word representations},
  booktitle={Proc. of NAACL},
  year={2018}
}
```


#### I'm seeing a WARNING when serializing models, is it a problem?
The below warning can be safely ignored:
```
2018-08-24 13:04:08,779 : WARNING : Error encountered when serializing lstm_output_embeddings.
Type is unsupported, or the types of the items don't match field type in CollectionDef.
'list' object has no attribute 'name'
```
