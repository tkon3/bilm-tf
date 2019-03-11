# bilm-tf
Fork of elmo implementation, see https://github.com/allenai/bilm-tf

# Installation
```
pip install tensorflow-gpu==1.2 h5py
```
Clone and run setup.py

# Running embedding
```
export CUDA_VISIBLE_DEVICES=0,1,2
python bin/train_elmo.py \
    --train_prefix='/path/to/dataset/**' \ # ** allow to get files in directory and subdirectories
    --vocab_file='/path/to/vocab-2016-09-10.txt' \ #optional if --build_vocab=True
    --save_dir='/output_path/to/checkpoint' \
    --output_dim=256 \ #default 256
    --epochs=10 \ #default 10
    --n_tokens=20000000 \ #optional if --build_vocab=True 
    --batch_size=128 \ #default 128
    --build_vocab=False \ #default False
    
```

Citation:

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
