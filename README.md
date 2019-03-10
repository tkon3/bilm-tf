# bilm-tf
Fork of elmo implementation, see https://github.com/allenai/bilm-tf

```
pip install tensorflow-gpu==1.2 h5py sk-learn
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
