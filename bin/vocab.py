import glob

def build_vocab(filepattern):
    self._all_shards = glob.glob(filepattern+"/*.txt",recursive=True)
