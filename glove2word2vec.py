from gensim.scripts.glove2word2vec import glove2word2vec

glove_file = './data/glove.42B.300d.txt'
word2vec_file = './model/word2vec.txt'

glove2word2vec(glove_file, word2vec_file)

