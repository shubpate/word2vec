import os
import pickle
import numpy as np
import scipy.spatial.distance



model_path = './models/'
loss_model = 'cross_entropy'
#loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))

input = ["first","american","would"]
for string in input:
    input_word_vec = embeddings[dictionary[string]]
    output = []
    for word in dictionary:
        word_vec = embeddings[dictionary[word]]
        cosine_dist = scipy.spatial.distance.cosine(input_word_vec, word_vec)
        similarity = 1-cosine_dist
        output.append((similarity, word))

    output.sort(key=lambda tup: tup[0], reverse=True)
    print ("20 words similar to ",string," :")
    for i in range(20):
        print (output[i][1])

