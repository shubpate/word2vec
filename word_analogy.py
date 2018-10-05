import os
import pickle
import numpy as np
from scipy.spatial import distance


model_path = './models/'
loss_model = 'cross_entropy'
#loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))

"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""

##Readind file word_analogy_dev.txt
with open('word_analogy_dev.txt') as f:
    read_data = f.readlines()
f.closed
given_set = []
unknown_set = []

for line in read_data:
    given = []
    unknown = []
    line = line.replace("\"", "")
    line = line.replace("\n", "")
    ##print (line)
    word = line.split("||")
    ##print (word)
    for i in range(0, len(word)):
        for inword in word[i].split(","):
            final = (inword.split(":"))
            if (i == 0):
                given.append([final[0], final[1]])
            else:
                unknown.append([final[0], final[1]])
    given_set.append(given)
    unknown_set.append(unknown)

##computation  and file writing
w = open("word_analogy_dev_predictions_bymodel.txt", "w")
for i in range(0, len(given_set)):
    diff_set = []
    compare = []
    for j in range(0, 3):
        diff_set.append(
            np.subtract(embeddings[dictionary[given_set[i][j][0]]], embeddings[dictionary[given_set[i][j][1]]]))

    average_vec = np.average(diff_set)
    min_diff = 1
    max_diff = 0
    index_max = -1
    index_min = -1
    for j in range(0, 4):
        diff_unknown = np.subtract(embeddings[dictionary[unknown_set[i][j][0]]],
                                   embeddings[dictionary[unknown_set[i][j][1]]])
        ## checking similarity
        cosine_diff = distance.cosine(average_vec, diff_unknown)
        similarity = 1-cosine_diff
        if (similarity >= max_diff):
            max_diff = similarity
            index_max = j
        elif (similarity < min_diff):
            min_diff = similarity
            index_min = j
        w.write("\"" + unknown_set[i][j][0] + ":" + unknown_set[i][j][1] + "\" ")
    w.write("\"" + unknown_set[i][index_min][0] + ":" + unknown_set[i][index_min][1] + "\" ")
    w.write("\"" + unknown_set[i][index_max][0] + ":" + unknown_set[i][index_max][1] + "\"")
    w.write("\n")
w.closed
