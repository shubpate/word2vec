import tensorflow as tf

def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))

    A =


    And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})


    B =

    ==========================================================================
    """

    ##SHUBHAM##
    # vc = inputs, uo = true_w, thus computing uo_transpose
    uot = tf.transpose(true_w)
    # print("uot shape")
    # print(uot.shape)
    uotvc = tf.matmul(uot, inputs)
    # print("uotvc shape")
    # print(uotvc.shape)
    exp_uotvc = tf.exp(uotvc)
    # print("exp_uotvc shape")
    # print(exp_uotvc.shape)
    sigma_exp_uotvc = tf.reduce_sum(exp_uotvc, 1)
    # print("sigma_exp_uotvc shape")
    # print(sigma_exp_uotvc.shape)
    A = tf.log(exp_uotvc + 0.00000001)   # 0.00000001 is added to prevent occurrence of log 0
    B = tf.log(sigma_exp_uotvc + 0.00000001)
    return tf.subtract(B, A)


def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weigths: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
    labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
    samples: Word_ids for negative samples. Dimension is [num_sampled].
    unigram_prob: Unigram probability. Dimesion is [Vocabulary].

    Implement Noise Contrastive Estimation Loss Here

    ==========================================================================
    """

    ##SHUBHAM##

    # print("Input shape")
    # print (inputs.shape)
    # print("weights shape")
    # print (weights.shape)
    # print("biases shape")
    # print (biases.shape)
    # print("labels shape")
    # print (labels.shape)
    # print("sample shape")
    # print (sample.shape)
    # print("unigram_prob length")
    # print (len(unigram_prob))
    uo = tf.reshape(tf.nn.embedding_lookup(weights, labels), [-1, weights.shape[1]])
    # print(uo.shape)
    bo = tf.nn.embedding_lookup(biases, labels)
    # print("bo shape")
    # print(b.shape)
    ucuo = tf.reduce_sum(tf.multiply(inputs, uo), 1)
    # print ("ucuo shape before")
    # print(ucuo.shape)
    ucuo = tf.reshape(ucuo, [ucuo.shape[0], 1])
    # print ("ucuo shape after")
    # print(ucuo.shape)
    soc = tf.add(ucuo, bo)

    pwo = (tf.nn.embedding_lookup([unigram_prob], labels))
    # print("pwo shape")
    # print(pwo.shape)
    k = len(sample)
    # print(" k= ",k)
    logkpwo = tf.log(tf.scalar_mul(k, pwo) + 0.00000001)
    # print("logkpwo shape after scale and log")
    # print(logkpwo.shape)
    part1 = tf.subtract(soc, logkpwo)
    part1 = tf.sigmoid(part1)
    part1 = tf.log(part1 + 0.00000001)
    # print("part1 shape")
    # print(part1.shape)

    wx = tf.nn.embedding_lookup(weights, sample)
    # print("wx shape before")
    # print(wx.shape)
    wx = tf.reshape(wx, [sample.shape[0], -1])
    # print("wx shape after")  ##
    # print(wx.shape)
    bx = tf.nn.embedding_lookup(biases, sample)
    # print("bx before shape")
    # print(bx.shape)
    bx = tf.reshape(bx, [bx.shape[0], 1])
    # print("bx after shape")
    # print(bx.shape)
    wcwx = tf.matmul(inputs, tf.transpose(wx))
    # #print("wcwx")
    # #print(wcwx.shape)
    sxc = tf.add(wcwx, tf.transpose(bx))
    # print("sxc shape")
    # print(sxc.shape)

    pwx = tf.nn.embedding_lookup([unigram_prob], sample)
    # print("pwx before shape")
    # print(pwx.shape)
    pwx = tf.reshape(pwx, [pwx.shape[0], 1])
    # print("pwx after shape")
    # print(pwx.shape)
    logkpwx = tf.log(tf.scalar_mul(k, pwx) + 0.00000001)
    pwc = tf.subtract(sxc, tf.transpose(logkpwx))
    pwc = tf.nn.sigmoid(pwc)
    part2 = tf.subtract(tf.ones([1, len(sample)]), pwc)
    # print("pr2 shape before")
    # print(pr2.shape)
    part2 = tf.reduce_sum(tf.log(part2 + 0.00000001), 1)
    # print("pr2 shape after")
    # print(pr2.shape)
    final_prob = tf.negative(tf.add(part1, part2))
    # print("final_prob shape before")
    # print(final_prob.shape)
    #final_prob = tf.reshape(final_prob, [128, 1])
    # print("final_prob shape after")
    # print(final_prob.shape)
    return final_prob