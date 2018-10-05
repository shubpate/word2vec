# REPORT: Assignment 1

##### - Shubham Patel



## Hyper-parameters

### Descriptions :

Hyper-parameter, is a termed coined for those parameters whose value is set before the learning process begins. This is on contrary to other parameters, in which values are derived during the training process. Tuning hyper-parameters adds another layer of optimality over our model, thus ensured parameters value during training can converge faster, thus resulting in minimum loss.

Hyper-parameters involved in this assignment :

1. **batch_size** - Training process is done in batches, batch_size determines while creating batches from the corpus, what will be the size of each batch. With an increase in batch_size, loss should decrease and accuracy should increase, but there comes a threshold post which, on increasing batch size, loss increases and accuracy decreases (src : http://www.offconvex.org/2016/02/14/word-embeddings-2/).

2. **skip_window** - Within a provided window of elements, skip_window determines how many elements, from both left and right, we pair with the central element ,to form a label,batch [context,predicted word] respectively.  

3. **num_skips** - Within a window, num_skips determine how many time can we use a central element as label [context] and generate batch, label pairs. 

   ```python
   num_skips <= 2*skip_window
   ```

   For lower loss, and higher accuracy, skip_window and num_skips should not be too high, as this will means we are assuming that a center label is related to elements see too previous and too after this center element. However, as studied, many relations can be  explored by going only few steps left and right from center element. 

4. **max_num_steps** - Determines, number of times we will be training our model over the same data, also known as epochs. Similar to batch_size, with increase is max_num_steps loss should decrease and accuracy initially increase, but there comes a threshold post which, on increasing max_num_steps, loss increases and accuracy decreases.

   ------

## Hyper-parameters tuning for reduced loss

|              | batch_size | skip_window | num_skips | max_num_steps | NCE loss   | Cross Entropy loss |
| ------------ | ---------- | ----------- | --------- | ------------- | ---------- | ------------------ |
| **baseline** | **128**    | **4**       | **8**     | **1**         | 794.3354   | 4.9619             |
| default      | **128**    | 4           | 8         | 200000        | 1.4431     | 4.8594             |
| test1        | **64**     | 4           | 8         | 200000        | 2.4109     | 4.8567             |
| test2        | **32**     | 4           | 8         | 200000        | 5.0856     | **4.851**          |
| test3        | 128        | **8**       | **16**    | 200000        | 1.859      | 4.8598             |
| test4        | 128        | **2**       | **4**     | 200000        | **1.2621** | 4.8579             |
| test5        | 128        | **8**       | **8**     | 200000        | 1.3911     | 4.859              |
| test6        | 128        | 4           | 8         | **300000**    | 1.6218     | 4.8571             |
| test7        | 128        | 4           | 8         | **100000**    | 1.4336     | 4.8607             |



|                  |        | batch_size | skip_window | num_skips | max_num_steps |
| ---------------- | ------ | ---------- | ----------- | --------- | ------------- |
| **Min NCE Loss** | 1.2621 | 128        | 2           | 4         | 200000        |
| **Min CE Loss**  | 4.851  | 32         | 4           | 8         | 200000        |

*CE - cross entropy

### Analysis :

##### For NCE loss -

1. **batch_size** : Increasing batch size, helps minimize NCE loss
2. **skip_window**, **num_skips** : decreasing skip_window and num_skips helps minimize NCE loss, for a set skip_window, NCE peaks lower when num_skips = 2*skip_window
3. **max_num_steps** : On increasing epochs, NCE loss decreases reaches a minima, post this, further increasing epochs increases NCE loss

##### For Cross Entropy loss -

1. **batch_size** : Increasing batch size, cross entropy loss increases very gradually. 

2. **skip_window**, **num_skips** : decreasing skip_window and num_skips helps decrease cross entropy loss, however very gradually

3. **max_num_steps** : On increasing epochs, cross entropy loss decreases reaches a minima, post this, further increasing epochs increases cross entropy loss

   #Analysis mentioned here is specific to the data obtained, as per the tables above. 

   ------

## Hyper-parameters tuning for analogy accuracy

| s.no.        | batch_size | skip_window | num_skips | max_num_steps | Accuracy (NCE loss) | Accuracy (Cross Entropy loss) |
| ------------ | ---------- | ----------- | --------- | ------------- | ------------------- | ----------------------------- |
| **baseline** | **128**    | **4**       | **8**     | **1**         | 32.3                | 35.4                          |
| default      | **128**    | **4**       | **8**     | **200000**    | 32.3                | 32.3                          |
| test1        | **64**     | 4           | 8         | 200000        | 33.8                | 28.1                          |
| test2        | **32**     | 4           | 8         | 200000        | 28.4                | 32.5                          |
| test3        | 128        | **8**       | **16**    | 200000        | 29.4                | 31.2                          |
| test4        | 128        | **2**       | **4**     | 200000        | 31.3                | **36.1**                      |
| test5        | 128        | **8**       | **8**     | 200000        | 31.7                | 31.1                          |
| test6        | 128        | 4           | 8         | **300000**    | **34.5**            | 31.1                          |
| test7        | 128        | 4           | 8         | **100000**    | 29                  | 33.9                          |



|                             |      | batch_size | skip_window | num_skips | max_num_steps |
| --------------------------- | ---- | ---------- | ----------- | --------- | ------------- |
| **Max Accuracy (NCE Loss**) | 34.5 | 128        | 4           | 8         | 300000        |
| **Max Accuracy (CE Loss)**  | 36.1 | 128        | 2           | 4         | 200000        |

*CE - cross entropy

### Analysis :

##### Accuracy (NCE loss) -

1. **batch_size** : Increasing batch size, helps maximize accuracy till a maxima. Further increment in batch_size results in decrease in accuracy
2. **skip_window**, **num_skips** : Increasing skip_window and num_skips, helps maximize accuracy till a maxima. Further increment, results in decrease in accuracy
3. **max_num_steps** : High epochs helps in increasing the accuracy

##### Accuracy (Cross Entropy loss) -

1. **batch_size** :  Increasing batch size, helps maximize accuracy till a maxima. Further increment in batch_size results in decrease in accuracy. Also, in between for certain batch sizes, we can get relative dip in accuracy.

2. **skip_window**, **num_skips** : Decreasing skip_window and num_skips, helps maximize accuracy.

3. **max_num_steps** : On increasing epochs, accuracy decreases. 

   #Analysis mentioned here is specific to the data obtained, as per the tables above. 

------



## Similar Words

**Task :** As per the most accurate model, fine 20 similar words, for words : {first, american, would} 

##### For NCE model -

| first       | american | would    |
| ----------- | -------- | -------- |
| first       | american | would    |
| early       | french   | could    |
| term        | civil    | must     |
| th          | so       | will     |
| work        | german   | to       |
| against     | europe   | may      |
| state       | modern   | said     |
| society     | british  | can      |
| what        | all      | if       |
| including   | ancient  | found    |
| economic    | italian  | we       |
| relations   | etzel    | only     |
| before      | composer | believed |
| history     | use      | do       |
| anti        | william  | that     |
| nihilism    | author   | made     |
| does        | scotton  | cannot   |
| association | widely   | did      |
| rique       | he       | you      |
| based       | insult   | india    |

##### For Cross Entropy model -

| first     | american       | would    |
| --------- | -------------- | -------- |
| first     | american       | would    |
| last      | german         | said     |
| name      | british        | might    |
| following | french         | we       |
| most      | english        | could    |
| original  | italian        | must     |
| during    | constellations | will     |
| same      | ca             | been     |
| end       | russian        | does     |
| until     | eadmer         | seems    |
| second    | flory          | did      |
| best      | wiretap        | do       |
| next      | european       | you      |
| largest   | eashi          | should   |
| city      | kirk           | believed |
| main      | lolo           | they     |
| rest      | observation    | may      |
| book      | fart           | who      |
| beginning | vander         | become   |
| th        | war            | argued   |

------



## Summarization : NCE loss method

##### Problem -

Neural probabilistic language model (NPLMs) specify the distribution for the target word *w*, given a sequence of words *h*, called the context. For a given context *h*, an NPLM defines the distribution for the word to be predicted using score function *s(w,h)*. The scores are converted to probabilities by exponentiating and normalizing :
$$
P_{\Theta }^{h}(w)=\frac{\exp (s_{\Theta }(w,h)))}{\sum \exp s_{\Theta }(w{}',h)}
$$
Here, theta, is model parameter, which includes word embedding.

Problem here is, evaluating both *P(w)* and computing likelihood gradient requires normalization over the entire vocabulary, which can be a daunting process if vocabulary is too large. 

##### Resolution : NCE -

With Noise Contrastive Estimation, we reduce the above mentioned density estimation problem to probabilistic binary classification. With this we are able to drop the denominator (normalization factor) in the above equation. While converting to a auxiliary binary classification problem, we treat training data as positive example and samples from noise distribution ad negative examples. We are free to choose any noise distribution that is easy to sample from and compute probabilities under and does not assign zero probability to any word.

Thus, equation mentioned above is reduced to :
$$
P_{\Theta }^{h}(w)=\exp (s_{\Theta }(w,h)))
$$

##### Assumptions -

- Noise distribution - Unigram distribution of training data

- Noise samples are k times more frequent then data samples

  Hence, the probability that the given sample came from data is :
  $$
  P_{}^{h}(D =1|w)=\frac{P_{d}^{h}(w)}{P_{d}^{h}(w)+kP_{n}^{}(w)}
  $$
  Our estimate of this probability is obtained by using our model distribution :
  $$
  P_{}^{h}(D =1|w,\Theta)=\frac{P_{\Theta}^{h}(w)}{P_{\Theta}^{h}(w)+kP_{n}^{}(w)}
  $$


##### Advantage -

We no longer need normalization over the entire vocabulary, as that part is replaced via binary probabilistic classification. So, no matter the size of the corpus, we can compute parameters and train our model independent of it. 
