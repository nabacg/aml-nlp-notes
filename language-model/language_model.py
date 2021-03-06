from __future__ import absolute_import, division, print_function

# Import TensorFlow >= 1.10 and enable eager execution
import tensorflow as tf

tf.enable_eager_execution()

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from seq2seqdataprocessing import word_to_idx, preprocess_sentence
import unicodedata
import re
import numpy as np
import os
import time
from collections import Counter

print(tf.__version__)


def gru(units):
  # If you have a GPU, we recommend using CuDNNGRU(provides a 3x speedup than GRU)
  # the code automatically does that.
    # if tf.test.is_gpu_available():

    # since this has caused issues with restoring model weights trained on Colab
    # i.e. Checkpoint files are not compatible between CuDNNGRU and GRU I'm hardcoding it to CuDNNGRU
    # more details on this https://github.com/tensorflow/tensorflow/issues/25081
    # if True:
    #     return tf.keras.layers.CuDNNGRU(units, 
    #                                 return_sequences=True, 
    #                                 return_state=True, 
    #                                 recurrent_initializer='glorot_uniform')
    # else:
    return tf.keras.layers.GRU(units, 
                               return_sequences=True, 
                               return_state=True, 
                               recurrent_activation='sigmoid', 
                               recurrent_initializer='glorot_uniform')

def load_embeddings(embeddings_path, word2idx):
    """Loads pre-trained word embeddings from tsv file.

    Args:
      embeddings_path - path to the embeddings file.

    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.

    
    """

    starspace_embeddings = {}
    for line in open(embeddings_path, 'r'):
        word, *embs = line.strip().split('\t')
        starspace_embeddings[word] = np.array(list(map(float, embs)),  dtype=np.float32)

    embedding_dim = starspace_embeddings[next(iter(starspace_embeddings))].shape[0]

    # 2.prepare embedding matrix - inspired by  https://stackoverflow.com/a/56820340
    print('Filling pre-trained embeddings...')
    num_words = len(word2idx)
    # initialization by zeros
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word2idx.items():
        embedding_vector = starspace_embeddings.get(word)
        if embedding_vector is not None:
                  # words not found in embedding index will be all zeros.
            embedding_matrix[i] = embedding_vector

    return starspace_embeddings, embedding_dim, embedding_matrix

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, embedding ): # todo figure out how to load default of = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = embedding

        self.gru = gru(self.enc_units)
        
    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)        
        return output, state
    
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, embedding):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = embedding

        self.gru = gru(self.dec_units)
        self.fc = tf.keras.layers.Dense(vocab_size)
        
        # used for attention
        self.W1 = tf.keras.layers.Dense(self.dec_units)
        self.W2 = tf.keras.layers.Dense(self.dec_units)
        self.V = tf.keras.layers.Dense(1)
        
    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        
        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying tanh(FC(EO) + FC(H)) to self.V
        score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis)))
        
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        
        # passing the concatenated vector to the GRU
        output, state = self.gru(x)
        
        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))
        
        # output shape == (batch_size * 1, vocab)
        x = self.fc(output)
        
        return x, state, attention_weights
        
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.dec_units))


def loss_function(real, pred):
    mask = 1 - np.equal(real, 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss_)


def create_checkpoints(optimizer, encoder, decoder, root_dir):
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

    manager = tf.train.CheckpointManager(checkpoint, 
                                    root_dir, 
                                    max_to_keep=3)

    return checkpoint, manager


def create_models(vocab_size, word2idx, units, batch_size, pretrained_embeddings_file, embedding_dim=256):
    if pretrained_embeddings_file:
        _, embedding_dim, emb_matrix  = load_embeddings(pretrained_embeddings_file, word2idx)

        embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, 
                                           embeddings_initializer = tf.initializers.constant(emb_matrix),
                                           trainable=True)
    else:
        embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    encoder = Encoder(vocab_size, embedding_dim, units, batch_size, embedding)
    decoder = Decoder(vocab_size, embedding_dim, units, batch_size, embedding)
    optimizer = tf.train.AdamOptimizer()

    return encoder, decoder, optimizer

def train_model(encoder, decoder, optimizer, dataset, 
                batch_size, n_batch,  start_word_index, epochs, save_checkpoint):
    for epoch in range(epochs):
        start = time.time()
        hidden = encoder.initialize_hidden_state()
        total_loss = 0
        
        for (batch, (inp, targ)) in enumerate(dataset):
            loss = 0
            with tf.GradientTape() as tape:
                enc_output, enc_hidden = encoder(inp, hidden)
                dec_hidden = enc_hidden            
                dec_input = tf.expand_dims([start_word_index] * batch_size, 1)       
                
                # Teacher forcing - feeding the target as the next input
                for t in range(1, targ.shape[1]):
                    # passing enc_output to the decoder
                    predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
                    loss += loss_function(targ[:, t], predictions)
                    # using teacher forcing
                    dec_input = tf.expand_dims(targ[:, t], 1)
            
            batch_loss = (loss / int(targ.shape[1]))
            total_loss += batch_loss
            variables = encoder.variables + decoder.variables
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))
            
            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                            batch,
                                                            batch_loss.numpy()))
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            save_checkpoint()
        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / n_batch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')
    
    fontdict = {'fontsize': 14}
    
    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    plt.show()


def evaluate(sentence, encoder, decoder, word2idx, idx2word, units, max_length):

    attention_plot = np.zeros((max_length, max_length))
    
    sentence = preprocess_sentence(sentence)

    inputs = [word_to_idx(word2idx, i) for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    
    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([word_to_idx(word2idx, '<start>')], 0)

    for t in range(max_length):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
        
        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += idx2word[predicted_id] + ' '

        if idx2word[predicted_id] == '<end>':
            return result, sentence, attention_plot
        
        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot



def gen_answer(sentence, encoder, decoder, word2idx, idx2word, units, max_length, print_debug=True):
    result, sentence, attention_plot = evaluate(sentence, encoder, decoder, word2idx, idx2word, units, max_length)
    
    if print_debug:
      print('Input: {}'.format(sentence))
      print('Answer: {}'.format(result))
      attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
      plot_attention(attention_plot, sentence.split(' '), result.split(' '))
    return result

def create_bot(encoder, decoder, word2idx, idx2word, units, max_length):
    return lambda q: gen_answer(q, encoder, decoder, word2idx, idx2word, units, max_length, print_debug=False)