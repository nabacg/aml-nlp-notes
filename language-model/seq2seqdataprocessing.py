import datasets
import os
import re
from collections import Counter
import tensorflow as tf
from sklearn.model_selection import train_test_split


# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
#     w = unicode_to_ascii(w.lower().strip())
    w = w.lower().strip()
    
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ." 
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = re.sub(r"[^a-zA-Z]+", " ", w)
    w = w.rstrip().strip()
    
    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w

# This class creates a word -> index mapping (e.g,. "dad" -> 5) and vice-versa 
# (e.g., 5 -> "dad") for each language,


UNK_WORD_INDEX = 1
def create_index(phrases, max_vocab_size): 
    word2idx = {}
#     vocab_size = 15000 
    idx2word = {}
    # vocab = set()
    # https://docs.python.org/3/library/collections.html#collections.Counter
    wordcount = Counter([p for s in phrases for p in s.split(' ')])
    wordcount = sorted([(k,v) for (k,v) in wordcount.items() if v > 1], key=lambda kv: kv[1], reverse=True)[:max_vocab_size]
    vocab = sorted([w for (w, c) in wordcount]) # todo vocabsize should be -2 for <pad> and <unk>
    
    word2idx['<pad>'] = 0
    word2idx['<unk>'] = UNK_WORD_INDEX
    for index, word in enumerate(vocab):
        word2idx[word] = index + 2
    
    for word, index in word2idx.items():
        idx2word[index] = word
        
    return word2idx, idx2word, vocab
  
def word_to_idx(lookup, word):
  return lookup.get(word, UNK_WORD_INDEX)

def idx_to_word(lookup, idx): 
  return lookup.get(idx, '<unk>')


def grouped(iterable, n):
    "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
    return zip(*[iter(iterable)]*n)

def readTwitterData():
  data_file =  open('data/chat_corpus/twitter_en.txt')
  return [(preprocess_sentence(q), preprocess_sentence(a)) for (q,a) in grouped(data_file, 2)]
 
def read_dataset(dataset_name, max_sentence_length):
    dataset_path = 'data/{}'.format(dataset_name)

    if dataset_name == "cornell":
        data = datasets.readCornellData(dataset_path, max_len=max_sentence_length)
    elif dataset_name == "opensubs":
        data = datasets.readOpensubsData(dataset_path, max_len=max_sentence_length)
    elif dataset_name == 'twitter':
        data = readTwitterData()
    else:
        raise ValueError("Unrecognized dataset: {!r}".format(dataset_name))
    
    return data

def max_length(tensor):
    return max(len(t) for t in tensor)

def load_dataset(dataset_name = 'cornell', max_sentence_length= 10, vocab_size=15000): # why 15k ? because that's what they used in https://arxiv.org/pdf/1406.1078.pdf
    # creating cleaned input, output pairs
    pairs = [(preprocess_sentence(a), preprocess_sentence(b)) for (a,b) in read_dataset(dataset_name, max_sentence_length)]

    # index language using the class defined above    
    word2idx, idx2word, vocab = create_index([p for ps in pairs for p in ps], max_vocab_size=vocab_size)
    # Vectorize the input and target languages
    
    # question sentences
    input_tensor = [[word_to_idx(word2idx, w) for w in qs.split(' ')] for qs, a in pairs]
    
    # answer sentences
    target_tensor = [[word_to_idx(word2idx, w) for w in a.split(' ')] for q, a in pairs]
    
    # Calculate max_length of input and output tensor
    # Here, we'll set those to the longest sentence in the dataset
    # max_length_inp, max_length_tar = max_length(input_tensor), max_length(target_tensor)

    
    # Padding the input and output tensor to the maximum length
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, 
                                                                 maxlen=max_sentence_length,
                                                                 padding='post')
    
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor, 
                                                                  maxlen=max_sentence_length, 
                                                                  padding='post')
    
    return input_tensor, target_tensor, (word2idx, idx2word, vocab)


def create_dataset(batch_size, dataset_name = 'cornell', max_sentence_length= 10, vocab_size=15000):

  input_tensor, target_tensor, dict_index = load_dataset(dataset_name = dataset_name, 
                                                       max_sentence_length=max_sentence_length, 
                                                       vocab_size=vocab_size) 

  word2idx = dict_index[0]
  idx2word = dict_index[1]

  # Creating training and validation sets using an 80-20 split
  input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.1)

  buffer_size = len(input_tensor_train)
  train_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(buffer_size)
  train_dataset = train_dataset.batch(batch_size, drop_remainder=True)

  validation_ds = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val)).shuffle(len(input_tensor_val))
  validation_ds = validation_ds.batch(batch_size, drop_remainder=True)

  return train_dataset, validation_ds, word2idx, idx2word