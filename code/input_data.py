""" input data preprocess.
"""

import tensorflow
import numpy as np
import tool
from gensim.models.keyedvectors import KeyedVectors


# use below for korean training (MULTI)
'''
data_prefix = '../data/aihub_data/'
word2vec_path = data_prefix+'ko.vec'
training_data_path = data_prefix + '/existing_khaiii_.txt'
test_data_path = data_prefix + '/emerging_khaiii_.txt'

ex_intent = ['음식점','의류','학원','떡 집','제과','정육','과일 채소 판매','화장품','미용실','약국','숙박']
em_intent = ['음식 배달','의류 색상','방 예약']
'''

# use below for korean training (SINGLE)

data_prefix = '../data/aihub_data/'
word2vec_path = data_prefix+'ko.vec'
training_data_path = data_prefix + '/existing_khaiii_single.txt'
test_data_path = data_prefix + '/emerging_khaiii_single.txt'

ex_intent = ['음식점','의류','학원','떡','제과','정육','농수산물','화장품','미용실','약국','숙박']
em_intent = ['배달','색상','예약']


# english multi-word
'''
# use below for english training
data_prefix = '../data/nlu_data/'
word2vec_path = data_prefix+'wiki.en.vec'
training_data_path = data_prefix + 'train_multi.txt'
test_data_path = data_prefix + 'test_multi.txt'

ex_intent = ['play music', 'search creative work', 'search screening event', 'get weather', 'book restaurant']
em_intent = ['add to playlist', 'rate book']
'''

# english single-word
'''
data_prefix = '../data/nlu_data/'
word2vec_path = data_prefix+'wiki.en.vec'
training_data_path = data_prefix + 'train_shuffle.txt'
test_data_path = data_prefix + 'test.txt'

ex_intent = ['music', 'search', 'movie', 'weather', 'restaurant']
em_intent = ['playlist', 'book']
'''


def load_w2v(file_name):
    """ load w2v model
        input: model file name
        output: w2v model
    """
    w2v = KeyedVectors.load_word2vec_format(
            file_name, binary=False)
    return w2v

def process_label(intents, w2v):
    """ pre process class labels
        input: class label file name, w2v model
        output: class dict and label vectors
    """
    class_dict = {}
    label_vec = []
    class_id = 0
    for line in intents:
        # check whether all the words in w2v dict
        label = line.split(' ') # for multi-word intent
        for w in label:
            if w not in w2v.vocab:
                print ("not in w2v dict", w)

        # compute label vec
        label_sum = np.sum([w2v[w] for w in label], axis = 0)
        label_vec.append(label_sum)
        # store class names => index
        class_dict[' '.join(label)] = class_id
        class_id = class_id + 1
    return class_dict, np.asarray(label_vec)

def load_vec(file_path, w2v, class_dict, in_max_len):
    """ load input data
        input:
            file_path: input data file
            w2v: word2vec model
            max_len: max length of sentence
        output:
            input_x: input sentence word ids
            input_y: input label ids
            s_len: input sentence length
            max_len: max length of sentence
    """
    input_x = [] # input sentence word ids
    input_y = [] # input label ids
    s_len = [] # input sentence length
    max_len = 0

    for line in open(file_path):
        arr = line.strip().split('\t')
        label = [w for w in arr[0].split(' ')]
        question = [w for w in arr[1].split(' ')]
        cname = ' '.join(label)
        if cname not in class_dict:
            continue

        # trans words into indexes
        x_arr = []
        for w in question:
            if w in w2v.vocab:
                x_arr.append(w2v.vocab[w].index)
        s_l = len(x_arr)
        if s_l <= 1:
            continue
        if in_max_len == 0:
            if s_l > max_len:
                max_len = len(x_arr)

        input_x.append(np.asarray(x_arr))
        input_y.append(class_dict[cname])
        s_len.append(s_l)

    # add paddings
    max_len = max(in_max_len, max_len)
    x_padding = []
    for i in range(len(input_x)):
        if (max_len < s_len[i]):
            x_padding.append(input_x[i][0:max_len])
            continue
        tmp = np.append(input_x[i], np.zeros((max_len - s_len[i],), dtype=np.int64))
        x_padding.append(tmp)

    x_padding = np.asarray(x_padding)
    input_y = np.asarray(input_y)
    s_len = np.asarray(s_len)
    return x_padding, input_y, s_len, max_len

def get_label(data):
    ex_label = data['y_ex']
    sample_num = ex_label.shape[0]
    labels = np.unique(ex_label)
    class_num = labels.shape[0]
    labels = range(class_num)
    # get label index
    lb_idx = np.zeros((sample_num, class_num), dtype=np.float32)
    for i in range(class_num):
        lb_idx[ex_label == labels[i], i] = 1
    return lb_idx

def load_vec_label(intent, w2v, max_len):
    input_x = [] # input sentence word ids
    s_len = [] # input sentence length
    for item in intent:
        question = [w for w in item.split(' ')]
        
        # trans words into indexes
        x_arr = []
        for w in question:
            if w in w2v.vocab:
                x_arr.append(w2v.vocab[w].index)
        s_l = len(x_arr)

        input_x.append(np.asarray(x_arr))
        s_len.append(s_l)

    # add paddings
    x_padding = []
    for i in range(len(input_x)):
        if (max_len < s_len[i]):
            x_padding.append(input_x[i][0:max_len])
            continue
        tmp = np.append(input_x[i], np.zeros((max_len - s_len[i],), dtype=np.int64))
        x_padding.append(tmp)

    x_padding = np.asarray(x_padding)
    s_len = np.asarray(s_len)
    return x_padding, s_len


def read_datasets():
    print ("[Start Dataset Reading.]")
    data = {}
    # load word2vec model
    w2v = load_w2v(word2vec_path)
    print ("[Load Word2Vec model.]")

    # load normalized word embeddings
    norm_embedding = tool.norm_matrix(w2v.syn0)
    data['embedding'] = norm_embedding
    print ("[Load normalized word embedding.]")

    # preprocess seen and unseen labels
    ex_dict, ex_vec = process_label(ex_intent, w2v)
    em_dict, em_vec = process_label(em_intent, w2v)
    print ("[Preprocess labels.]")

    # trans data into embedding vectors
    max_len = 0
    x_ex, y_ex, ex_len, max_len = load_vec(
            training_data_path, w2v, ex_dict, max_len)
    x_em, y_em, em_len, max_len = load_vec(
            test_data_path, w2v, em_dict, max_len)

    label_ex, label_ex_len = load_vec_label(ex_intent, w2v, max_len)
    label_em, label_em_len = load_vec_label(em_intent, w2v, max_len) 
    # existing intent

    #data['ex_intent']=ex_intent
    #data['em_intent']=em_intent

    data['label_ex']=label_ex
    data['label_ex_len']=label_ex_len

    data['label_em']=label_em
    data['label_em_len']=label_em_len

    data['x_ex'] = x_ex
    data['y_ex'] = y_ex

    data['ex_len'] = ex_len
    data['ex_vec'] = ex_vec
    data['ex_dict'] = ex_dict

    # emerging intent
    data['x_em'] = x_em
    data['y_em'] = y_em

    data['em_len'] = em_len
    data['em_vec'] = em_vec
    data['em_dict'] = em_dict

    data['max_len'] = max_len
    data['ex_label'] = get_label(data) 
    # [0.0, 0.0, ..., 1.0, ..., 0.0]
    
    print ("[Complete Dataset Reading.]")
    return data
