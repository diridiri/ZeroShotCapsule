import os
from random import *
import input_data
import numpy as np
import tensorflow as tf
import model
import tool
import math
import time
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt

a = Random()
a.seed(1)

def setting(data):
    vocab_size, word_emb_size = data['embedding'].shape
    sample_num, max_time = data['x_ex'].shape
    test_num = data['x_em'].shape[0]
    s_cnum = np.unique(data['y_ex']).shape[0]
    u_cnum = np.unique(data['y_em']).shape[0]

    FLAGS = tf.compat.v1.flags.FLAGS
    tf.compat.v1.flags.DEFINE_float("keep_prob", 0.8, "embedding dropout keep rate")
    tf.compat.v1.flags.DEFINE_integer("hidden_size", 32, "embedding vector size")
    tf.compat.v1.flags.DEFINE_integer("batch_size", 64, "vocab size of word vectors")
    tf.compat.v1.flags.DEFINE_integer("num_epochs", 100, "num of epochs")
    tf.compat.v1.flags.DEFINE_integer("vocab_size", vocab_size, "vocab size of word vectors")
    tf.compat.v1.flags.DEFINE_integer("max_time", max_time, "max number of words in one sentesnce")
    tf.compat.v1.flags.DEFINE_integer("sample_num", sample_num, "sample number of training data")
    tf.compat.v1.flags.DEFINE_integer("test_num", test_num, "number of test data")
    tf.compat.v1.flags.DEFINE_integer("s_cnum", s_cnum, "seen class num")
    tf.compat.v1.flags.DEFINE_integer("u_cnum", u_cnum, "unseen class num")
    tf.compat.v1.flags.DEFINE_integer("word_emb_size", word_emb_size, "embedding size of word vectors")
    tf.compat.v1.flags.DEFINE_string("ckpt_dir", './saved_models/' , "check point dir")
    tf.compat.v1.flags.DEFINE_boolean("use_embedding", True, "whether to use embedding or not.")
    tf.compat.v1.flags.DEFINE_float("learning_rate", 0.0001, "learning rate")
    tf.compat.v1.flags.DEFINE_float("sim_scale", 4, "sim scale")
    tf.compat.v1.flags.DEFINE_float("margin", 1.0, "ranking loss margin")
    tf.compat.v1.flags.DEFINE_float("alpha", 0.0001, "coefficient for self attention loss")
    tf.compat.v1.flags.DEFINE_integer("num_routing", 2, "capsule routing num")
    tf.compat.v1.flags.DEFINE_integer("output_atoms", 10, "capsule output atoms")
    tf.compat.v1.flags.DEFINE_boolean("save_model", False, "save model to disk")
    tf.compat.v1.flags.DEFINE_integer("d_a", 20, "self attention weight hidden units number")
    tf.compat.v1.flags.DEFINE_integer("r", 3, "self attention weight hops")
    return FLAGS

def get_sim(data):
    # get unseen and seen categories similarity
    ex = normalize(data['ex_vec'])
    em = normalize(data['em_vec'])
    sim = tool.compute_label_sim(em, ex, FLAGS.sim_scale)
    return sim

def evaluate_zsl(data, FLAGS, sess):
    # zero-shot testing state
    # seen votes shape (110, 2, 34, 10)
    x_em = data['x_em']
    y_em_id = data['y_em']
    em_len = data['em_len']
    #em_intent = data['em_intent']

    # get unseen and seen categories similarity
    # sim shape (8, 34)
    sim_ori = get_sim(data)
    
    # origin_sim, new_sim
    # similarity = 0.5(1+1/cak)origin_sim+0.5(1-1/k)new_sim

    #sim_ori = data['em_logits']

    total_unseen_pred = np.array([], dtype=np.int64)

    batch_size  = FLAGS.test_num
    test_batch = int(math.ceil(FLAGS.test_num / float(batch_size)))
    #test_batch = int(math.ceil(FLAGS.test_num / float(FLAGS.batch_size)))
    for i in range(test_batch):
        begin_index = i * batch_size
        end_index = min((i + 1) * batch_size, FLAGS.test_num)
        batch_te = x_em[begin_index : end_index]
        batch_id = y_em_id[begin_index : end_index]
        batch_len = em_len[begin_index : end_index]

        [attentions, seen_logits, seen_votes, seen_weights_c] = sess.run([
            lstm.attention, lstm.logits, lstm.votes, lstm.weights_c],
            feed_dict={lstm.input_x: batch_te, lstm.s_len: batch_len})

        sim = tf.expand_dims(sim_ori, [0])
        sim = tf.tile(sim, [seen_votes.shape[1],1,1])
        sim = tf.expand_dims(sim, [0])
        sim = tf.tile(sim, [seen_votes.shape[0],1,1,1])
        seen_weights_c = np.tile(np.expand_dims(seen_weights_c, -1), [1, 1, 1, FLAGS.output_atoms])
        mul = np.multiply(seen_votes, seen_weights_c)

        # compute unseen features 
        # unseen votes shape (110, 2, 8, 10)
        unseen_votes = tf.matmul(sim, mul)

        # routing unseen classes
        u_activations, u_weights_c = update_unseen_routing(unseen_votes, FLAGS, 3)
        unseen_logits = tf.norm(u_activations, axis=-1)
        te_votes, te_logits, te_weights, te_activations = sess.run([
            unseen_votes, unseen_logits, u_weights_c, u_activations])

        te_batch_pred = np.argmax(te_logits, 1)
        total_unseen_pred = np.concatenate((total_unseen_pred, te_batch_pred))

    print ("Zero-shot Intent Detection Results [INTENTCAPSNET-ZSL]")
    acc = accuracy_score(y_em_id, total_unseen_pred)
    print (classification_report(y_em_id, total_unseen_pred, digits=4))
    return acc

def generate_batch(n, batch_size):
    batch_index = a.sample(range(n), batch_size)
    return batch_index

def assign_pretrained_word_embedding(sess, data, textRNN):
    print("using pre-trained word emebedding.begin...")
    embedding = data['embedding']

    word_embedding = tf.constant(embedding, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.compat.v1.assign(textRNN.Embedding,word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding)
    print("using pre-trained word emebedding.ended...")

def squash(input_tensor):
    norm = tf.norm(input_tensor, axis=2, keepdims=True)
    norm_squared = norm * norm
    return (input_tensor / norm) * (norm_squared / (1 + norm_squared))

def update_unseen_routing(votes, FLAGS, num_routing=3):
    votes_t_shape = [3, 0, 1, 2]
    r_t_shape = [1, 2, 3, 0]
    votes_trans = tf.transpose(votes, votes_t_shape)
    num_dims = 4
    input_dim = FLAGS.r
    output_dim = FLAGS.u_cnum
    input_shape = tf.shape(votes)
    logit_shape = tf.stack([input_shape[0], input_dim, output_dim])

    def _body(i, logits, activations, route):
        route = tf.nn.softmax(logits)
        preactivate_unrolled = route * votes_trans
        preact_trans = tf.transpose(preactivate_unrolled, r_t_shape)
        preactivate = tf.reduce_sum(preact_trans, axis=1)
        activation = squash(preactivate)
        activations = activations.write(i, activation)

        act_3d = tf.expand_dims(activation, 1)
        tile_shape = np.ones(num_dims, dtype=np.int32).tolist()
        tile_shape[1] = input_dim
        act_replicated = tf.tile(act_3d, tile_shape)
        distances = tf.reduce_sum(votes * act_replicated, axis=3)
        logits += distances
        return (i + 1, logits, activations, route)

    activations = tf.TensorArray(
        dtype=tf.float32, size=num_routing, clear_after_read=False)
    logits = tf.fill(logit_shape, 0.0)
    i = tf.constant(0, dtype=tf.int32)
    route = tf.compat.v1.math.softmax(logits, dim=2)
    _, logits, activations, route = tf.while_loop(
        lambda i, logits, activations, route: i < num_routing,
        _body,
        loop_vars=[i, logits, activations, route],
        swap_memory=True)

    return activations.read(num_routing - 1), route

if __name__ == "__main__":
    # load data
    
    data = input_data.read_datasets()

    embedding = data['embedding']

    x_ex = data['x_ex']
    y_ex_id = data['y_ex']
    ex_len = data['ex_len']
    y_idx = data['ex_label']

    x_em = data['x_em']
    y_em_id = data['y_em']
    em_len = data['em_len']

    label_em=data['label_em']
    label_em_len=data['label_em_len']

    # load settings
    FLAGS = setting(data)

    caps_train_loss=[]
    caps_train_acc=[]
    
    caps_val_loss=[]
    caps_val_acc=[]
    
    zsl_acc=[]

    # start
    tf.compat.v1.reset_default_graph()
    config=tf.compat.v1.ConfigProto()
    with tf.compat.v1.Session(config=config) as sess:

        # Instantiate Model
        lstm = model.lstm_model(FLAGS)

        if os.path.exists(FLAGS.ckpt_dir):
            print("Restoring Variables from Checkpoint for rnn model.")
            saver = tf.compat.v1.train.Saver()
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print('Initializing Variables')
            sess.run(tf.compat.v1.global_variables_initializer())
            if FLAGS.use_embedding: #load pre-trained word embedding
                assign_pretrained_word_embedding(sess, data, lstm)

        xex_train, xex_test, yex_train, yex_test, ex_len_train, ex_len_test, y_idx_train, y_idx_test = train_test_split(x_ex, y_ex_id, ex_len, y_idx, test_size=0.33, shuffle=False)
        
        best_caps_acc = 0
        best_zsl_acc = 0
        var_saver = tf.compat.v1.train.Saver()
        
        # Training cycle
        for epoch in range(FLAGS.num_epochs):
            # training
            epoch_start_time=time.time()
            total_batch=math.ceil(xex_train.shape[0]/FLAGS.batch_size)
            
            for batch in tqdm(range(total_batch)):
                begin=batch*FLAGS.batch_size
                end=min(xex_train.shape[0], (batch+1)*FLAGS.batch_size)
                batch_x = xex_train[begin:end]
                batch_y = yex_train[begin:end]
                batch_len = ex_len_train[begin:end]
                batch_ind = y_idx_train[begin:end]

                [_, train_loss, train_logits] = sess.run([lstm.train_op, lstm.loss_val, lstm.logits],
                        feed_dict={lstm.input_x: batch_x, lstm.IND: batch_ind, lstm.s_len: batch_len})

            caps_train_loss.append(str(train_loss))
            # training acc.
            total_seen_pred_train = np.array([], dtype=np.int64)
            
            test_batch_pred = np.argmax(train_logits, 1)
            total_seen_pred = np.concatenate((total_seen_pred_train, test_batch_pred))
            train_acc = accuracy_score(batch_y, total_seen_pred)
            caps_train_acc.append(str(train_acc))

            # validation
            
            [val_loss, logits] = sess.run([lstm.loss_val, lstm.logits], 
                feed_dict={lstm.input_x: xex_test, lstm.IND: y_idx_test, lstm.s_len: ex_len_test})

            caps_val_loss.append(str(val_loss))
            
            total_seen_pred = np.array([], dtype=np.int64)

            print("Epoch %d/%d - train loss: %f - val loss: %f" % (epoch, max(0, FLAGS.num_epochs-1), 
                                                                   train_loss, val_loss))
            
            test_batch_pred = np.argmax(logits, 1)
            total_seen_pred = np.concatenate((total_seen_pred, test_batch_pred))
            val_acc = accuracy_score(yex_test, total_seen_pred)
            caps_val_acc.append(str(val_acc))
            
            if val_acc > best_caps_acc:
                best_caps_acc = val_acc
            
            print("Intent Detection Results [INTENTCAPSNET]")
            print(classification_report(yex_test, total_seen_pred, digits=4))
            print("Curr CAPS acc: %f - Best CAPS acc: %f" % (val_acc, best_caps_acc))

            #########
            em_logits = sess.run(lstm.logits, 
                feed_dict={lstm.input_x: label_em, lstm.s_len: label_em_len})
            data['em_logits']=em_logits/em_logits.sum(axis=1,keepdims=1)
            #########

            print("=================================================================================")
            # check INTENTCAPSNET-ZSL performance
            cur_zsl_acc=evaluate_zsl(data, FLAGS, sess)
            zsl_acc.append(str(cur_zsl_acc))
            if cur_zsl_acc > best_zsl_acc:
                best_zsl_acc = cur_zsl_acc
                var_saver.save(sess, os.path.join(FLAGS.ckpt_dir, "model.ckpt"), 1) # save model
            print("Curr ZSL acc: %f - Best ZSL acc: %f" % (cur_zsl_acc, best_zsl_acc))
            epoch_end_time=time.time()
            print("Epoch elapsed time: %f" % (epoch_end_time-epoch_start_time))
        #timelist=np.linspace(0,99,num=100)
        with open('./statistics/eng_single.txt', 'w') as f:
            f.write("caps_train_loss: "+", ".join(caps_train_loss)+'\n')
            f.write("caps_train_acc: "+", ".join(caps_train_acc)+'\n')
            f.write("caps_val_loss: "+", ".join(caps_val_loss)+'\n')
            f.write("caps_val_acc: "+", ".join(caps_val_acc)+'\n')
            f.write("zsl_acc: "+", ".join(zsl_acc)+'\n')