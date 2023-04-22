import sys
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import Read_Data_Triplet_train as rd
import Triplet_Loss as tl
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support
from tools.go_order import *
import sys
import random
from tqdm import tqdm
sys.setrecursionlimit(10000)
np.random.seed(48)
tf.set_random_seed(48)
random.seed(48)

#go = GO_order()
log_location = "/mnt/f/ATGO/re-train/train.log"

def wirte_log(log, log_file):
    os.system("echo " + log + " >> " + log_file)


class Triplet_Network(object):

    def __init__(self, workdir, type, cut_off, layer_list, dropout_rate, learning_rate, margin, alpha, beta, weight, max_iteration, batch_size, round, input_length):

        self.train_feature, self.train_label, self.train_name, \
        self.evaluate_feature, self.evaluate_label, self.evaluate_name, \
        self.test_feature, self.test_label, self.test_name \
            = rd.read_data(workdir, type)

        self.type = type
        self.cut_off = cut_off
        self.layer_list = layer_list
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.weight = weight

        self.max_iteration = max_iteration
        self.batch_size = batch_size
        self.round = round

        self.input_length = input_length
        self.output_length = self.train_label.shape[1]

        self.lamda = 0.01

        self.factor = 1 # note the changed
        if(self.type=="MF"):
            self.factor = 2

        self.model_dir = workdir+"/"+type+"/model/"+str(round)+"/"
        os.system("rm -rf "+ self.model_dir)
        os.makedirs(self.model_dir)

    def dnn(self, x1, x2, x3, keep_prob, is_train):   # deep neural network

        x1 = tf.layers.batch_normalization(x1, training=is_train)
        x2 = tf.layers.batch_normalization(x2, training=is_train)
        x3 = tf.layers.batch_normalization(x3, training=is_train)

        for layer in self.layer_list:

            x1 = tf.layers.dense(inputs = x1, units = layer, activation = tf.nn.tanh)
            x1 = tf.nn.dropout(x1, keep_prob)
            x1 = tf.layers.batch_normalization(x1, training=is_train)

        for layer in self.layer_list:

            x2 = tf.layers.dense(inputs = x2, units = layer, activation = tf.nn.tanh)
            x2 = tf.nn.dropout(x2, keep_prob)
            x2 = tf.layers.batch_normalization(x2, training=is_train)

        for layer in self.layer_list:

            x3 = tf.layers.dense(inputs = x3, units = layer, activation = tf.nn.tanh)
            x3 = tf.nn.dropout(x3, keep_prob)
            x3 = tf.layers.batch_normalization(x3, training=is_train)

        x = tf.concat([x1, x2], axis = 1)
        x = tf.concat([x , x3], axis = 1)

        for i in range(1):

            x = tf.layers.dense(inputs=x, units=int(layer*self.factor), activation=tf.nn.tanh) # note factor
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.batch_normalization(x, training=is_train)

        embeddings = tf.nn.l2_normalize(x, axis=1)

        y_pred = tf.layers.dense(inputs=x, units=self.output_length, activation=tf.nn.sigmoid)

        return embeddings, y_pred

    # def dnn(self, x1, x2, x3, keep_prob, is_train):   # deep neural network

    #     sequence_length = self.input_length # Change this to your sequence length
    #     input_shape = x1.shape[1:]  # assuming x1, x2, x3 have the same shape

    #     x1 = tf.layers.batch_normalization(x1, training=is_train)
    #     x1 = tf.reshape(x1, [-1, sequence_length, int(input_shape[0] // sequence_length)])

    #     x2 = tf.layers.batch_normalization(x2, training=is_train)
    #     x2 = tf.reshape(x2, [-1, sequence_length, int(input_shape[0] // sequence_length)])

    #     x3 = tf.layers.batch_normalization(x3, training=is_train)
    #     x3 = tf.reshape(x3, [-1, sequence_length, int(input_shape[0] // sequence_length)])

    #     # Add a Bidirectional LSTM layer
    #     lstm_units = 16  # Set the number of LSTM units
    #     lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_units)
    #     lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(lstm_units)

    #     x1_bi_lstm, _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell, lstm_cell_bw, x1, dtype=tf.float32)
    #     x1_bi_lstm = tf.concat(x1_bi_lstm, axis=-1)
    #     x2_bi_lstm, _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell, lstm_cell_bw, x2, dtype=tf.float32)
    #     x2_bi_lstm = tf.concat(x2_bi_lstm, axis=-1)
    #     x3_bi_lstm, _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell, lstm_cell_bw, x3, dtype=tf.float32)
    #     x3_bi_lstm = tf.concat(x3_bi_lstm, axis=-1)

    #     x1_flat = tf.reshape(x1_bi_lstm, [-1, 2 * lstm_units])
    #     x2_flat = tf.reshape(x2_bi_lstm, [-1, 2 * lstm_units])
    #     x3_flat = tf.reshape(x3_bi_lstm, [-1, 2 * lstm_units])


    #     for layer in self.layer_list:
    #         x1_flat = tf.keras.layers.Dense(units=layer, activation=tf.nn.tanh)(x1_flat)
    #         x1_flat = tf.keras.layers.Dropout(rate=1-keep_prob)(x1_flat)
    #         x1 = tf.keras.layers.BatchNormalization()(x1_flat, training=is_train)

    #     for layer in self.layer_list:
    #         x2_flat = tf.keras.layers.Dense(units=layer, activation=tf.nn.tanh)(x2_flat)
    #         x2_flat = tf.keras.layers.Dropout(rate=1-keep_prob)(x2_flat)
    #         x2 = tf.keras.layers.BatchNormalization()(x2_flat, training=is_train)

    #     for layer in self.layer_list:
    #         x3_flat = tf.keras.layers.Dense(units=layer, activation=tf.nn.tanh)(x3_flat)
    #         x3_flat = tf.keras.layers.Dropout(rate=1-keep_prob)(x3_flat)
    #         x3 = tf.keras.layers.BatchNormalization()(x3_flat, training=is_train)


    #     x = tf.concat([x1, x2], axis=1)
    #     x = tf.concat([x, x3], axis=1)

    #     for i in range(1):

    #         x = tf.layers.dense(inputs=x, units=int(layer*self.factor), activation=tf.nn.tanh)  # note factor
    #         x = tf.nn.dropout(x, keep_prob)
    #         x = tf.layers.batch_normalization(x, training=is_train)

    #     embeddings = tf.nn.l2_normalize(x, axis=1)

    #     y_pred = tf.layers.dense(inputs=x, units=self.output_length, activation=tf.nn.sigmoid)

    #     return embeddings, y_pred

    def process(self, x1, x2, x3, y, keep_prob, is_train, t_cut_off, t_margin, t_lamda):   # process

        with tf.name_scope('embedding'):
            embeddings, y_pred = self.dnn(x1, x2, x3, keep_prob, is_train)

        with tf.name_scope('caculate_loss'):

            triplet_loss = tl.batch_hard_triplet_loss(embeddings, y, t_cut_off, t_margin)
            global_loss = tl.global_loss(embeddings, y, t_cut_off, t_margin, t_lamda)
            # pos weight 0.5
            cross_entropy = y * tf.log(y_pred + 1e-6) + (1 - y) * tf.log(1+1e-6-y_pred)
            cross_entropy = -tf.reduce_mean(cross_entropy)

            triplet_loss = triplet_loss + self.alpha * cross_entropy + self.beta * global_loss

        with tf.name_scope('adam_optimizer'):

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(triplet_loss)

        return train_step, triplet_loss, embeddings, y_pred


    def running(self, restore=True): # main process
        
        tf.reset_default_graph()
        tf.global_variables_initializer()

        x1 = tf.placeholder(tf.float32, [None, self.input_length])
        x2 = tf.placeholder(tf.float32, [None, self.input_length])
        x3 = tf.placeholder(tf.float32, [None, self.input_length])

        y = tf.placeholder(tf.float32, [None, self.output_length])
        keep_prob = tf.placeholder(tf.float32)
        is_train = tf.placeholder(tf.bool)
        t_cut_off = tf.placeholder(tf.float32)
        t_margin = tf.placeholder(tf.float32)
        t_lamda = tf.placeholder(tf.float32)

        train_step, triplet_loss, embeddings, y_pred = self.process(x1, x2, x3, y, keep_prob, is_train, t_cut_off, t_margin, t_lamda)

        train_data_list = rd.create_batch(self.train_feature, self.train_label, self.train_name, self.batch_size, True)

        evaluate_data_list = rd.create_batch(self.evaluate_feature, self.evaluate_label, self.evaluate_name, self.batch_size, False)

        test_data_list = rd.create_batch(self.test_feature, self.test_label, self.test_name, self.batch_size, False)

        saver = tf.train.Saver(max_to_keep=2)

        with tf.Session() as sess:

            if restore:
                ckpt = tf.train.latest_checkpoint(self.model_dir)
                saver = tf.train.Saver()
                saver.restore(sess, ckpt)
            else:
                sess.run(tf.global_variables_initializer())

            for i in range(1, self.max_iteration + 1):

                train_loss = 0
                train_output = np.zeros([1, int(self.layer_list[len(self.layer_list)-1]*self.factor)])#note
                train_name_list = []

                for train_data in tqdm(train_data_list):

                    sub_train_feature, sub_train_label, sub_train_name = train_data

                    sub_train_feature1 = sub_train_feature[:, 0: self.input_length]
                    sub_train_feature2 = sub_train_feature[:, self.input_length:self.input_length*2]
                    sub_train_feature3 = sub_train_feature[:, self.input_length*2:]


                    train_step.run(feed_dict={x1: sub_train_feature1, x2: sub_train_feature2, x3: sub_train_feature3, y: sub_train_label,
                                              keep_prob: self.dropout_rate,
                                              is_train: True, t_cut_off: self.cut_off,
                                              t_margin: self.margin, t_lamda: self.lamda})


                for train_data in train_data_list:

                    sub_train_feature, sub_train_label, sub_train_name = train_data

                    sub_train_feature1 = sub_train_feature[:, 0: self.input_length]
                    sub_train_feature2 = sub_train_feature[:, self.input_length:self.input_length * 2]
                    sub_train_feature3 = sub_train_feature[:, self.input_length * 2:]

                    train_loss = train_loss + sess.run(triplet_loss, feed_dict={x1: sub_train_feature1, x2: sub_train_feature2, x3: sub_train_feature3, y: sub_train_label,
                                                                                keep_prob: self.dropout_rate,
                                                                                is_train: True, t_cut_off: self.cut_off,
                                                                                t_margin: self.margin, t_lamda: self.lamda})*(len(sub_train_name))

                    current_train_output = sess.run(embeddings, feed_dict={x1: sub_train_feature1, x2: sub_train_feature2, x3: sub_train_feature3, y: sub_train_label,
                                                                           keep_prob: self.dropout_rate,
                                                                           is_train: True, t_cut_off: self.cut_off,
                                                                           t_margin: self.margin, t_lamda: self.lamda})

                    train_output = np.concatenate((train_output, current_train_output), axis = 0)
                    train_name_list.extend(sub_train_name)

                train_output = train_output[1:, :]

                train_loss = train_loss/len(train_name_list)

                print("The "+str(i)+" iteration: ")
                print("Training loss: "+str(train_loss))
                if (i % 10 == 0):
                    # shuffle train
                    random.shuffle(train_data_list)


                    evaluate_loss = 0
                    evaluate_output = np.zeros([1, int(self.layer_list[len(self.layer_list)-1]*self.factor)]) #note
                    evaluate_name_list = []

                    evaluate_pred_matrix = np.zeros([1, self.output_length])

                    for evaluate_data in evaluate_data_list:
                        sub_evaluate_feature, sub_evaluate_label, sub_evaluate_name = evaluate_data

                        sub_evaluate_feature1 = sub_evaluate_feature[:, 0: self.input_length]
                        sub_evaluate_feature2 = sub_evaluate_feature[:, self.input_length:self.input_length * 2]
                        sub_evaluate_feature3 = sub_evaluate_feature[:, self.input_length * 2:]

                        evaluate_loss = evaluate_loss + sess.run(triplet_loss,
                                                                feed_dict={x1: sub_evaluate_feature1, x2: sub_evaluate_feature2, x3: sub_evaluate_feature3, y: sub_evaluate_label,
                                                                            keep_prob: 1,
                                                                            is_train: False, t_cut_off: self.cut_off,
                                                                            t_margin: self.margin, t_lamda: self.lamda}) * (len(sub_evaluate_name))

                        current_evaluate_output = sess.run(embeddings,
                                                        feed_dict={x1: sub_evaluate_feature1, x2: sub_evaluate_feature2, x3: sub_evaluate_feature3, y: sub_evaluate_label,
                                                                    keep_prob: 1,
                                                                    is_train: False, t_cut_off: self.cut_off,
                                                                    t_margin: self.margin, t_lamda: self.lamda})

                        current_y_pred = sess.run(y_pred,
                                                feed_dict={x1: sub_evaluate_feature1, x2: sub_evaluate_feature2, x3: sub_evaluate_feature3, y: sub_evaluate_label,
                                                            keep_prob: 1,
                                                            is_train: False, t_cut_off: self.cut_off,
                                                            t_margin: self.margin, t_lamda: self.lamda})

                        evaluate_output = np.concatenate((evaluate_output, current_evaluate_output), axis=0)
                        evaluate_name_list.extend(sub_evaluate_name)
                        evaluate_pred_matrix = np.concatenate((evaluate_pred_matrix, current_y_pred), axis=0)

                    evaluate_loss = evaluate_loss / len(evaluate_name_list)
                    evaluate_output = evaluate_output[1:, :]
                    evaluate_pred_matrix = evaluate_pred_matrix[1:]

                    print("evaluate loss: " + str(evaluate_loss))


                    test_loss = 0
                    test_output = np.zeros([1, int(self.layer_list[len(self.layer_list)-1]*self.factor)])#note
                    test_name_list = []

                    test_true = []
                    test_pred = []

                    test_pred_matrix = np.zeros([1, self.output_length])

                    for test_data in test_data_list:

                        sub_test_feature, sub_test_label, sub_test_name = test_data

                        sub_test_feature1 = sub_test_feature[:, 0: self.input_length]
                        sub_test_feature2 = sub_test_feature[:, self.input_length:self.input_length * 2]
                        sub_test_feature3 = sub_test_feature[:, self.input_length * 2:]

                        test_loss = test_loss + sess.run(triplet_loss, feed_dict={x1: sub_test_feature1, x2: sub_test_feature2, x3: sub_test_feature3, y: sub_test_label,
                                                                                keep_prob: 1,
                                                                                is_train: False, t_cut_off: self.cut_off,
                                                                                t_margin: self.margin, t_lamda: self.lamda})*(len(sub_test_name))

                        current_test_output = sess.run(embeddings, feed_dict={x1: sub_test_feature1, x2: sub_test_feature2, x3: sub_test_feature3, y: sub_test_label,
                                                                            keep_prob: 1,
                                                                            is_train: False, t_cut_off: self.cut_off,
                                                                            t_margin: self.margin, t_lamda: self.lamda})

                        current_y_pred = sess.run(y_pred, feed_dict={x1: sub_test_feature1, x2: sub_test_feature2, x3: sub_test_feature3, y: sub_test_label,
                                                                    keep_prob: 1,
                                                                    is_train: False, t_cut_off: self.cut_off,
                                                                    t_margin: self.margin, t_lamda: self.lamda})

                        test_output = np.concatenate((test_output, current_test_output), axis=0)
                        test_name_list.extend(sub_test_name)
                        test_pred_matrix = np.concatenate((test_pred_matrix, current_y_pred), axis=0)

                        test_true.extend(sub_test_label)
                        test_pred.extend(current_y_pred)

                    test_loss = test_loss / len(test_name_list)
                    test_output = test_output[1:, :]
                    test_pred_matrix = test_pred_matrix[1:]

                    # # check if have the backpropagation order attribute
                    # if not hasattr(self, 'back_order'):
                    #     #self.back_order = go.brute_order(aspect=self.type)
                    #     self.back_order = go.init_one_aspect(aspect=self.type)
                    # # backpropagation
                    # test_true = np.array([list(x) for x in test_true])
                    # test_pred = np.array([list(x) for x in test_pred])
                    # # backpropagation for the prediction and true label
                    # for c, p in self.back_order:
                    #     test_true[:, p] = np.maximum(test_true[:, p], test_true[:, c])
                    #     test_pred[:, p] = np.maximum(test_pred[:, p], test_pred[:, c])


                    print("Testing loss: "+str(test_loss))
                    # print every 10 iterations
                    test_true = np.array([list(x) for x in test_true])
                    test_pred = np.array([list(x) for x in test_pred])
                    t, precision, recall, f1 = find_fmax(test_true, test_pred)
                    precision = str(precision)[0:5]
                    recall = str(recall)[0:5]
                    f1 = str(f1)[0:5]
                    log = f'Round {self.round} {self.type}: Testing loss: {test_loss}, p:{precision}, r:{recall}, f1:{f1}, t:{t}, iteration: {i}'
                    print(log)
                    wirte_log(log, log_location)

                if (i % 50 == 0):
                    saver.save(sess, self.model_dir + "/model" + str(i), global_step=i)

                if(i==self.max_iteration):
                    saver.save(sess, self.model_dir + "/model" + str(i), global_step=i)

    def eval(self, restore=True): # main process
        
        tf.reset_default_graph()
        tf.global_variables_initializer()

        x1 = tf.placeholder(tf.float32, [None, self.input_length])
        x2 = tf.placeholder(tf.float32, [None, self.input_length])
        x3 = tf.placeholder(tf.float32, [None, self.input_length])

        y = tf.placeholder(tf.float32, [None, self.output_length])
        keep_prob = tf.placeholder(tf.float32)
        is_train = tf.placeholder(tf.bool)
        t_cut_off = tf.placeholder(tf.float32)
        t_margin = tf.placeholder(tf.float32)
        t_lamda = tf.placeholder(tf.float32)

        train_step, triplet_loss, embeddings, y_pred = self.process(x1, x2, x3, y, keep_prob, is_train, t_cut_off, t_margin, t_lamda)
        test_data_list = rd.create_batch(self.test_feature, self.test_label, self.test_name, self.batch_size, False)

        with tf.Session() as sess:

            if restore:
                ckpt = tf.train.latest_checkpoint(self.model_dir)
                saver = tf.train.Saver()
                saver.restore(sess, ckpt)
            else:
                sess.run(tf.global_variables_initializer())

            for i in range(1, self.max_iteration + 1):
                test_loss = 0
                test_output = np.zeros([1, int(self.layer_list[len(self.layer_list)-1]*self.factor)])#note
                test_name_list = []

                test_true = []
                test_pred = []

                test_pred_matrix = np.zeros([1, self.output_length])

                for test_data in test_data_list:

                    sub_test_feature, sub_test_label, sub_test_name = test_data

                    sub_test_feature1 = sub_test_feature[:, 0: self.input_length]
                    sub_test_feature2 = sub_test_feature[:, self.input_length:self.input_length * 2]
                    sub_test_feature3 = sub_test_feature[:, self.input_length * 2:]

                    test_loss = test_loss + sess.run(triplet_loss, feed_dict={x1: sub_test_feature1, x2: sub_test_feature2, x3: sub_test_feature3, y: sub_test_label,
                                                                              keep_prob: 1,
                                                                              is_train: False, t_cut_off: self.cut_off,
                                                                              t_margin: self.margin, t_lamda: self.lamda})*(len(sub_test_name))

                    current_test_output = sess.run(embeddings, feed_dict={x1: sub_test_feature1, x2: sub_test_feature2, x3: sub_test_feature3, y: sub_test_label,
                                                                          keep_prob: 1,
                                                                           is_train: False, t_cut_off: self.cut_off,
                                                                          t_margin: self.margin, t_lamda: self.lamda})

                    current_y_pred = sess.run(y_pred, feed_dict={x1: sub_test_feature1, x2: sub_test_feature2, x3: sub_test_feature3, y: sub_test_label,
                                                                 keep_prob: 1,
                                                                 is_train: False, t_cut_off: self.cut_off,
                                                                 t_margin: self.margin, t_lamda: self.lamda})

                    test_output = np.concatenate((test_output, current_test_output), axis=0)
                    test_name_list.extend(sub_test_name)
                    test_pred_matrix = np.concatenate((test_pred_matrix, current_y_pred), axis=0)

                    test_true.extend(sub_test_label)
                    test_pred.extend(current_y_pred)

                test_loss = test_loss / len(test_name_list)
                test_output = test_output[1:, :]
                test_pred_matrix = test_pred_matrix[1:]
                print("Testing loss: "+str(test_loss))

                test_true = np.array([list(x) for x in test_true])
                test_pred = np.array([list(x) for x in test_pred])


                # # check if have the backpropagation order attribute
                # if not hasattr(self, 'back_order'):
                #     self.back_order = go.brute_order(aspect=self.type)
                #     #self.back_order = go.init_one_aspect(aspect=self.type)
                # # backpropagation for the prediction and true label
                # for c, p in self.back_order:
                #     test_true[:, p] = np.maximum(test_true[:, p], test_true[:, c])
                #     test_pred[:, p] = np.maximum(test_pred[:, p], test_pred[:, c])

                t, precision, recall, f1 = find_fmax(test_true, test_pred)
                log = f'Round {self.round} {self.type}: Testing loss: {test_loss}, p:{precision}, r:{recall}, f1:{f1}, t:{t}'
                print(log)
                wirte_log(log, log_location)

                # if (i % 50 == 0):
                #     saver.save(sess, self.model_dir + "/model" + str(i), global_step=i)

                # if(i==self.max_iteration):
                #     saver.save(sess, self.model_dir + "/model" + str(i), global_step=i)



def find_fmax(y_true, y_pred):
    fmax_list = []
    max_index = 0
    max_f1 = 0
    for i in range(0, 101):
        threshold = i / 100
        tp, fp, fn = 0, 0, 0
        p, r, f1 = find_f1(y_true, y_pred, threshold)
        if f1 >= max_f1:
            max_f1 = f1
            max_index = i
        fmax_list.append((threshold, p, r, f1))
    f05 = fmax_list[50]
    return fmax_list[max_index]


def find_f1(y_true, y_pred, threshold):
    precision = []
    recall = []
    y_pred = y_pred.copy()
    y_pred = np.where(y_pred > threshold, 1, 0)
    for i in range(len(y_true)):
        tp, fp, fn = 0, 0, 0
        tp = tp + np.sum(y_true[i] * y_pred[i])
        fp = fp + np.sum((1 - y_true[i]) * y_pred[i])
        fn = fn + np.sum(y_true[i] * (1 - y_pred[i]))
        precision.append(tp / (tp + fp) if (tp + fp) != 0 else 0)
        recall.append(tp / (tp + fn) if (tp + fn) != 0 else 0)
    precision = np.mean(precision)
    recall = np.mean(recall)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
    return precision, recall, f1



def use_cuda():
    # use GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f'GPU {gpu} dynamic memory growth enabled')
        except RuntimeError as e:
            print(e)


if __name__=="__main__":
    index_dict = dict()
    # 630 580 270
    index_dict["MF"] = 63*5
    index_dict["BP"] = 58*5
    index_dict["CC"] = 27*5
    workdir = sys.argv[1]
    type = sys.argv[2]
    cut_off = 0.8
    layer_list = [1024]
    dropout_rate = 0.6
    learning_rate = 0.0001
    #learning_rate = 0.0004
    margin = 0.1
    alpha = 5 #default 5.0
    beta = 2 #default 2.0
    weight = 0.8
    batch_size = 64
    #batch_size = 256

    use_cuda()

    #for round in range(1, 11):
    #for round in range(21, 22):
    for round in range(1, 2):
        max_iteration = index_dict[type]
        #max_iteration = 2
        tn = Triplet_Network(workdir, type, cut_off, layer_list, dropout_rate, learning_rate,
                             margin, alpha, beta, weight, max_iteration, batch_size, round, 2560)
        tn.running(restore=False)
        #tn.eval()

    
    
























