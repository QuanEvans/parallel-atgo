import sys
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import Read_Data_Triplet as rd
import Triplet_Loss as tl
import numpy as np
import Calculate_results_thread as cr
np.random.seed(1)
tf.set_random_seed(1)
from configure import data_dir
import time


class Triplet_Network(object):

    def __init__(self, workdir, type, cut_off, layer_list, dropout_rate, learning_rate, margin, alpha, beta, max_iteration, batch_size, round, input_length):

        self.train_feature, self.train_label, self.train_name = rd.read_train_evaluate_data(data_dir, type)
        self.test_feature, self.test_label, self.test_name  = rd.read_test_data(workdir, type)

        self.type = type
        self.cut_off = cut_off
        self.layer_list = layer_list
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.margin = margin
        self.alpha = alpha
        self.beta = beta

        self.max_iteration = max_iteration
        self.batch_size = batch_size
        self.round = round

        self.input_length = input_length
        self.output_length = self.train_label.shape[1]

        self.lamda = 0.01
        self.model_dir = data_dir + "/" + type + "/model/" + str(round)+"/"
        self.rewrite_single(self.model_dir, max_iteration)

        self.factor = 1
        if(self.type=="MF"):
            self.factor = 2

    def rewrite_single(sekf, modeldir, index):

        model_name = "model" + str(index) + "-" + str(index)
        f = open(modeldir + "/checkpoint", "w")
        f.write("model_checkpoint_path:" + "\"" + model_name + "\"" + "\n")
        f.write("all_model_checkpoint_paths:" + "\"" + model_name + "\"" + "\n")
        f.flush()
        f.close()

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

            x = tf.layers.dense(inputs=x, units=int(layer*self.factor), activation=tf.nn.tanh)
            x = tf.nn.dropout(x, keep_prob)
            x = tf.layers.batch_normalization(x, training=is_train)

        embeddings = tf.nn.l2_normalize(x, axis=1)

        y_pred = tf.layers.dense(inputs=x, units=self.output_length, activation=tf.nn.sigmoid)

        return embeddings, y_pred

    def process(self, x1, x2, x3, y, keep_prob, is_train, t_cut_off, t_margin, t_lamda):   # process

        with tf.name_scope('embedding'):
            embeddings, y_pred = self.dnn(x1, x2, x3, keep_prob, is_train)

        with tf.name_scope('caculate_loss'):

            triplet_loss = tl.batch_hard_triplet_loss(embeddings, y, t_cut_off, t_margin)
            global_loss = tl.global_loss(embeddings, y, t_cut_off, t_margin, t_lamda)

            cross_entropy = y * tf.log(y_pred + 1e-6) + (1 - y) * tf.log(1+1e-6-y_pred)
            cross_entropy = -tf.reduce_mean(cross_entropy)

            triplet_loss = triplet_loss + self.alpha * cross_entropy + self.beta * global_loss

        with tf.name_scope('adam_optimizer'):

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(triplet_loss)

        return train_step, triplet_loss, embeddings, y_pred


    def running(self): # main process
	    
        time0 = time.time()
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

            ckpt = tf.train.latest_checkpoint(self.model_dir)
            saver = tf.train.Saver()
            saver.restore(sess, ckpt)

            train_loss = 0
            train_output = np.zeros([1, int(self.layer_list[len(self.layer_list) - 1])*self.factor])
            train_name_list = []

            train_data_list = rd.create_batch(self.train_feature, self.train_label, self.train_name, self.batch_size, True)

            
            for train_data in train_data_list:

                sub_train_feature, sub_train_label, sub_train_name = train_data

                sub_train_feature1 = sub_train_feature[:, 0: self.input_length]
                sub_train_feature2 = sub_train_feature[:, self.input_length:self.input_length * 2]
                sub_train_feature3 = sub_train_feature[:, self.input_length * 2:]

                train_loss = train_loss + sess.run(triplet_loss,
                                                   feed_dict={x1: sub_train_feature1, x2: sub_train_feature2,
                                                              x3: sub_train_feature3, y: sub_train_label,
                                                              keep_prob: self.dropout_rate,
                                                              is_train: True, t_cut_off: self.cut_off,
                                                              t_margin: self.margin, t_lamda: self.lamda}) * (len(sub_train_name))

                current_train_output = sess.run(embeddings, feed_dict={x1: sub_train_feature1, x2: sub_train_feature2,
                                                                       x3: sub_train_feature3, y: sub_train_label,
                                                                       keep_prob: self.dropout_rate,
                                                                       is_train: True, t_cut_off: self.cut_off,
                                                                       t_margin: self.margin, t_lamda: self.lamda})

                train_output = np.concatenate((train_output, current_train_output), axis=0)
                train_name_list.extend(sub_train_name)

            train_output = train_output[1:, :]

            train_loss = train_loss / len(train_name_list)

            print("The " + str(max_iteration) + " iteration: ")
            print("Training loss: " + str(train_loss))



            test_loss = 0
            test_output = np.zeros([1, int(self.layer_list[len(self.layer_list) - 1])*self.factor])
            test_name_list = []

            test_pred_matrix = np.zeros([1, self.output_length])

            for test_data in test_data_list:
                sub_test_feature, sub_test_label, sub_test_name = test_data

                sub_test_feature1 = sub_test_feature[:, 0: self.input_length]
                sub_test_feature2 = sub_test_feature[:, self.input_length:self.input_length * 2]
                sub_test_feature3 = sub_test_feature[:, self.input_length * 2:]

                test_loss = test_loss + sess.run(triplet_loss, feed_dict={x1: sub_test_feature1, x2: sub_test_feature2,
                                                                          x3: sub_test_feature3, y: sub_test_label,
                                                                          keep_prob: 1,
                                                                          is_train: False, t_cut_off: self.cut_off,
                                                                          t_margin: self.margin,
                                                                          t_lamda: self.lamda}) * (len(sub_test_name))

                current_test_output = sess.run(embeddings, feed_dict={x1: sub_test_feature1, x2: sub_test_feature2,
                                                                      x3: sub_test_feature3, y: sub_test_label,
                                                                      keep_prob: 1,
                                                                      is_train: False, t_cut_off: self.cut_off,
                                                                      t_margin: self.margin, t_lamda: self.lamda})

                current_y_pred = sess.run(y_pred, feed_dict={x1: sub_test_feature1, x2: sub_test_feature2,
                                                             x3: sub_test_feature3, y: sub_test_label,
                                                             keep_prob: 1,
                                                             is_train: False, t_cut_off: self.cut_off,
                                                             t_margin: self.margin, t_lamda: self.lamda})

                test_output = np.concatenate((test_output, current_test_output), axis=0)
                test_name_list.extend(sub_test_name)
                test_pred_matrix = np.concatenate((test_pred_matrix, current_y_pred), axis=0)

            test_loss = test_loss / len(test_name_list)
            test_output = test_output[1:, :]
            test_pred_matrix = test_pred_matrix[1:]

            print("Testing loss: " + str(test_loss))


            time1 = time.time()
            print("train and test time: " + str(time1 - time0))
            localtime = time.asctime(time.localtime(time.time()))
            print("Local current time :", localtime)
            cr.calculate_result(workdir, np.array(train_output), np.array(test_output), train_name_list, test_name_list, max_iteration, type, round)
            # note: this step is heavily I/O bound, using SSD is recommended
            time2 = time.time()
            print("calculate_result time: " + str(time2 - time1))
            localtime = time.asctime(time.localtime(time.time()))
            print("Local current time :", localtime)
            cr.calculate_pred_label(workdir, type, test_name_list, test_pred_matrix, round, self.max_iteration)
            time3 = time.time()
            print("calculate_pred_label time: " + str(time3 - time2))
            localtime = time.asctime(time.localtime(time.time()))
            print("Local current time :", localtime)


def use_cuda():
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

    workdir = sys.argv[1]
    type = sys.argv[2]
    round = int(sys.argv[3])
    max_iteration = int(sys.argv[4])
    cut_off = 0.8
    layer_list = [1024]
    dropout_rate = 0.6
    learning_rate = 0.0001
    margin = 0.1
    alpha = 5.0
    beta = 2.0
    batch_size = 512

    use_cuda()

    print(type + " " + str(round) + " " + str(max_iteration))

    tn = Triplet_Network(workdir, type, cut_off, layer_list, dropout_rate, learning_rate,
                         margin, alpha, beta, max_iteration, batch_size, round, 1280)
    tn.running()























