import os
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf  # Version 1.0.0 (some previous versions are used in past commits)
from sklearn import metrics
import random
from random import randint
import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from itertools import chain, count
from sklearn.neighbors import NearestNeighbors

n_steps = 32

class openpose_human:
    def __init__(self, camera=0,resize='0x0',resize_out_ratio=4.0,model='mobilenet_thin',show_process=False):
        logger = logging.getLogger('TfPoseEstimator-WebCam')
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.debug('initialization %s : %s' % (model, get_graph_path(model)))
        w, h = model_wh(resize)
        if w > 0 and h > 0:
            e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))
        else:
            e = TfPoseEstimator(get_graph_path(model), target_size=(432, 368))
        logger.debug('cam read+')
        cam = cv2.VideoCapture(camera)
        ret_val, image = cam.read()
        image_h, image_w = image.shape[:2]
        # logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
        fps_time = 0
        videostep = 0
        human_keypoint = []
        while True:
            ret_val, image = cam.read()
            logger.debug('image process+')
            humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)
            for human in humans:
                human_keypoint.append(openpose_human.write_coco_json(human,image_w,image_h))
            videostep += 1
            if (videostep == 32):
                videostep = 0
                activity_human(human_keypoint)
                human_keypoint = []
            logger.debug('postprocess+')
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
            logger.debug('show+')
            cv2.putText(image,
                        "FPS: %f" % (1.0 / (time.time() - fps_time)),
                        (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
            cv2.imshow('tf-pose-estimation result', image)
            fps_time = time.time()
            
            tf.reset_default_graph() # Reset the graph
            
            if cv2.waitKey(1) == 27:
                break
            logger.debug('finished+')
        cv2.destroyAllWindows()
        
    def write_coco_json(human, image_w, image_h):
        keypoints = []
        coco_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        for coco_id in coco_ids:
            if coco_id not in human.body_parts.keys():
                keypoints.extend([0, 0])
                continue
            body_part = human.body_parts[coco_id]
            keypoints.extend([round(body_part.x * image_w, 3), round(body_part.y * image_h, 3)])
        return keypoints
        
        
class activity_human:

    def __init__(self, human_keypoint):
        # Useful Constants
        # Output classes to learn how to classify
        LABELS = [    
            "JUMPING",
            "JUMPING_JACKS",
            "BOXING",
            "WAVING_2HANDS",
            "WAVING_1HAND",
            "CLAPPING_HANDS"

        ] 
        DATASET_PATH = "data/HAR_pose_activities/database/"
        # X_train_path = DATASET_PATH + "X_train.txt"
        # X_test_path = DATASET_PATH + "X_test.txt"
        # X_test_path = "utilities/something/something.txt"
        # y_train_path = DATASET_PATH + "Y_train.txt"
        # y_test_path = DATASET_PATH + "Y_test.txt"
        # n_steps = 32 # 32 timesteps per series
        # n_steps = 1 # 32 timesteps per series
        # X_train = load_X(X_train_path)
        # X_test = activity_human.load_X(X_test_path)
        # X_test = activity_human.load_XLive(human_keypoint)
        #print X_test
        # y_train = load_y(y_train_path)
        # y_test = activity_human.load_y(y_test_path)
        # proof that it actually works for the skeptical: replace labelled classes with random classes to train on
        #for i in range(len(y_train)):
        #    y_train[i] = randint(0, 5)
        
        # Input Data 
        X_test = activity_human.load_XLive(human_keypoint)
        # n_input = len(X_train[0][0])  # num input parameters per timestep
        # training_data_count = len(X_train)  # 4519 training series (with 50% overlap between each serie)
        # test_data_count = len(X_test)  # 1197 test series
        
        self.n_input = len(X_test[0][0])
        
        self.n_hidden = 34 # Hidden layer num of features
        n_classes = 6
        n_steps = 32
        
        #updated for learning-rate decay
        # calculated as: decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
        decaying_learning_rate = True
        learning_rate = 0.0025 #used if decaying_learning_rate set to False
        init_learning_rate = 0.005
        decay_rate = 0.96 #the base of the exponential in the decay
        decay_steps = 100000 #used in decay every 60000 steps with a base of 0.96
        global_step = tf.Variable(0, trainable=False)
        lambda_loss_amount = 0.0015
        # training_iters = training_data_count *300  # Loop 300 times on the dataset, ie 300 epochs
        # training_iters = training_data_count *60
        # training_iters = training_data_count *120
        # training_iters = training_data_count *1
        batch_size = 512
        display_iter = batch_size*8  # To show test set accuracy during training
        
        
        # print("(X shape, y shape, every X's mean, every X's standard deviation)")
        # print(X_train.shape, y_test.shape, np.mean(X_test), np.std(X_test))
        # print("\nThe dataset has not been preprocessed, is not normalised etc")
        # for _ in range(3):
        #     tf.reset_default_graph()
        #     var = tf.Variable(0)
        #     with tf.Session() as session:
        #         session.run(tf.global_variables_initializer())
        #         print(len(session.graph._nodes_by_name.keys()))
        
        
        #### Build the network
        # Graph input/output
        x = tf.placeholder(tf.float32, [None, n_steps, self.n_input])
        y = tf.placeholder(tf.float32, [None, n_classes])
        # Graph weights
        weights = {
            'hidden': tf.Variable(tf.random_normal([self.n_input, self.n_hidden])), # Hidden layer weights
            'out': tf.Variable(tf.random_normal([self.n_hidden, n_classes], mean=1.0))
        }
        biases = {
            'hidden': tf.Variable(tf.random_normal([self.n_hidden])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }
        pred = activity_human.LSTM_RNN(self, x, weights, biases)
        
        # Loss, optimizer and evaluation
        l2 = lambda_loss_amount * sum(
            tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
        ) # L2 loss prevents this overkill neural network to overfit the data
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) + l2 # Softmax loss
        
        if decaying_learning_rate:
            learning_rate = tf.train.exponential_decay(init_learning_rate, global_step*batch_size, decay_steps, decay_rate, staircase=True)
            
        #decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps) #exponentially decayed learning rate
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,global_step=global_step) # Adam Optimizer
        # correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
        # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        # if decaying_learning_rate:
        #     learning_rate = tf.train.exponential_decay(init_learning_rate, global_step*batch_size, decay_steps, decay_rate, staircase=True)
        
        test_losses = []
        test_accuracies = []
        train_losses = []
        train_accuracies = []
        sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
        # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        init = tf.global_variables_initializer()
        sess.run(init)
        
        
        # training_iters = training_data_count *30
        #create saver before training
        saver = tf.train.Saver(var_list={'wh':weights['hidden'], 'wo':weights['out'], 'bh':biases['hidden'], 'bo':biases['out']})
        load = True
        train = False
        update = False
        #check if you want to retrain or import a saved model
        
        print("aaa")
        if load:
            saver.restore(sess, DATASET_PATH + "model.ckpt")
            print("Model restored.")
        print("bbb")
        
        correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        # Perform Training steps with "batch_size" amount of data at each loop. 
        # Elements of each batch are chosen randomly, without replacement, from X_train, 
        # restarting when remaining datapoints < batch_size
        step = 1
        # unsampled_indices = range(0,len(X_train))
        
        ##### Check if you want to save your current model
        # if update:
            # save_path = saver.save(sess, DATASET_PATH + "model.ckpt")
            # print("Model saved in file: %s" % save_path)
            
        time_start = time.time()
            
        ##### Inferencing
        # X_infer_path = "utilities/something/something.txt"
        # X_infer_path = DATASET_PATH + "X_test.txt"
        # X_val = load_X(X_infer_path)
        X_test = activity_human.load_XLive(human_keypoint)
        preds = sess.run(
            [pred],
            feed_dict={
                x: X_test
           }
        )
        print(preds)
        
        time_stop = time.time()
        print("TOTAL TIME:  {}".format(time_stop - time_start))
        
        # (Inline plots: )
        # %matplotlib inline
        # font = {
            # 'family' : 'Bitstream Vera Sans',
            # 'weight' : 'bold',
            # 'size'   : 18
        # }
        # matplotlib.rc('font', **font)
        # width = 12
        # height = 12
        # plt.figure(figsize=(width, height))
        # indep_train_axis = np.array(range(batch_size, (len(train_losses)+1)*batch_size, batch_size))
        #plt.plot(indep_train_axis, np.array(train_losses),     "b--", label="Train losses")
        # plt.plot(indep_train_axis, np.array(train_accuracies), "g--", label="Train accuracies")
        # indep_test_axis = np.append(
            # np.array(range(batch_size, len(test_losses)*display_iter, display_iter)[:-1]),
            # [training_iters]
        # )
        # plt.plot(indep_test_axis, np.array(test_losses), "b-", linewidth=2.0, label="Test losses")
        # plt.plot(indep_test_axis, np.array(test_accuracies), "b-", linewidth=2.0, label="Test accuracies")
        # print(len(test_accuracies))
        # print(len(train_accuracies))
        # plt.title("Training session's Accuracy over Iterations")
        # plt.legend(loc='lower right', shadow=True)
        # plt.ylabel('Training Accuracy')
        # plt.xlabel('Training Iteration')
        # plt.show()
        # Results
        # predictions = one_hot_predictions.argmax(1)
        # print("Testing Accuracy: {}%".format(100*accuracy_fin))
        # print("")
        # print("Precision: {}%".format(100*metrics.precision_score(y_test, predictions, average="weighted")))
        # print("Recall: {}%".format(100*metrics.recall_score(y_test, predictions, average="weighted")))
        # print("f1_score: {}%".format(100*metrics.f1_score(y_test, predictions, average="weighted")))
        # print("")
        # print("Confusion Matrix:")
        # print("Created using test set of {} datapoints, normalised to % of each class in the test dataset".format(len(y_test)))
        # confusion_matrix = metrics.confusion_matrix(y_test, predictions)
        #print(confusion_matrix)
        # normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100
        # Plot Results: 
        # width = 12
        # height = 12
        # plt.figure(figsize=(width, height))
        # plt.imshow(
            # normalised_confusion_matrix, 
            # interpolation='nearest', 
            # cmap=plt.cm.Blues
        # )
        # plt.title("Confusion matrix \n(normalised to % of total test data)")
        # plt.colorbar()
        # tick_marks = np.arange(n_classes)
        # plt.xticks(tick_marks, LABELS, rotation=90)
        # plt.yticks(tick_marks, LABELS)
        # plt.tight_layout()
        # plt.ylabel('True label')
        # plt.xlabel('Predicted label')
        # plt.show()
        #
        #X_val_path = DATASET_PATH + "X_val.txt"
        #X_val = load_X(X_val_path)
        #print X_val
        #
        #preds = sess.run(
        #    [pred],
        #    feed_dict={
        #        x: X_val
        #   }
        #)
        #
        #print preds
        #sess.close()
        # print(test_accuracies)
        
    # Load the networks inputs

    def load_X(X_path):
        file = open(X_path, 'r')
        X_ = np.array(
            [elem for elem in [
                row.split(',') for row in file
            ]], 
            dtype=np.float32
        )
        file.close()
        blocks = int(len(X_) / n_steps)
        X_ = np.array(np.split(X_,blocks))
        return X_ 

    # Load the networks outputs
    def load_XLive(keypoints):
        # print(keypoints)
        
        for row in keypoints:
            print(len(row))
        
        X_ = np.array(keypoints,dtype=np.float32)
        
        blocks = int(len(X_) / n_steps)
        X_ = np.array(np.split(X_,blocks))
        return X_ 

    def load_y(y_path):
        file = open(y_path, 'r')
        y_ = np.array(
            [elem for elem in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]], 
            dtype=np.int32
        )
        file.close()
        
        # for 0-based indexing 
        return y_ - 1

    def LSTM_RNN(self, _X, _weights, _biases):
        # model architecture based on "guillaume-chevalier" and "aymericdamien" under the MIT license.

        _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
        _X = tf.reshape(_X, [-1, self.n_input])   
        # Rectifies Linear Unit activation function used
        _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
        # Split data because rnn cell needs a list of inputs for the RNN inner loop
        _X = tf.split(_X, n_steps, 0) 

        # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
        lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

        # A single output is produced, in style of "many to one" classifier, refer to http://karpathy.github.io/2015/05/21/rnn-effectiveness/ for details
        lstm_last_output = outputs[-1]
        
        # Linear activation
        return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']


    def extract_batch_size(_train, _labels, _unsampled, batch_size):
        # Fetch a "batch_size" amount of data and labels from "(X|y)_train" data. 
        # Elements of each batch are chosen randomly, without replacement, from X_train with corresponding label from Y_train
        # unsampled_indices keeps track of sampled data ensuring non-replacement. Resets when remaining datapoints < batch_size    
        
        shape = list(_train.shape)
        shape[0] = batch_size
        batch_s = np.empty(shape)
        batch_labels = np.empty((batch_size,1)) 

        for i in range(batch_size):
            # Loop index
            # index = random sample from _unsampled (indices)
            index = random.choice(_unsampled)
            batch_s[i] = _train[index] 
            batch_labels[i] = _labels[index]
            
            _unsampled = list(_unsampled)
            
            _unsampled.remove(index)

        return batch_s, batch_labels, _unsampled

    def one_hot(y_):
        # One hot encoding of the network outputs
        # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
        y_ = y_.reshape(len(y_))
        n_values = int(np.max(y_)) + 1
        return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS
        
if __name__ == '__main__':
    openpose_human()

