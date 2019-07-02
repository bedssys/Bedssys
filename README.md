
## RNN for Human Activity Recognition - 2D Pose Input

This experiment is the classification of human activities using a 2D pose time series dataset and an LSTM RNN.
The idea is to prove the concept that using a series of 2D poses, rather than 3D poses or a raw 2D images, can produce an accurate estimation of the behaviour of a person or animal.
This is a step towards creating a method of classifying an animal's current behaviour state and predicting it's likely next state, allowing for better interaction with an autonomous mobile robot.

## Objectives

The aims of this experiment are:

-  To determine if 2D pose has comparable accuracy to 3D pose for use in activity recognition. This would allow the use of RGB only cameras for human and animal pose estimation, as opposed to RGBD or a large motion capture dataset.


- To determine if  2D pose has comparable accuracy to using raw RGB images for use in activity recognition. This is based on the idea that limiting the input feature vector can help to deal with a limited dataset, as is likely to occur in animal activity recognition, by allowing for a smaller model to be used (citation required).


- To verify the concept for use in future works involving behaviour prediction from motion in 2D images.

The network used in this experiment is based on that of Guillaume Chevalier, 'LSTMs for Human Activity Recognition, 2016'  https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition, available under the MIT License.
Notable changes that have been made (other than accounting for dataset sizes) are:
 - Adapting for use with a large dataset ordered by class, using random sampling without replacement for mini-batch.  
 This allows for use of smaller batch sizes when using a dataset ordered by class. "It has been observed in practice that when using a larger batch there is a significant degradation in the quality of the model, as measured by its ability to generalize"  
      _N.S Keskar, D. Mudigere, et al, 'On Large-Batch Training for Deep Learning: Generalization Gap and Sharp 
      Minima', ICLR 2017_ https://arxiv.org/abs/1609.04836
      
 - Exponentially decaying learning rate implemented



## Dataset overview

The dataset consists of pose estimations, made using the software OpenPose (https://github.com/CMU-Perceptual-Computing-Lab/openpose's) on a subset of the Berkeley Multimodal Human Action Database (MHAD) dataset http://tele-immersion.citris-uc.org/berkeley_mhad.

This dataset is comprised of 12 subjects doing the following 6 actions for 5 repetitions, filmed from 4 angles, repeated 5 times each.  

- JUMPING,
- JUMPING_JACKS,
- BOXING,
- WAVING_2HANDS,
- WAVING_1HAND,
- CLAPPING_HANDS.

In total, there are 1438 videos (2 were missing) made up of 211200 individual frames.

The below image is an example of the 4 camera views during the 'boxing' action for subject 1

![alt text](images/boxing_all_views.gif.png "Title")

The input for the LSTM is the 2D position of 18 joints across a timeseries of frames numbering n_steps (window-width), with an associated class label for the frame series.  
A single frame's input (where j refers to a joint) is stored as:

[  j0_x,  j0_y, j1_x, j1_y , j2_x, j2_y, j3_x, j3_y, j4_x, j4_y, j5_x, j5_y, j6_x, j6_y, j7_x, j7_y, j8_x, j8_y, j9_x, j9_y, j10_x, j10_y, j11_x, j11_y, j12_x, j12_y, j13_x, j13_y, j14_x, j14_y, j15_x, j15_y, j16_x, j16_y, j17_x, j17_y ]

For the following experiment, very little preprocessing has been done to the dataset.  
The following steps were taken:
1. openpose run on individual frames, for each subject, action and view, outputting JSON of 18 joint x and y position keypoints and accuracies per frame
2. JSONs converted into txt format, keeping only x and y positions of each frame, action being performed during frame, and order of frames. This is used to create a database of associated activity class number and corresponding series of joint 2D positions
3. No further prepossessing was performed.  

In some cases, multiple people were detected in each frame, in which only the first detection was used.

The data has not been normalised with regards to subject position in the frame, motion across frame (if any), size of the subject, speed of action etc. It is essentially the raw 2D position of each joint viewed from a stationary camera.  
In many cases, individual joints were not located and a position of [0.0,0.0] was given for that joint

A summary of the dataset used for input is:

 - 211200 individual images 
 - n_steps = 32 frames (~=1.5s at 22Hz)
 - Images with noisy pose detection (detection of >=2 people) = 5132  
 - Training_split = 0.8
 - Overlap = 0.8125 (26 / 32) ie 26 frame overlap
   - Length X_train = 22625 * 32 frames
   - Length X_test = 5751 * 32 frames
   
Note that their is no overlap between test and train sets, which were seperated by activity repetition entirely, before creating the 26 of 32 frame overlap.




## Training and Results below: 
Training took approximately 4 mins running on a single GTX1080Ti, and was run for 22,000,000ish iterations with a batch size of 5000  (600 epochs)



```python
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf  # Version 1.0.0 (some previous versions are used in past commits)
from sklearn import metrics
import random
from random import randint
import time
import os
```

## Preparing dataset:


```python
# Useful Constants

# Output classes to learn how to classify
# LABELS = [    
#     "JUMPING",
#     "JUMPING_JACKS",
# #     "BOXING",
#     "WAVING_2HANDS",
#     "WAVING_1HAND",
#     "CLAPPING_HANDS"
# ] 

# LABELS = [    
#     "GO_IN",
#     "GO_OUT",
#     "WALK_LEFT",
#     "WALK_RIGHT"
# ] 

# LABELS = [    
#     "normal", "anomaly"
# ] 

LABELS = [
    "jalan_DR", "jalan_UR", "jalan_DL", "jalan_UL",
    "sapu_DR", "sapu_UR", "sapu_DL", "sapu_UL",
    "suspicious_DR", "suspicious_UR", "suspicious_DL", "suspicious_UL",
    "out_door_SE", "out_door_SW", "in_door_SE", "in_door_SW",
    "idle"
]

# LABELS = [    
#     "jalan_NE", "jalan_NW", "jalan_SE", "jalan_SW",
#     "menyapu_NE", "menyapu_NW", "menyapu_SE", "menyapu_SW",
#     "barang_NE", "barang_NW", "barang_SE", "barang_SW",
#     "diam_NE", "diam_NW", "diam_SE", "diam_SW"
# ] 

# DATASET_PATH = "data/HAR_pose_activities/database/"
# DATASET_PATH = "data/HAR_pose_activities/database/Training Default/"

# DATASET_PATH = "data/Overlap_fixed4_separated/"

DATASET_PATH = "data/DirectNO/Normalize/"

X_train_path = DATASET_PATH + "X_train.txt"
X_test_path = DATASET_PATH + "X_test.txt"
# X_test_path = "utilities/something/something.txt"

y_train_path = DATASET_PATH + "Y_train.txt"
y_test_path = DATASET_PATH + "Y_test.txt"

# n_steps = 32 # 32 timesteps per series
# n_steps = 1

# n_steps = 5
n_steps = 8
```


```python

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

X_train = load_X(X_train_path)
X_test = load_X(X_test_path)
#print X_test

y_train = load_y(y_train_path)
y_test = load_y(y_test_path)
# proof that it actually works for the skeptical: replace labelled classes with random classes to train on
#for i in range(len(y_train)):
#    y_train[i] = randint(0, 5)

```

## Set Parameters:



```python
# Input Data 

training_data_count = len(X_train)  # 4519 training series (with 50% overlap between each serie)
test_data_count = len(X_test)  # 1197 test series
n_input = len(X_train[0][0])  # num input parameters per timestep

n_hidden = 36 # Hidden layer num of features
# n_classes = 6 
n_classes = len(LABELS)

#updated for learning-rate decay
# calculated as: decayed_learning_rate = init_learning_rate * decay_rate ^ (global_step / decay_steps)
decaying_learning_rate = True
learning_rate = 0.0025 #used if decaying_learning_rate set to False

init_learning_rate = 0.005
# init_learning_rate = 0.00015

decay_rate = 0.96 #the base of the exponential in the decay
decay_steps = 100000 #used in decay every 60000 steps with a base of 0.96

global_step = tf.Variable(0, trainable=False)
lambda_loss_amount = 0.0015

# training_iters = training_data_count *300  # Loop 300 times on the dataset, ie 300 epochs
# training_iters = training_data_count *60
# training_iters = training_data_count *120
# training_iters = training_data_count *1
# batch_size = 5
batch_size = 64
# batch_size = 512
display_iter = batch_size*8  # To show test set accuracy during training

print("(X shape, y shape, every X's mean, every X's standard deviation)")
print(X_train.shape, y_test.shape, np.mean(X_test), np.std(X_test))
print("\nThe dataset has not been preprocessed, is not normalised etc")



```

    (X shape, y shape, every X's mean, every X's standard deviation)
    (1382, 8, 36) (2470, 1) -7.722059e-10 25.582153
    
    The dataset has not been preprocessed, is not normalised etc
    

## Utility functions for training:


```python
def LSTM_RNN(_X, _weights, _biases):
    # model architecture based on "guillaume-chevalier" and "aymericdamien" under the MIT license.

    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    _X = tf.reshape(_X, [-1, n_input])   
    # Rectifies Linear Unit activation function used
    _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, n_steps, 0) 

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
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


```

## Build the network:


```python
# for _ in range(3):
#     tf.reset_default_graph()
#     var = tf.Variable(0)
#     with tf.Session() as session:
#         session.run(tf.global_variables_initializer())
#         print(len(session.graph._nodes_by_name.keys()))
```


```python

# Graph input/output
x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

# Graph weights
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

pred = LSTM_RNN(x, weights, biases)

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


```

    WARNING:tensorflow:From <ipython-input-5-488cfd9da3d0>:12: BasicLSTMCell.__init__ (from tensorflow.python.ops.rnn_cell_impl) is deprecated and will be removed in a future version.
    Instructions for updating:
    This class is deprecated, please use tf.nn.rnn_cell.LSTMCell, which supports all the feature this cell currently has. Please replace the existing code with tf.nn.rnn_cell.LSTMCell(name='basic_lstm_cell').
    WARNING:tensorflow:From <ipython-input-7-982ce1458cc4>:22: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    
    Future major versions of TensorFlow will allow gradients to flow
    into the labels input on backprop by default.
    
    See `tf.nn.softmax_cross_entropy_with_logits_v2`.
    
    


```python
print(biases)
print(weights['hidden'])
```

    {'hidden': <tf.Variable 'Variable_3:0' shape=(36,) dtype=float32_ref>, 'out': <tf.Variable 'Variable_4:0' shape=(17,) dtype=float32_ref>}
    <tf.Variable 'Variable_1:0' shape=(36, 36) dtype=float32_ref>
    


```python
# if decaying_learning_rate:
#     learning_rate = tf.train.exponential_decay(init_learning_rate, global_step*batch_size, decay_steps, decay_rate, staircase=True)
```

## Train the network:


```python
test_losses = []
test_accuracies = []
train_losses = []
train_accuracies = []
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
init = tf.global_variables_initializer()

sess.run(init)
```


```python
# training_iters = training_data_count *120
# training_iters = training_data_count *5120
# training_iters = training_data_count *2560
# training_iters = training_data_count *1024
training_iters = training_data_count *2046
# training_iters = training_data_count *1024
# training_iters = training_data_count *5120

#create saver before training
# saver = tf.train.Saver()
saver = tf.train.Saver(var_list={'wh':weights['hidden'], 'wo':weights['out'], 'bh':biases['hidden'], 'bo':biases['out']})
load = False
train = True
update = True

#check if you want to retrain or import a saved model
if load:
    saver.restore(sess, DATASET_PATH + "model.ckpt")
    print("Model restored.")

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
```


```python
# Perform Training steps with "batch_size" amount of data at each loop. 
# Elements of each batch are chosen randomly, without replacement, from X_train, 
# restarting when remaining datapoints < batch_size
step = 1
time_start = time.time()
unsampled_indices = range(0,len(X_train))

if not train:
    
    # only perform testing - on training set
    loss, acc = sess.run(
        [cost, accuracy], 
        feed_dict={
            x: X_train,
            y: one_hot(y_train)
        }
    )
    
    print("PERFORMANCE ON TRAIN SET:             " + \
          "Batch Loss = {}".format(loss) + \
          ", Accuracy = {}".format(acc))
    
    # only perform testing - on test set
    loss, acc = sess.run(
        [cost, accuracy], 
        feed_dict={
            x: X_test,
            y: one_hot(y_test)
        }
    )
    
    print("PERFORMANCE ON TEST SET:             " + \
          "Batch Loss = {}".format(loss) + \
          ", Accuracy = {}".format(acc))

while train and (step * batch_size <= training_iters):
    #print (sess.run(learning_rate)) #decaying learning rate
    #print (sess.run(global_step)) # global number of iterations
    if len(unsampled_indices) < batch_size:
        unsampled_indices = range(0,len(X_train)) 
    batch_xs, raw_labels, unsampled_indicies = extract_batch_size(X_train, y_train, unsampled_indices, batch_size)
    batch_ys = one_hot(raw_labels)
    # check that encoded output is same length as num_classes, if not, pad it 
    if len(batch_ys[0]) < n_classes:
        temp_ys = np.zeros((batch_size, n_classes))
        temp_ys[:batch_ys.shape[0],:batch_ys.shape[1]] = batch_ys
        batch_ys = temp_ys
       
    # Fit training using batch data
    _, loss, acc = sess.run(
        [optimizer, cost, accuracy],
        feed_dict={
            x: batch_xs, 
            y: batch_ys
        }
    )
    train_losses.append(loss)
    train_accuracies.append(acc)
    
    # Evaluate network only at some steps for faster training: 
    if (step*batch_size % display_iter == 0) or (step == 1) or (step * batch_size > training_iters):
        
        # To not spam console, show training accuracy/loss in this "if"
        print("Iter #" + str(step*batch_size) + \
              ":  Learning rate = " + "{:.6f}".format(sess.run(learning_rate)) + \
              ":   Batch Loss = " + "{:.6f}".format(loss) + \
              ", Accuracy = {}".format(acc))
        
        # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
        loss, acc = sess.run(
            [cost, accuracy], 
            feed_dict={
                x: X_test,
                y: one_hot(y_test)
            }
        )
        test_losses.append(loss)
        test_accuracies.append(acc)
        print("PERFORMANCE ON TEST SET:             " + \
              "Batch Loss = {}".format(loss) + \
              ", Accuracy = {}".format(acc))

    step += 1

print("Optimization Finished!")

# Accuracy for test data

one_hot_predictions, accuracy_fin, final_loss = sess.run(
    [pred, accuracy, cost],
    feed_dict={
        x: X_test,
        y: one_hot(y_test)
    }
)

test_losses.append(final_loss)
test_accuracies.append(accuracy_fin)

print("FINAL RESULT: " + \
      "Batch Loss = {}".format(final_loss) + \
      ", Accuracy = {}".format(accuracy_fin))
time_stop = time.time()
print("TOTAL TIME:  {}".format(time_stop - time_start))
```

    Iter #64:  Learning rate = 0.005000:   Batch Loss = 6.251019, Accuracy = 0.03125
    PERFORMANCE ON TEST SET:             Batch Loss = 5.608214378356934, Accuracy = 0.10485830157995224
    Iter #512:  Learning rate = 0.005000:   Batch Loss = 4.608506, Accuracy = 0.25
    PERFORMANCE ON TEST SET:             Batch Loss = 4.487692832946777, Accuracy = 0.3085020184516907
    Iter #1024:  Learning rate = 0.005000:   Batch Loss = 4.212497, Accuracy = 0.3125
    PERFORMANCE ON TEST SET:             Batch Loss = 4.142374038696289, Accuracy = 0.3360323905944824
    Iter #1536:  Learning rate = 0.005000:   Batch Loss = 3.697629, Accuracy = 0.53125
    PERFORMANCE ON TEST SET:             Batch Loss = 3.9438347816467285, Accuracy = 0.41093116998672485
    Iter #2048:  Learning rate = 0.005000:   Batch Loss = 3.533481, Accuracy = 0.53125
    PERFORMANCE ON TEST SET:             Batch Loss = 3.75866436958313, Accuracy = 0.4663967490196228
    Iter #2560:  Learning rate = 0.005000:   Batch Loss = 3.390174, Accuracy = 0.5625
    PERFORMANCE ON TEST SET:             Batch Loss = 3.614309787750244, Accuracy = 0.47732794284820557
    Iter #3072:  Learning rate = 0.005000:   Batch Loss = 3.613254, Accuracy = 0.5
    PERFORMANCE ON TEST SET:             Batch Loss = 3.517122983932495, Accuracy = 0.5101214647293091
    Iter #3584:  Learning rate = 0.005000:   Batch Loss = 3.527060, Accuracy = 0.390625
    PERFORMANCE ON TEST SET:             Batch Loss = 3.4209682941436768, Accuracy = 0.535627543926239
    Iter #4096:  Learning rate = 0.005000:   Batch Loss = 3.334085, Accuracy = 0.546875
    PERFORMANCE ON TEST SET:             Batch Loss = 3.3547048568725586, Accuracy = 0.5481781363487244
    Iter #4608:  Learning rate = 0.005000:   Batch Loss = 3.040650, Accuracy = 0.625
    PERFORMANCE ON TEST SET:             Batch Loss = 3.3113465309143066, Accuracy = 0.5404858589172363
    Iter #5120:  Learning rate = 0.005000:   Batch Loss = 3.319960, Accuracy = 0.5625
    PERFORMANCE ON TEST SET:             Batch Loss = 3.249574661254883, Accuracy = 0.5469635725021362
    Iter #5632:  Learning rate = 0.005000:   Batch Loss = 2.886031, Accuracy = 0.703125
    PERFORMANCE ON TEST SET:             Batch Loss = 3.210524082183838, Accuracy = 0.5708501935005188
    Iter #6144:  Learning rate = 0.005000:   Batch Loss = 2.907953, Accuracy = 0.609375
    PERFORMANCE ON TEST SET:             Batch Loss = 3.1316349506378174, Accuracy = 0.5769230723381042
    Iter #6656:  Learning rate = 0.005000:   Batch Loss = 2.713437, Accuracy = 0.78125
    PERFORMANCE ON TEST SET:             Batch Loss = 3.0849199295043945, Accuracy = 0.5834007859230042
    Iter #7168:  Learning rate = 0.005000:   Batch Loss = 2.850178, Accuracy = 0.6875
    PERFORMANCE ON TEST SET:             Batch Loss = 3.1001386642456055, Accuracy = 0.5850202441215515
    Iter #7680:  Learning rate = 0.005000:   Batch Loss = 2.897235, Accuracy = 0.65625
    PERFORMANCE ON TEST SET:             Batch Loss = 3.060065269470215, Accuracy = 0.5935222506523132
    Iter #8192:  Learning rate = 0.005000:   Batch Loss = 2.996761, Accuracy = 0.625
    PERFORMANCE ON TEST SET:             Batch Loss = 3.040836811065674, Accuracy = 0.5829959511756897
    Iter #8704:  Learning rate = 0.005000:   Batch Loss = 2.673188, Accuracy = 0.71875
    PERFORMANCE ON TEST SET:             Batch Loss = 2.9698753356933594, Accuracy = 0.6068825721740723
    Iter #9216:  Learning rate = 0.005000:   Batch Loss = 2.565559, Accuracy = 0.75
    PERFORMANCE ON TEST SET:             Batch Loss = 2.9160594940185547, Accuracy = 0.6060729026794434
    Iter #9728:  Learning rate = 0.005000:   Batch Loss = 2.718278, Accuracy = 0.625
    PERFORMANCE ON TEST SET:             Batch Loss = 2.8431453704833984, Accuracy = 0.6259109377861023
    Iter #10240:  Learning rate = 0.005000:   Batch Loss = 2.614418, Accuracy = 0.671875
    PERFORMANCE ON TEST SET:             Batch Loss = 2.8836021423339844, Accuracy = 0.5947368144989014
    Iter #10752:  Learning rate = 0.005000:   Batch Loss = 2.434367, Accuracy = 0.734375
    PERFORMANCE ON TEST SET:             Batch Loss = 2.8533737659454346, Accuracy = 0.5983805656433105
    Iter #11264:  Learning rate = 0.005000:   Batch Loss = 2.557019, Accuracy = 0.703125
    PERFORMANCE ON TEST SET:             Batch Loss = 2.7934958934783936, Accuracy = 0.6259109377861023
    Iter #11776:  Learning rate = 0.005000:   Batch Loss = 2.752167, Accuracy = 0.671875
    PERFORMANCE ON TEST SET:             Batch Loss = 2.7899885177612305, Accuracy = 0.621052622795105
    Iter #12288:  Learning rate = 0.005000:   Batch Loss = 2.337287, Accuracy = 0.78125
    PERFORMANCE ON TEST SET:             Batch Loss = 2.7562456130981445, Accuracy = 0.626720666885376
    Iter #12800:  Learning rate = 0.005000:   Batch Loss = 2.523067, Accuracy = 0.703125
    PERFORMANCE ON TEST SET:             Batch Loss = 2.71077036857605, Accuracy = 0.6291497945785522
    Iter #13312:  Learning rate = 0.005000:   Batch Loss = 2.524009, Accuracy = 0.71875
    PERFORMANCE ON TEST SET:             Batch Loss = 2.7305641174316406, Accuracy = 0.6230769157409668
    Iter #13824:  Learning rate = 0.005000:   Batch Loss = 2.310101, Accuracy = 0.78125
    PERFORMANCE ON TEST SET:             Batch Loss = 2.7022953033447266, Accuracy = 0.6404858231544495
    Iter #14336:  Learning rate = 0.005000:   Batch Loss = 2.374018, Accuracy = 0.765625
    PERFORMANCE ON TEST SET:             Batch Loss = 2.630164623260498, Accuracy = 0.6396760940551758
    Iter #14848:  Learning rate = 0.005000:   Batch Loss = 2.141116, Accuracy = 0.796875
    PERFORMANCE ON TEST SET:             Batch Loss = 2.697237968444824, Accuracy = 0.6125506162643433
    Iter #15360:  Learning rate = 0.005000:   Batch Loss = 2.399039, Accuracy = 0.71875
    PERFORMANCE ON TEST SET:             Batch Loss = 2.6267426013946533, Accuracy = 0.6412955522537231
    Iter #15872:  Learning rate = 0.005000:   Batch Loss = 2.402875, Accuracy = 0.703125
    PERFORMANCE ON TEST SET:             Batch Loss = 2.607969284057617, Accuracy = 0.6453441381454468
    Iter #16384:  Learning rate = 0.005000:   Batch Loss = 2.361730, Accuracy = 0.671875
    PERFORMANCE ON TEST SET:             Batch Loss = 2.620114803314209, Accuracy = 0.6283400654792786
    Iter #16896:  Learning rate = 0.005000:   Batch Loss = 2.188533, Accuracy = 0.8125
    PERFORMANCE ON TEST SET:             Batch Loss = 2.546508550643921, Accuracy = 0.6554656028747559
    Iter #17408:  Learning rate = 0.005000:   Batch Loss = 2.273599, Accuracy = 0.734375
    PERFORMANCE ON TEST SET:             Batch Loss = 2.4738240242004395, Accuracy = 0.6906882524490356
    Iter #17920:  Learning rate = 0.005000:   Batch Loss = 2.239964, Accuracy = 0.734375
    PERFORMANCE ON TEST SET:             Batch Loss = 2.4926185607910156, Accuracy = 0.670040488243103
    Iter #18432:  Learning rate = 0.005000:   Batch Loss = 2.475393, Accuracy = 0.640625
    PERFORMANCE ON TEST SET:             Batch Loss = 2.487919569015503, Accuracy = 0.6587044596672058
    Iter #18944:  Learning rate = 0.005000:   Batch Loss = 2.296917, Accuracy = 0.609375
    PERFORMANCE ON TEST SET:             Batch Loss = 2.4361770153045654, Accuracy = 0.6631578803062439
    Iter #19456:  Learning rate = 0.005000:   Batch Loss = 2.044596, Accuracy = 0.859375
    PERFORMANCE ON TEST SET:             Batch Loss = 2.4030470848083496, Accuracy = 0.6854251027107239
    Iter #19968:  Learning rate = 0.005000:   Batch Loss = 2.070111, Accuracy = 0.796875
    PERFORMANCE ON TEST SET:             Batch Loss = 2.4076039791107178, Accuracy = 0.6821862459182739
    Iter #20480:  Learning rate = 0.005000:   Batch Loss = 1.920730, Accuracy = 0.859375
    PERFORMANCE ON TEST SET:             Batch Loss = 2.348842144012451, Accuracy = 0.6947368383407593
    Iter #20992:  Learning rate = 0.005000:   Batch Loss = 2.170816, Accuracy = 0.75
    PERFORMANCE ON TEST SET:             Batch Loss = 2.302466869354248, Accuracy = 0.6951416730880737
    Iter #21504:  Learning rate = 0.005000:   Batch Loss = 2.085415, Accuracy = 0.734375
    PERFORMANCE ON TEST SET:             Batch Loss = 2.333615779876709, Accuracy = 0.6850202679634094
    Iter #22016:  Learning rate = 0.005000:   Batch Loss = 2.045399, Accuracy = 0.828125
    PERFORMANCE ON TEST SET:             Batch Loss = 2.383655548095703, Accuracy = 0.6736842393875122
    Iter #22528:  Learning rate = 0.005000:   Batch Loss = 1.992653, Accuracy = 0.78125
    PERFORMANCE ON TEST SET:             Batch Loss = 2.395308256149292, Accuracy = 0.6655870676040649
    Iter #23040:  Learning rate = 0.005000:   Batch Loss = 2.000734, Accuracy = 0.734375
    PERFORMANCE ON TEST SET:             Batch Loss = 2.3472838401794434, Accuracy = 0.6748988032341003
    Iter #23552:  Learning rate = 0.005000:   Batch Loss = 2.134220, Accuracy = 0.75
    PERFORMANCE ON TEST SET:             Batch Loss = 2.408886432647705, Accuracy = 0.6441295742988586
    Iter #24064:  Learning rate = 0.005000:   Batch Loss = 1.958626, Accuracy = 0.765625
    PERFORMANCE ON TEST SET:             Batch Loss = 2.316964864730835, Accuracy = 0.6785424947738647
    Iter #24576:  Learning rate = 0.005000:   Batch Loss = 2.011322, Accuracy = 0.734375
    PERFORMANCE ON TEST SET:             Batch Loss = 2.344590663909912, Accuracy = 0.670040488243103
    Iter #25088:  Learning rate = 0.005000:   Batch Loss = 1.927314, Accuracy = 0.796875
    PERFORMANCE ON TEST SET:             Batch Loss = 2.337568759918213, Accuracy = 0.6744939088821411
    Iter #25600:  Learning rate = 0.005000:   Batch Loss = 1.965910, Accuracy = 0.734375
    PERFORMANCE ON TEST SET:             Batch Loss = 2.2424018383026123, Accuracy = 0.6842105388641357
    Iter #26112:  Learning rate = 0.005000:   Batch Loss = 1.808360, Accuracy = 0.828125
    PERFORMANCE ON TEST SET:             Batch Loss = 2.2601215839385986, Accuracy = 0.6914979815483093
    Iter #26624:  Learning rate = 0.005000:   Batch Loss = 2.069788, Accuracy = 0.78125
    PERFORMANCE ON TEST SET:             Batch Loss = 2.1777377128601074, Accuracy = 0.7141700387001038
    Iter #27136:  Learning rate = 0.005000:   Batch Loss = 1.826614, Accuracy = 0.78125
    PERFORMANCE ON TEST SET:             Batch Loss = 2.1959409713745117, Accuracy = 0.6959514021873474
    Iter #27648:  Learning rate = 0.005000:   Batch Loss = 1.957487, Accuracy = 0.734375
    PERFORMANCE ON TEST SET:             Batch Loss = 2.129697322845459, Accuracy = 0.7182186245918274
    Iter #28160:  Learning rate = 0.005000:   Batch Loss = 1.815660, Accuracy = 0.828125
    PERFORMANCE ON TEST SET:             Batch Loss = 2.1150765419006348, Accuracy = 0.7242915034294128
    Iter #28672:  Learning rate = 0.005000:   Batch Loss = 1.791250, Accuracy = 0.8125
    PERFORMANCE ON TEST SET:             Batch Loss = 2.152894973754883, Accuracy = 0.6975708603858948
    Iter #29184:  Learning rate = 0.005000:   Batch Loss = 1.637786, Accuracy = 0.828125
    PERFORMANCE ON TEST SET:             Batch Loss = 2.079009532928467, Accuracy = 0.7149797677993774
    Iter #29696:  Learning rate = 0.005000:   Batch Loss = 1.732195, Accuracy = 0.859375
    PERFORMANCE ON TEST SET:             Batch Loss = 2.1277856826782227, Accuracy = 0.7121457457542419
    Iter #30208:  Learning rate = 0.005000:   Batch Loss = 1.633213, Accuracy = 0.875
    PERFORMANCE ON TEST SET:             Batch Loss = 2.0853238105773926, Accuracy = 0.7251012325286865
    Iter #30720:  Learning rate = 0.005000:   Batch Loss = 1.670691, Accuracy = 0.875
    PERFORMANCE ON TEST SET:             Batch Loss = 2.1265227794647217, Accuracy = 0.7109311819076538
    Iter #31232:  Learning rate = 0.005000:   Batch Loss = 1.750490, Accuracy = 0.78125
    PERFORMANCE ON TEST SET:             Batch Loss = 2.089581251144409, Accuracy = 0.7085019946098328
    Iter #31744:  Learning rate = 0.005000:   Batch Loss = 1.665339, Accuracy = 0.859375
    PERFORMANCE ON TEST SET:             Batch Loss = 2.069221019744873, Accuracy = 0.7125505805015564
    Iter #32256:  Learning rate = 0.005000:   Batch Loss = 1.567568, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 2.0232291221618652, Accuracy = 0.7259109020233154
    Iter #32768:  Learning rate = 0.005000:   Batch Loss = 1.583673, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 2.0123543739318848, Accuracy = 0.7368420958518982
    Iter #33280:  Learning rate = 0.005000:   Batch Loss = 1.768758, Accuracy = 0.8125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.9910447597503662, Accuracy = 0.7336032390594482
    Iter #33792:  Learning rate = 0.005000:   Batch Loss = 1.724530, Accuracy = 0.78125
    PERFORMANCE ON TEST SET:             Batch Loss = 2.0754659175872803, Accuracy = 0.7093117237091064
    Iter #34304:  Learning rate = 0.005000:   Batch Loss = 1.625848, Accuracy = 0.8125
    PERFORMANCE ON TEST SET:             Batch Loss = 2.082444190979004, Accuracy = 0.708906888961792
    Iter #34816:  Learning rate = 0.005000:   Batch Loss = 1.767418, Accuracy = 0.84375
    PERFORMANCE ON TEST SET:             Batch Loss = 2.0623655319213867, Accuracy = 0.7016194462776184
    Iter #35328:  Learning rate = 0.005000:   Batch Loss = 1.693321, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 2.0334534645080566, Accuracy = 0.7182186245918274
    Iter #35840:  Learning rate = 0.005000:   Batch Loss = 1.714188, Accuracy = 0.859375
    PERFORMANCE ON TEST SET:             Batch Loss = 2.027407169342041, Accuracy = 0.7178137898445129
    Iter #36352:  Learning rate = 0.005000:   Batch Loss = 1.594664, Accuracy = 0.84375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.9756951332092285, Accuracy = 0.7267206311225891
    Iter #36864:  Learning rate = 0.005000:   Batch Loss = 1.600424, Accuracy = 0.875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.9597644805908203, Accuracy = 0.7340080738067627
    Iter #37376:  Learning rate = 0.005000:   Batch Loss = 1.482919, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 2.011580467224121, Accuracy = 0.7137652039527893
    Iter #37888:  Learning rate = 0.005000:   Batch Loss = 1.743419, Accuracy = 0.796875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.966151237487793, Accuracy = 0.7287449240684509
    Iter #38400:  Learning rate = 0.005000:   Batch Loss = 1.552003, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.964311122894287, Accuracy = 0.7271255254745483
    Iter #38912:  Learning rate = 0.005000:   Batch Loss = 1.718012, Accuracy = 0.78125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.9536082744598389, Accuracy = 0.7170040607452393
    Iter #39424:  Learning rate = 0.005000:   Batch Loss = 1.498148, Accuracy = 0.875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.9510552883148193, Accuracy = 0.7206477522850037
    Iter #39936:  Learning rate = 0.005000:   Batch Loss = 1.509103, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.9151285886764526, Accuracy = 0.7263157963752747
    Iter #40448:  Learning rate = 0.005000:   Batch Loss = 1.499906, Accuracy = 0.875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.9156537055969238, Accuracy = 0.7234817743301392
    Iter #40960:  Learning rate = 0.005000:   Batch Loss = 1.591453, Accuracy = 0.796875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.8929089307785034, Accuracy = 0.7356275320053101
    Iter #41472:  Learning rate = 0.005000:   Batch Loss = 1.391900, Accuracy = 0.859375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.9279353618621826, Accuracy = 0.7230769395828247
    Iter #41984:  Learning rate = 0.005000:   Batch Loss = 1.505633, Accuracy = 0.859375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.926896095275879, Accuracy = 0.7214574813842773
    Iter #42496:  Learning rate = 0.005000:   Batch Loss = 1.412186, Accuracy = 0.875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.8751389980316162, Accuracy = 0.7315789461135864
    Iter #43008:  Learning rate = 0.005000:   Batch Loss = 1.512258, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.8669962882995605, Accuracy = 0.7392712831497192
    Iter #43520:  Learning rate = 0.005000:   Batch Loss = 1.419556, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.8528876304626465, Accuracy = 0.7465587258338928
    Iter #44032:  Learning rate = 0.005000:   Batch Loss = 1.439718, Accuracy = 0.875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.8350180387496948, Accuracy = 0.7493926882743835
    Iter #44544:  Learning rate = 0.005000:   Batch Loss = 1.594284, Accuracy = 0.828125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.8404055833816528, Accuracy = 0.7396761178970337
    Iter #45056:  Learning rate = 0.005000:   Batch Loss = 1.518275, Accuracy = 0.765625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.914432406425476, Accuracy = 0.7113360166549683
    Iter #45568:  Learning rate = 0.005000:   Batch Loss = 1.343183, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.9157261848449707, Accuracy = 0.7153846025466919
    Iter #46080:  Learning rate = 0.005000:   Batch Loss = 1.298690, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.893249273300171, Accuracy = 0.7315789461135864
    Iter #46592:  Learning rate = 0.005000:   Batch Loss = 1.488590, Accuracy = 0.84375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.8593230247497559, Accuracy = 0.7372469902038574
    Iter #47104:  Learning rate = 0.005000:   Batch Loss = 1.411320, Accuracy = 0.859375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.9259059429168701, Accuracy = 0.7194331884384155
    Iter #47616:  Learning rate = 0.005000:   Batch Loss = 1.574947, Accuracy = 0.8125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.872042179107666, Accuracy = 0.7238866686820984
    Iter #48128:  Learning rate = 0.005000:   Batch Loss = 1.515374, Accuracy = 0.875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.8799481391906738, Accuracy = 0.7190283536911011
    Iter #48640:  Learning rate = 0.005000:   Batch Loss = 1.351109, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.798417568206787, Accuracy = 0.7489878535270691
    Iter #49152:  Learning rate = 0.005000:   Batch Loss = 1.508841, Accuracy = 0.765625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.8329895734786987, Accuracy = 0.7384615540504456
    Iter #49664:  Learning rate = 0.005000:   Batch Loss = 1.446384, Accuracy = 0.84375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.7683911323547363, Accuracy = 0.7538461685180664
    Iter #50176:  Learning rate = 0.005000:   Batch Loss = 1.376636, Accuracy = 0.875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.7620315551757812, Accuracy = 0.7449392676353455
    Iter #50688:  Learning rate = 0.005000:   Batch Loss = 1.298896, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.754198670387268, Accuracy = 0.7493926882743835
    Iter #51200:  Learning rate = 0.005000:   Batch Loss = 1.293454, Accuracy = 0.875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.7177547216415405, Accuracy = 0.7740890979766846
    Iter #51712:  Learning rate = 0.005000:   Batch Loss = 1.268494, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.737529993057251, Accuracy = 0.75789475440979
    Iter #52224:  Learning rate = 0.005000:   Batch Loss = 1.221731, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.6761701107025146, Accuracy = 0.7801619172096252
    Iter #52736:  Learning rate = 0.005000:   Batch Loss = 1.304815, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.7521203756332397, Accuracy = 0.7534412741661072
    Iter #53248:  Learning rate = 0.005000:   Batch Loss = 1.161985, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.7591216564178467, Accuracy = 0.7562752962112427
    Iter #53760:  Learning rate = 0.005000:   Batch Loss = 1.453978, Accuracy = 0.84375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.725250005722046, Accuracy = 0.7631579041481018
    Iter #54272:  Learning rate = 0.005000:   Batch Loss = 1.347628, Accuracy = 0.859375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.769383430480957, Accuracy = 0.731174111366272
    Iter #54784:  Learning rate = 0.005000:   Batch Loss = 1.289227, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.71048903465271, Accuracy = 0.7603238821029663
    Iter #55296:  Learning rate = 0.005000:   Batch Loss = 1.168944, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.7143888473510742, Accuracy = 0.7595141530036926
    Iter #55808:  Learning rate = 0.005000:   Batch Loss = 1.230439, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.7113733291625977, Accuracy = 0.7538461685180664
    Iter #56320:  Learning rate = 0.005000:   Batch Loss = 1.215657, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.786629557609558, Accuracy = 0.7429149746894836
    Iter #56832:  Learning rate = 0.005000:   Batch Loss = 1.246898, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.6600379943847656, Accuracy = 0.7732793688774109
    Iter #57344:  Learning rate = 0.005000:   Batch Loss = 1.179720, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.6485958099365234, Accuracy = 0.7789473533630371
    Iter #57856:  Learning rate = 0.005000:   Batch Loss = 1.166635, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.6558480262756348, Accuracy = 0.7704453468322754
    Iter #58368:  Learning rate = 0.005000:   Batch Loss = 1.259703, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.617664098739624, Accuracy = 0.7765182256698608
    Iter #58880:  Learning rate = 0.005000:   Batch Loss = 1.164334, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.6947197914123535, Accuracy = 0.7631579041481018
    Iter #59392:  Learning rate = 0.005000:   Batch Loss = 1.173921, Accuracy = 0.875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.6298067569732666, Accuracy = 0.7765182256698608
    Iter #59904:  Learning rate = 0.005000:   Batch Loss = 1.253698, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.6359889507293701, Accuracy = 0.7712550759315491
    Iter #60416:  Learning rate = 0.005000:   Batch Loss = 1.226364, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.723980188369751, Accuracy = 0.7364372611045837
    Iter #60928:  Learning rate = 0.005000:   Batch Loss = 1.461789, Accuracy = 0.828125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.677239179611206, Accuracy = 0.7518218755722046
    Iter #61440:  Learning rate = 0.005000:   Batch Loss = 1.226671, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.781118392944336, Accuracy = 0.7331984043121338
    Iter #61952:  Learning rate = 0.005000:   Batch Loss = 1.253704, Accuracy = 0.84375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.722804307937622, Accuracy = 0.7384615540504456
    Iter #62464:  Learning rate = 0.005000:   Batch Loss = 1.468555, Accuracy = 0.796875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.7030234336853027, Accuracy = 0.747773289680481
    Iter #62976:  Learning rate = 0.005000:   Batch Loss = 1.060437, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.6372002363204956, Accuracy = 0.7663967609405518
    Iter #63488:  Learning rate = 0.005000:   Batch Loss = 1.188170, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.6468515396118164, Accuracy = 0.7599190473556519
    Iter #64000:  Learning rate = 0.005000:   Batch Loss = 1.235713, Accuracy = 0.84375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.638444185256958, Accuracy = 0.7506073117256165
    Iter #64512:  Learning rate = 0.005000:   Batch Loss = 1.242155, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.612955093383789, Accuracy = 0.755465567111969
    Iter #65024:  Learning rate = 0.005000:   Batch Loss = 1.186975, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.6228394508361816, Accuracy = 0.7643724679946899
    Iter #65536:  Learning rate = 0.005000:   Batch Loss = 1.351151, Accuracy = 0.859375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.7087427377700806, Accuracy = 0.7340080738067627
    Iter #66048:  Learning rate = 0.005000:   Batch Loss = 1.219650, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.664814829826355, Accuracy = 0.747773289680481
    Iter #66560:  Learning rate = 0.005000:   Batch Loss = 1.014749, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.6147689819335938, Accuracy = 0.7615384459495544
    Iter #67072:  Learning rate = 0.005000:   Batch Loss = 1.024971, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.588569164276123, Accuracy = 0.7676113247871399
    Iter #67584:  Learning rate = 0.005000:   Batch Loss = 1.075047, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.5731302499771118, Accuracy = 0.7728744745254517
    Iter #68096:  Learning rate = 0.005000:   Batch Loss = 1.137336, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.563478946685791, Accuracy = 0.7684210538864136
    Iter #68608:  Learning rate = 0.005000:   Batch Loss = 1.126282, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.5601282119750977, Accuracy = 0.7817813754081726
    Iter #69120:  Learning rate = 0.005000:   Batch Loss = 1.113791, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.5635485649108887, Accuracy = 0.7765182256698608
    Iter #69632:  Learning rate = 0.005000:   Batch Loss = 1.141466, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.6201591491699219, Accuracy = 0.755465567111969
    Iter #70144:  Learning rate = 0.005000:   Batch Loss = 1.152397, Accuracy = 0.875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.609195351600647, Accuracy = 0.7643724679946899
    Iter #70656:  Learning rate = 0.005000:   Batch Loss = 1.156065, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.590993881225586, Accuracy = 0.7724696397781372
    Iter #71168:  Learning rate = 0.005000:   Batch Loss = 1.160572, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.5642640590667725, Accuracy = 0.7842105031013489
    Iter #71680:  Learning rate = 0.005000:   Batch Loss = 1.121613, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.5792391300201416, Accuracy = 0.7781376242637634
    Iter #72192:  Learning rate = 0.005000:   Batch Loss = 1.071214, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.5893107652664185, Accuracy = 0.7712550759315491
    Iter #72704:  Learning rate = 0.005000:   Batch Loss = 1.043845, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.5432472229003906, Accuracy = 0.7838056683540344
    Iter #73216:  Learning rate = 0.005000:   Batch Loss = 1.048110, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.5832154750823975, Accuracy = 0.7728744745254517
    Iter #73728:  Learning rate = 0.005000:   Batch Loss = 1.059497, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.55588698387146, Accuracy = 0.7813765406608582
    Iter #74240:  Learning rate = 0.005000:   Batch Loss = 1.068795, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.5478050708770752, Accuracy = 0.7765182256698608
    Iter #74752:  Learning rate = 0.005000:   Batch Loss = 1.071987, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.5360212326049805, Accuracy = 0.7829959392547607
    Iter #75264:  Learning rate = 0.005000:   Batch Loss = 1.031150, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.5278242826461792, Accuracy = 0.7821862101554871
    Iter #75776:  Learning rate = 0.005000:   Batch Loss = 1.031858, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.4919657707214355, Accuracy = 0.7886639833450317
    Iter #76288:  Learning rate = 0.005000:   Batch Loss = 0.990217, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.4565554857254028, Accuracy = 0.8068826198577881
    Iter #76800:  Learning rate = 0.005000:   Batch Loss = 0.946776, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.4582533836364746, Accuracy = 0.8097165822982788
    Iter #77312:  Learning rate = 0.005000:   Batch Loss = 0.959859, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.4433021545410156, Accuracy = 0.8097165822982788
    Iter #77824:  Learning rate = 0.005000:   Batch Loss = 0.974151, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.4500788450241089, Accuracy = 0.7991902828216553
    Iter #78336:  Learning rate = 0.005000:   Batch Loss = 0.926965, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.4694056510925293, Accuracy = 0.794331967830658
    Iter #78848:  Learning rate = 0.005000:   Batch Loss = 0.980852, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.4510509967803955, Accuracy = 0.7963562607765198
    Iter #79360:  Learning rate = 0.005000:   Batch Loss = 1.016874, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.479796290397644, Accuracy = 0.7959514260292053
    Iter #79872:  Learning rate = 0.005000:   Batch Loss = 0.941113, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.499219536781311, Accuracy = 0.7890688180923462
    Iter #80384:  Learning rate = 0.005000:   Batch Loss = 1.001587, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.4882609844207764, Accuracy = 0.7959514260292053
    Iter #80896:  Learning rate = 0.005000:   Batch Loss = 0.978468, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.5086588859558105, Accuracy = 0.7680162191390991
    Iter #81408:  Learning rate = 0.005000:   Batch Loss = 1.009310, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.4871982336044312, Accuracy = 0.7728744745254517
    Iter #81920:  Learning rate = 0.005000:   Batch Loss = 1.052764, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.4822005033493042, Accuracy = 0.7894737124443054
    Iter #82432:  Learning rate = 0.005000:   Batch Loss = 0.987093, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.4674568176269531, Accuracy = 0.7850202322006226
    Iter #82944:  Learning rate = 0.005000:   Batch Loss = 1.083465, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.459859013557434, Accuracy = 0.7898785471916199
    Iter #83456:  Learning rate = 0.005000:   Batch Loss = 0.921369, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.4699950218200684, Accuracy = 0.7838056683540344
    Iter #83968:  Learning rate = 0.005000:   Batch Loss = 1.013276, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.4407610893249512, Accuracy = 0.7959514260292053
    Iter #84480:  Learning rate = 0.005000:   Batch Loss = 0.979225, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.4106370210647583, Accuracy = 0.8056679964065552
    Iter #84992:  Learning rate = 0.005000:   Batch Loss = 0.982201, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.437579870223999, Accuracy = 0.7939271330833435
    Iter #85504:  Learning rate = 0.005000:   Batch Loss = 1.003493, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.441206693649292, Accuracy = 0.7987854480743408
    Iter #86016:  Learning rate = 0.005000:   Batch Loss = 0.960061, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.4846007823944092, Accuracy = 0.7813765406608582
    Iter #86528:  Learning rate = 0.005000:   Batch Loss = 0.905483, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.4840576648712158, Accuracy = 0.7850202322006226
    Iter #87040:  Learning rate = 0.005000:   Batch Loss = 0.900224, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.4459234476089478, Accuracy = 0.7781376242637634
    Iter #87552:  Learning rate = 0.005000:   Batch Loss = 0.957634, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.4247369766235352, Accuracy = 0.7902833819389343
    Iter #88064:  Learning rate = 0.005000:   Batch Loss = 0.905760, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.4627275466918945, Accuracy = 0.7890688180923462
    Iter #88576:  Learning rate = 0.005000:   Batch Loss = 0.991730, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.439756155014038, Accuracy = 0.7947368621826172
    Iter #89088:  Learning rate = 0.005000:   Batch Loss = 0.821199, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3987045288085938, Accuracy = 0.8012145757675171
    Iter #89600:  Learning rate = 0.005000:   Batch Loss = 0.919080, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3979706764221191, Accuracy = 0.7959514260292053
    Iter #90112:  Learning rate = 0.005000:   Batch Loss = 0.890143, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.4055083990097046, Accuracy = 0.7963562607765198
    Iter #90624:  Learning rate = 0.005000:   Batch Loss = 1.031693, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.4432669878005981, Accuracy = 0.7789473533630371
    Iter #91136:  Learning rate = 0.005000:   Batch Loss = 1.036178, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.435713768005371, Accuracy = 0.7838056683540344
    Iter #91648:  Learning rate = 0.005000:   Batch Loss = 0.875344, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.4092825651168823, Accuracy = 0.7963562607765198
    Iter #92160:  Learning rate = 0.005000:   Batch Loss = 0.864242, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.4196640253067017, Accuracy = 0.7874494194984436
    Iter #92672:  Learning rate = 0.005000:   Batch Loss = 0.948688, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.4527225494384766, Accuracy = 0.7858299612998962
    Iter #93184:  Learning rate = 0.005000:   Batch Loss = 0.888716, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.4688496589660645, Accuracy = 0.7704453468322754
    Iter #93696:  Learning rate = 0.005000:   Batch Loss = 0.900886, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.4225736856460571, Accuracy = 0.7708501815795898
    Iter #94208:  Learning rate = 0.005000:   Batch Loss = 1.020456, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.4269115924835205, Accuracy = 0.7704453468322754
    Iter #94720:  Learning rate = 0.005000:   Batch Loss = 0.911665, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.4073439836502075, Accuracy = 0.7866396903991699
    Iter #95232:  Learning rate = 0.005000:   Batch Loss = 0.832390, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3592885732650757, Accuracy = 0.8020243048667908
    Iter #95744:  Learning rate = 0.005000:   Batch Loss = 0.863103, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3945956230163574, Accuracy = 0.7935222387313843
    Iter #96256:  Learning rate = 0.005000:   Batch Loss = 0.866413, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3888871669769287, Accuracy = 0.7902833819389343
    Iter #96768:  Learning rate = 0.005000:   Batch Loss = 0.959062, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.4139113426208496, Accuracy = 0.7809716463088989
    Iter #97280:  Learning rate = 0.005000:   Batch Loss = 0.997495, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.386009931564331, Accuracy = 0.7874494194984436
    Iter #97792:  Learning rate = 0.005000:   Batch Loss = 0.898431, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.4338808059692383, Accuracy = 0.7821862101554871
    Iter #98304:  Learning rate = 0.005000:   Batch Loss = 0.849125, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.4326293468475342, Accuracy = 0.768825888633728
    Iter #98816:  Learning rate = 0.005000:   Batch Loss = 0.912386, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3648571968078613, Accuracy = 0.8008097410202026
    Iter #99328:  Learning rate = 0.005000:   Batch Loss = 0.898971, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.38651704788208, Accuracy = 0.7894737124443054
    Iter #99840:  Learning rate = 0.005000:   Batch Loss = 0.867543, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3660023212432861, Accuracy = 0.7931174039840698
    Iter #100352:  Learning rate = 0.004800:   Batch Loss = 0.804209, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3946313858032227, Accuracy = 0.7906882762908936
    Iter #100864:  Learning rate = 0.004800:   Batch Loss = 0.895939, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3949427604675293, Accuracy = 0.7838056683540344
    Iter #101376:  Learning rate = 0.004800:   Batch Loss = 0.858609, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.4283878803253174, Accuracy = 0.7769230604171753
    Iter #101888:  Learning rate = 0.004800:   Batch Loss = 0.857774, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.4119148254394531, Accuracy = 0.7829959392547607
    Iter #102400:  Learning rate = 0.004800:   Batch Loss = 0.828319, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.4353270530700684, Accuracy = 0.7781376242637634
    Iter #102912:  Learning rate = 0.004800:   Batch Loss = 0.998926, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.4130072593688965, Accuracy = 0.7846153974533081
    Iter #103424:  Learning rate = 0.004800:   Batch Loss = 0.840822, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.4569242000579834, Accuracy = 0.7684210538864136
    Iter #103936:  Learning rate = 0.004800:   Batch Loss = 0.996889, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3836784362792969, Accuracy = 0.7825911045074463
    Iter #104448:  Learning rate = 0.004800:   Batch Loss = 0.883602, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3578808307647705, Accuracy = 0.7975708246231079
    Iter #104960:  Learning rate = 0.004800:   Batch Loss = 0.898738, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.361607551574707, Accuracy = 0.7874494194984436
    Iter #105472:  Learning rate = 0.004800:   Batch Loss = 0.858944, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3458690643310547, Accuracy = 0.7995951175689697
    Iter #105984:  Learning rate = 0.004800:   Batch Loss = 0.823462, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3757734298706055, Accuracy = 0.7850202322006226
    Iter #106496:  Learning rate = 0.004800:   Batch Loss = 0.891843, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.382049322128296, Accuracy = 0.7886639833450317
    Iter #107008:  Learning rate = 0.004800:   Batch Loss = 0.851230, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3221919536590576, Accuracy = 0.8056679964065552
    Iter #107520:  Learning rate = 0.004800:   Batch Loss = 0.909684, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3757538795471191, Accuracy = 0.7882590889930725
    Iter #108032:  Learning rate = 0.004800:   Batch Loss = 0.991251, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3870030641555786, Accuracy = 0.7765182256698608
    Iter #108544:  Learning rate = 0.004800:   Batch Loss = 0.786043, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3241910934448242, Accuracy = 0.7975708246231079
    Iter #109056:  Learning rate = 0.004800:   Batch Loss = 0.869008, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3224713802337646, Accuracy = 0.8012145757675171
    Iter #109568:  Learning rate = 0.004800:   Batch Loss = 0.836764, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3401837348937988, Accuracy = 0.7906882762908936
    Iter #110080:  Learning rate = 0.004800:   Batch Loss = 0.894057, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3331756591796875, Accuracy = 0.7995951175689697
    Iter #110592:  Learning rate = 0.004800:   Batch Loss = 0.752577, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3025161027908325, Accuracy = 0.8121457695960999
    Iter #111104:  Learning rate = 0.004800:   Batch Loss = 0.736806, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.288109302520752, Accuracy = 0.8097165822982788
    Iter #111616:  Learning rate = 0.004800:   Batch Loss = 0.736784, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3023626804351807, Accuracy = 0.8113360404968262
    Iter #112128:  Learning rate = 0.004800:   Batch Loss = 0.720478, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2934167385101318, Accuracy = 0.8032388687133789
    Iter #112640:  Learning rate = 0.004800:   Batch Loss = 0.839004, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3536708354949951, Accuracy = 0.8020243048667908
    Iter #113152:  Learning rate = 0.004800:   Batch Loss = 0.761515, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3186607360839844, Accuracy = 0.7975708246231079
    Iter #113664:  Learning rate = 0.004800:   Batch Loss = 0.734167, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3050212860107422, Accuracy = 0.7955465316772461
    Iter #114176:  Learning rate = 0.004800:   Batch Loss = 0.821917, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2937265634536743, Accuracy = 0.8024291396141052
    Iter #114688:  Learning rate = 0.004800:   Batch Loss = 0.829012, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.292251706123352, Accuracy = 0.8040485978126526
    Iter #115200:  Learning rate = 0.004800:   Batch Loss = 0.787710, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3026236295700073, Accuracy = 0.8060728907585144
    Iter #115712:  Learning rate = 0.004800:   Batch Loss = 0.793959, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2965552806854248, Accuracy = 0.8040485978126526
    Iter #116224:  Learning rate = 0.004800:   Batch Loss = 0.813880, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2760841846466064, Accuracy = 0.8097165822982788
    Iter #116736:  Learning rate = 0.004800:   Batch Loss = 0.750703, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3120315074920654, Accuracy = 0.7995951175689697
    Iter #117248:  Learning rate = 0.004800:   Batch Loss = 0.721907, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2821359634399414, Accuracy = 0.8060728907585144
    Iter #117760:  Learning rate = 0.004800:   Batch Loss = 0.763807, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2662043571472168, Accuracy = 0.8165991902351379
    Iter #118272:  Learning rate = 0.004800:   Batch Loss = 0.747216, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2793773412704468, Accuracy = 0.8080971837043762
    Iter #118784:  Learning rate = 0.004800:   Batch Loss = 0.825872, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2635509967803955, Accuracy = 0.8121457695960999
    Iter #119296:  Learning rate = 0.004800:   Batch Loss = 0.708842, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2672157287597656, Accuracy = 0.8040485978126526
    Iter #119808:  Learning rate = 0.004800:   Batch Loss = 0.753957, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2683919668197632, Accuracy = 0.8008097410202026
    Iter #120320:  Learning rate = 0.004800:   Batch Loss = 0.751633, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2644476890563965, Accuracy = 0.8145748972892761
    Iter #120832:  Learning rate = 0.004800:   Batch Loss = 0.750522, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2583374977111816, Accuracy = 0.8121457695960999
    Iter #121344:  Learning rate = 0.004800:   Batch Loss = 0.764274, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.264624834060669, Accuracy = 0.8218623399734497
    Iter #121856:  Learning rate = 0.004800:   Batch Loss = 0.826243, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2183347940444946, Accuracy = 0.8279352188110352
    Iter #122368:  Learning rate = 0.004800:   Batch Loss = 0.753941, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2150051593780518, Accuracy = 0.8226720690727234
    Iter #122880:  Learning rate = 0.004800:   Batch Loss = 0.677420, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.21708083152771, Accuracy = 0.829959511756897
    Iter #123392:  Learning rate = 0.004800:   Batch Loss = 0.701825, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2273130416870117, Accuracy = 0.8149797320365906
    Iter #123904:  Learning rate = 0.004800:   Batch Loss = 0.692861, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2402257919311523, Accuracy = 0.8080971837043762
    Iter #124416:  Learning rate = 0.004800:   Batch Loss = 0.712393, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2176347970962524, Accuracy = 0.8186234831809998
    Iter #124928:  Learning rate = 0.004800:   Batch Loss = 0.717977, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2380599975585938, Accuracy = 0.8109311461448669
    Iter #125440:  Learning rate = 0.004800:   Batch Loss = 0.759124, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2248013019561768, Accuracy = 0.8234817981719971
    Iter #125952:  Learning rate = 0.004800:   Batch Loss = 0.686238, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1862895488739014, Accuracy = 0.8291497826576233
    Iter #126464:  Learning rate = 0.004800:   Batch Loss = 0.725759, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.19230055809021, Accuracy = 0.8319838047027588
    Iter #126976:  Learning rate = 0.004800:   Batch Loss = 0.698443, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2377755641937256, Accuracy = 0.8161943554878235
    Iter #127488:  Learning rate = 0.004800:   Batch Loss = 0.875307, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2134177684783936, Accuracy = 0.8234817981719971
    Iter #128000:  Learning rate = 0.004800:   Batch Loss = 0.763877, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2775039672851562, Accuracy = 0.804453432559967
    Iter #128512:  Learning rate = 0.004800:   Batch Loss = 0.736815, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.273572564125061, Accuracy = 0.8048583269119263
    Iter #129024:  Learning rate = 0.004800:   Batch Loss = 0.707614, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2620179653167725, Accuracy = 0.7971659898757935
    Iter #129536:  Learning rate = 0.004800:   Batch Loss = 0.707472, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.276855707168579, Accuracy = 0.8020243048667908
    Iter #130048:  Learning rate = 0.004800:   Batch Loss = 0.728762, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2010538578033447, Accuracy = 0.8226720690727234
    Iter #130560:  Learning rate = 0.004800:   Batch Loss = 0.699974, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2210232019424438, Accuracy = 0.8157894611358643
    Iter #131072:  Learning rate = 0.004800:   Batch Loss = 0.690133, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2305222749710083, Accuracy = 0.8145748972892761
    Iter #131584:  Learning rate = 0.004800:   Batch Loss = 0.708119, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2217433452606201, Accuracy = 0.8149797320365906
    Iter #132096:  Learning rate = 0.004800:   Batch Loss = 0.649365, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2515827417373657, Accuracy = 0.8032388687133789
    Iter #132608:  Learning rate = 0.004800:   Batch Loss = 0.743865, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2253458499908447, Accuracy = 0.8105263113975525
    Iter #133120:  Learning rate = 0.004800:   Batch Loss = 0.727290, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2940924167633057, Accuracy = 0.7789473533630371
    Iter #133632:  Learning rate = 0.004800:   Batch Loss = 0.914332, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3811640739440918, Accuracy = 0.75789475440979
    Iter #134144:  Learning rate = 0.004800:   Batch Loss = 0.864090, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.428891658782959, Accuracy = 0.7502024173736572
    Iter #134656:  Learning rate = 0.004800:   Batch Loss = 0.751807, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3014414310455322, Accuracy = 0.7886639833450317
    Iter #135168:  Learning rate = 0.004800:   Batch Loss = 0.694712, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3393750190734863, Accuracy = 0.7801619172096252
    Iter #135680:  Learning rate = 0.004800:   Batch Loss = 0.863003, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3292254209518433, Accuracy = 0.7692307829856873
    Iter #136192:  Learning rate = 0.004800:   Batch Loss = 0.958784, Accuracy = 0.828125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3423962593078613, Accuracy = 0.7797570824623108
    Iter #136704:  Learning rate = 0.004800:   Batch Loss = 0.738608, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3183540105819702, Accuracy = 0.7740890979766846
    Iter #137216:  Learning rate = 0.004800:   Batch Loss = 0.875262, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3198213577270508, Accuracy = 0.7821862101554871
    Iter #137728:  Learning rate = 0.004800:   Batch Loss = 0.817571, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3595421314239502, Accuracy = 0.7542510032653809
    Iter #138240:  Learning rate = 0.004800:   Batch Loss = 0.750699, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.374133586883545, Accuracy = 0.7651821970939636
    Iter #138752:  Learning rate = 0.004800:   Batch Loss = 0.883626, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3378010988235474, Accuracy = 0.7785425186157227
    Iter #139264:  Learning rate = 0.004800:   Batch Loss = 0.800659, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3508949279785156, Accuracy = 0.7724696397781372
    Iter #139776:  Learning rate = 0.004800:   Batch Loss = 0.860712, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3376975059509277, Accuracy = 0.791093111038208
    Iter #140288:  Learning rate = 0.004800:   Batch Loss = 0.876638, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.301652431488037, Accuracy = 0.7935222387313843
    Iter #140800:  Learning rate = 0.004800:   Batch Loss = 0.741117, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.307788610458374, Accuracy = 0.7821862101554871
    Iter #141312:  Learning rate = 0.004800:   Batch Loss = 0.730210, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2810022830963135, Accuracy = 0.794331967830658
    Iter #141824:  Learning rate = 0.004800:   Batch Loss = 0.754376, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2824759483337402, Accuracy = 0.7894737124443054
    Iter #142336:  Learning rate = 0.004800:   Batch Loss = 0.729484, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3117396831512451, Accuracy = 0.7748987674713135
    Iter #142848:  Learning rate = 0.004800:   Batch Loss = 0.779865, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2781003713607788, Accuracy = 0.7850202322006226
    Iter #143360:  Learning rate = 0.004800:   Batch Loss = 0.763059, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.26790452003479, Accuracy = 0.796761155128479
    Iter #143872:  Learning rate = 0.004800:   Batch Loss = 0.794392, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2399828433990479, Accuracy = 0.7951416969299316
    Iter #144384:  Learning rate = 0.004800:   Batch Loss = 0.757088, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3088738918304443, Accuracy = 0.7817813754081726
    Iter #144896:  Learning rate = 0.004800:   Batch Loss = 0.865278, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3165888786315918, Accuracy = 0.7716599106788635
    Iter #145408:  Learning rate = 0.004800:   Batch Loss = 0.943975, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3660167455673218, Accuracy = 0.7562752962112427
    Iter #145920:  Learning rate = 0.004800:   Batch Loss = 0.772866, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2937966585159302, Accuracy = 0.7813765406608582
    Iter #146432:  Learning rate = 0.004800:   Batch Loss = 1.003340, Accuracy = 0.859375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3119759559631348, Accuracy = 0.777732789516449
    Iter #146944:  Learning rate = 0.004800:   Batch Loss = 0.907073, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.392911434173584, Accuracy = 0.7595141530036926
    Iter #147456:  Learning rate = 0.004800:   Batch Loss = 0.747476, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.331629991531372, Accuracy = 0.7720648050308228
    Iter #147968:  Learning rate = 0.004800:   Batch Loss = 0.964120, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2844347953796387, Accuracy = 0.7781376242637634
    Iter #148480:  Learning rate = 0.004800:   Batch Loss = 0.963777, Accuracy = 0.84375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.316246509552002, Accuracy = 0.7809716463088989
    Iter #148992:  Learning rate = 0.004800:   Batch Loss = 0.868715, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3364825248718262, Accuracy = 0.7728744745254517
    Iter #149504:  Learning rate = 0.004800:   Batch Loss = 0.767515, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2925325632095337, Accuracy = 0.7821862101554871
    Iter #150016:  Learning rate = 0.004800:   Batch Loss = 0.720183, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2713327407836914, Accuracy = 0.7898785471916199
    Iter #150528:  Learning rate = 0.004800:   Batch Loss = 0.798649, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.270378589630127, Accuracy = 0.7898785471916199
    Iter #151040:  Learning rate = 0.004800:   Batch Loss = 0.777422, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2691054344177246, Accuracy = 0.7923076748847961
    Iter #151552:  Learning rate = 0.004800:   Batch Loss = 0.717524, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.3068716526031494, Accuracy = 0.7769230604171753
    Iter #152064:  Learning rate = 0.004800:   Batch Loss = 0.953723, Accuracy = 0.859375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.267011046409607, Accuracy = 0.7963562607765198
    Iter #152576:  Learning rate = 0.004800:   Batch Loss = 0.784136, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2210772037506104, Accuracy = 0.8105263113975525
    Iter #153088:  Learning rate = 0.004800:   Batch Loss = 0.766196, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2474613189697266, Accuracy = 0.7947368621826172
    Iter #153600:  Learning rate = 0.004800:   Batch Loss = 0.726351, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2409547567367554, Accuracy = 0.7963562607765198
    Iter #154112:  Learning rate = 0.004800:   Batch Loss = 0.685554, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2114732265472412, Accuracy = 0.8097165822982788
    Iter #154624:  Learning rate = 0.004800:   Batch Loss = 0.804290, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2005105018615723, Accuracy = 0.8093117475509644
    Iter #155136:  Learning rate = 0.004800:   Batch Loss = 0.665284, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2232515811920166, Accuracy = 0.800000011920929
    Iter #155648:  Learning rate = 0.004800:   Batch Loss = 0.787650, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2193055152893066, Accuracy = 0.7959514260292053
    Iter #156160:  Learning rate = 0.004800:   Batch Loss = 0.734338, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2075257301330566, Accuracy = 0.8036437034606934
    Iter #156672:  Learning rate = 0.004800:   Batch Loss = 0.701051, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2278468608856201, Accuracy = 0.8036437034606934
    Iter #157184:  Learning rate = 0.004800:   Batch Loss = 0.749518, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.214712142944336, Accuracy = 0.8064777255058289
    Iter #157696:  Learning rate = 0.004800:   Batch Loss = 0.658896, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1821027994155884, Accuracy = 0.8161943554878235
    Iter #158208:  Learning rate = 0.004800:   Batch Loss = 0.700431, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2065391540527344, Accuracy = 0.8113360404968262
    Iter #158720:  Learning rate = 0.004800:   Batch Loss = 0.727746, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1889935731887817, Accuracy = 0.8129554390907288
    Iter #159232:  Learning rate = 0.004800:   Batch Loss = 0.676081, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2000329494476318, Accuracy = 0.8141700625419617
    Iter #159744:  Learning rate = 0.004800:   Batch Loss = 0.695957, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1902506351470947, Accuracy = 0.8214575052261353
    Iter #160256:  Learning rate = 0.004800:   Batch Loss = 0.643827, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1867139339447021, Accuracy = 0.8190283179283142
    Iter #160768:  Learning rate = 0.004800:   Batch Loss = 0.604390, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1636955738067627, Accuracy = 0.8202429413795471
    Iter #161280:  Learning rate = 0.004800:   Batch Loss = 0.653410, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1586377620697021, Accuracy = 0.8251011967658997
    Iter #161792:  Learning rate = 0.004800:   Batch Loss = 0.690062, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.187312126159668, Accuracy = 0.8149797320365906
    Iter #162304:  Learning rate = 0.004800:   Batch Loss = 0.652289, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2318590879440308, Accuracy = 0.8012145757675171
    Iter #162816:  Learning rate = 0.004800:   Batch Loss = 0.690516, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.166792869567871, Accuracy = 0.8206477761268616
    Iter #163328:  Learning rate = 0.004800:   Batch Loss = 0.646215, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1881357431411743, Accuracy = 0.8170040249824524
    Iter #163840:  Learning rate = 0.004800:   Batch Loss = 0.653436, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.176128625869751, Accuracy = 0.8157894611358643
    Iter #164352:  Learning rate = 0.004800:   Batch Loss = 0.693957, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2013078927993774, Accuracy = 0.8113360404968262
    Iter #164864:  Learning rate = 0.004800:   Batch Loss = 0.636766, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.159737229347229, Accuracy = 0.8222672343254089
    Iter #165376:  Learning rate = 0.004800:   Batch Loss = 0.688199, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1495320796966553, Accuracy = 0.8234817981719971
    Iter #165888:  Learning rate = 0.004800:   Batch Loss = 0.630473, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1218461990356445, Accuracy = 0.8327935338020325
    Iter #166400:  Learning rate = 0.004800:   Batch Loss = 0.603362, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.130960464477539, Accuracy = 0.8348178267478943
    Iter #166912:  Learning rate = 0.004800:   Batch Loss = 0.647278, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.151009440422058, Accuracy = 0.8340080976486206
    Iter #167424:  Learning rate = 0.004800:   Batch Loss = 0.623716, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1363122463226318, Accuracy = 0.8344129323959351
    Iter #167936:  Learning rate = 0.004800:   Batch Loss = 0.582350, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.124526858329773, Accuracy = 0.8295546770095825
    Iter #168448:  Learning rate = 0.004800:   Batch Loss = 0.628931, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.10652756690979, Accuracy = 0.8295546770095825
    Iter #168960:  Learning rate = 0.004800:   Batch Loss = 0.618635, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0930944681167603, Accuracy = 0.8417003750801086
    Iter #169472:  Learning rate = 0.004800:   Batch Loss = 0.603658, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1029337644577026, Accuracy = 0.8323886394500732
    Iter #169984:  Learning rate = 0.004800:   Batch Loss = 0.650659, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1045880317687988, Accuracy = 0.8315789699554443
    Iter #170496:  Learning rate = 0.004800:   Batch Loss = 0.619457, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1003777980804443, Accuracy = 0.8352226614952087
    Iter #171008:  Learning rate = 0.004800:   Batch Loss = 0.558703, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0978434085845947, Accuracy = 0.8360323905944824
    Iter #171520:  Learning rate = 0.004800:   Batch Loss = 0.643736, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0781999826431274, Accuracy = 0.8404858112335205
    Iter #172032:  Learning rate = 0.004800:   Batch Loss = 0.574598, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1435924768447876, Accuracy = 0.8198380470275879
    Iter #172544:  Learning rate = 0.004800:   Batch Loss = 0.651369, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.147871732711792, Accuracy = 0.8263157606124878
    Iter #173056:  Learning rate = 0.004800:   Batch Loss = 0.633269, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1325749158859253, Accuracy = 0.8263157606124878
    Iter #173568:  Learning rate = 0.004800:   Batch Loss = 0.618113, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1718765497207642, Accuracy = 0.8137651681900024
    Iter #174080:  Learning rate = 0.004800:   Batch Loss = 0.603193, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1576056480407715, Accuracy = 0.8052631616592407
    Iter #174592:  Learning rate = 0.004800:   Batch Loss = 0.695706, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.182686686515808, Accuracy = 0.8145748972892761
    Iter #175104:  Learning rate = 0.004800:   Batch Loss = 0.650177, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1584022045135498, Accuracy = 0.8105263113975525
    Iter #175616:  Learning rate = 0.004800:   Batch Loss = 0.704811, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1516118049621582, Accuracy = 0.8109311461448669
    Iter #176128:  Learning rate = 0.004800:   Batch Loss = 0.605398, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.196305274963379, Accuracy = 0.7935222387313843
    Iter #176640:  Learning rate = 0.004800:   Batch Loss = 0.751762, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1652483940124512, Accuracy = 0.8117408752441406
    Iter #177152:  Learning rate = 0.004800:   Batch Loss = 0.732977, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1644847393035889, Accuracy = 0.8052631616592407
    Iter #177664:  Learning rate = 0.004800:   Batch Loss = 0.752838, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2343600988388062, Accuracy = 0.7842105031013489
    Iter #178176:  Learning rate = 0.004800:   Batch Loss = 0.725224, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.258070468902588, Accuracy = 0.7862347960472107
    Iter #178688:  Learning rate = 0.004800:   Batch Loss = 0.763185, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2040374279022217, Accuracy = 0.7971659898757935
    Iter #179200:  Learning rate = 0.004800:   Batch Loss = 0.737376, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2194892168045044, Accuracy = 0.7971659898757935
    Iter #179712:  Learning rate = 0.004800:   Batch Loss = 0.806249, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2323100566864014, Accuracy = 0.7829959392547607
    Iter #180224:  Learning rate = 0.004800:   Batch Loss = 0.681800, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.165488839149475, Accuracy = 0.8097165822982788
    Iter #180736:  Learning rate = 0.004800:   Batch Loss = 0.730459, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.156691074371338, Accuracy = 0.8206477761268616
    Iter #181248:  Learning rate = 0.004800:   Batch Loss = 0.680693, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1828815937042236, Accuracy = 0.8064777255058289
    Iter #181760:  Learning rate = 0.004800:   Batch Loss = 0.702401, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1855764389038086, Accuracy = 0.8125506043434143
    Iter #182272:  Learning rate = 0.004800:   Batch Loss = 0.814532, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1800434589385986, Accuracy = 0.8056679964065552
    Iter #182784:  Learning rate = 0.004800:   Batch Loss = 0.637610, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2436695098876953, Accuracy = 0.7927125692367554
    Iter #183296:  Learning rate = 0.004800:   Batch Loss = 0.726887, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1564154624938965, Accuracy = 0.8085020184516907
    Iter #183808:  Learning rate = 0.004800:   Batch Loss = 0.651751, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1513121128082275, Accuracy = 0.8137651681900024
    Iter #184320:  Learning rate = 0.004800:   Batch Loss = 0.724086, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.188154935836792, Accuracy = 0.8036437034606934
    Iter #184832:  Learning rate = 0.004800:   Batch Loss = 0.653969, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2104344367980957, Accuracy = 0.8024291396141052
    Iter #185344:  Learning rate = 0.004800:   Batch Loss = 0.729637, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1868903636932373, Accuracy = 0.8052631616592407
    Iter #185856:  Learning rate = 0.004800:   Batch Loss = 0.679582, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1814959049224854, Accuracy = 0.7991902828216553
    Iter #186368:  Learning rate = 0.004800:   Batch Loss = 0.708142, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1977630853652954, Accuracy = 0.7971659898757935
    Iter #186880:  Learning rate = 0.004800:   Batch Loss = 0.713276, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2072360515594482, Accuracy = 0.7874494194984436
    Iter #187392:  Learning rate = 0.004800:   Batch Loss = 0.698603, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1394071578979492, Accuracy = 0.8036437034606934
    Iter #187904:  Learning rate = 0.004800:   Batch Loss = 0.670887, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.143982172012329, Accuracy = 0.8141700625419617
    Iter #188416:  Learning rate = 0.004800:   Batch Loss = 0.733974, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1784287691116333, Accuracy = 0.8068826198577881
    Iter #188928:  Learning rate = 0.004800:   Batch Loss = 0.650930, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1217962503433228, Accuracy = 0.8242915272712708
    Iter #189440:  Learning rate = 0.004800:   Batch Loss = 0.604375, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1616590023040771, Accuracy = 0.8060728907585144
    Iter #189952:  Learning rate = 0.004800:   Batch Loss = 0.611427, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1272754669189453, Accuracy = 0.8251011967658997
    Iter #190464:  Learning rate = 0.004800:   Batch Loss = 0.616698, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1747376918792725, Accuracy = 0.8008097410202026
    Iter #190976:  Learning rate = 0.004800:   Batch Loss = 0.646845, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1754062175750732, Accuracy = 0.8040485978126526
    Iter #191488:  Learning rate = 0.004800:   Batch Loss = 0.598366, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.180971622467041, Accuracy = 0.8036437034606934
    Iter #192000:  Learning rate = 0.004800:   Batch Loss = 0.898406, Accuracy = 0.875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2837936878204346, Accuracy = 0.7797570824623108
    Iter #192512:  Learning rate = 0.004800:   Batch Loss = 0.767206, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.331423282623291, Accuracy = 0.7599190473556519
    Iter #193024:  Learning rate = 0.004800:   Batch Loss = 0.754575, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2306067943572998, Accuracy = 0.7736842036247253
    Iter #193536:  Learning rate = 0.004800:   Batch Loss = 0.854871, Accuracy = 0.875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2639656066894531, Accuracy = 0.7761133313179016
    Iter #194048:  Learning rate = 0.004800:   Batch Loss = 0.720680, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1954671144485474, Accuracy = 0.7882590889930725
    Iter #194560:  Learning rate = 0.004800:   Batch Loss = 0.737194, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2196580171585083, Accuracy = 0.7886639833450317
    Iter #195072:  Learning rate = 0.004800:   Batch Loss = 0.651455, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.149899959564209, Accuracy = 0.8202429413795471
    Iter #195584:  Learning rate = 0.004800:   Batch Loss = 0.790322, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1371638774871826, Accuracy = 0.8105263113975525
    Iter #196096:  Learning rate = 0.004800:   Batch Loss = 0.668390, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1227846145629883, Accuracy = 0.8198380470275879
    Iter #196608:  Learning rate = 0.004800:   Batch Loss = 0.609412, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.127645492553711, Accuracy = 0.8198380470275879
    Iter #197120:  Learning rate = 0.004800:   Batch Loss = 0.588749, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1076613664627075, Accuracy = 0.8170040249824524
    Iter #197632:  Learning rate = 0.004800:   Batch Loss = 0.589230, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0967024564743042, Accuracy = 0.8202429413795471
    Iter #198144:  Learning rate = 0.004800:   Batch Loss = 0.688759, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1307482719421387, Accuracy = 0.8198380470275879
    Iter #198656:  Learning rate = 0.004800:   Batch Loss = 0.604312, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.109179139137268, Accuracy = 0.8275303840637207
    Iter #199168:  Learning rate = 0.004800:   Batch Loss = 0.656531, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1188730001449585, Accuracy = 0.826720654964447
    Iter #199680:  Learning rate = 0.004800:   Batch Loss = 0.778077, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1312485933303833, Accuracy = 0.8145748972892761
    Iter #200192:  Learning rate = 0.004608:   Batch Loss = 0.600784, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0928171873092651, Accuracy = 0.8214575052261353
    Iter #200704:  Learning rate = 0.004608:   Batch Loss = 0.644950, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1044145822525024, Accuracy = 0.8141700625419617
    Iter #201216:  Learning rate = 0.004608:   Batch Loss = 0.604941, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1246992349624634, Accuracy = 0.8109311461448669
    Iter #201728:  Learning rate = 0.004608:   Batch Loss = 0.640608, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1137266159057617, Accuracy = 0.8145748972892761
    Iter #202240:  Learning rate = 0.004608:   Batch Loss = 0.629110, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0835630893707275, Accuracy = 0.8291497826576233
    Iter #202752:  Learning rate = 0.004608:   Batch Loss = 0.619991, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.059936761856079, Accuracy = 0.8319838047027588
    Iter #203264:  Learning rate = 0.004608:   Batch Loss = 0.560973, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0490899085998535, Accuracy = 0.8372469544410706
    Iter #203776:  Learning rate = 0.004608:   Batch Loss = 0.539057, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0440936088562012, Accuracy = 0.840080976486206
    Iter #204288:  Learning rate = 0.004608:   Batch Loss = 0.616525, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0389987230300903, Accuracy = 0.8453441262245178
    Iter #204800:  Learning rate = 0.004608:   Batch Loss = 0.547136, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0354456901550293, Accuracy = 0.8453441262245178
    Iter #205312:  Learning rate = 0.004608:   Batch Loss = 0.564969, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0170185565948486, Accuracy = 0.8485829830169678
    Iter #205824:  Learning rate = 0.004608:   Batch Loss = 0.583306, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0228326320648193, Accuracy = 0.8457489609718323
    Iter #206336:  Learning rate = 0.004608:   Batch Loss = 0.592630, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0744619369506836, Accuracy = 0.8364372253417969
    Iter #206848:  Learning rate = 0.004608:   Batch Loss = 0.516656, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0582728385925293, Accuracy = 0.8360323905944824
    Iter #207360:  Learning rate = 0.004608:   Batch Loss = 0.565287, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.043101191520691, Accuracy = 0.8437246680259705
    Iter #207872:  Learning rate = 0.004608:   Batch Loss = 0.536779, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0086371898651123, Accuracy = 0.8493927121162415
    Iter #208384:  Learning rate = 0.004608:   Batch Loss = 0.543656, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0281802415847778, Accuracy = 0.8449392914772034
    Iter #208896:  Learning rate = 0.004608:   Batch Loss = 0.525202, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0596587657928467, Accuracy = 0.8408907055854797
    Iter #209408:  Learning rate = 0.004608:   Batch Loss = 0.606977, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0529676675796509, Accuracy = 0.8502024412155151
    Iter #209920:  Learning rate = 0.004608:   Batch Loss = 0.505309, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0472517013549805, Accuracy = 0.8392712473869324
    Iter #210432:  Learning rate = 0.004608:   Batch Loss = 0.575342, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0645045042037964, Accuracy = 0.8319838047027588
    Iter #210944:  Learning rate = 0.004608:   Batch Loss = 0.568868, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.08102548122406, Accuracy = 0.8246963620185852
    Iter #211456:  Learning rate = 0.004608:   Batch Loss = 0.578462, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.104038119316101, Accuracy = 0.8089068531990051
    Iter #211968:  Learning rate = 0.004608:   Batch Loss = 0.624365, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0907224416732788, Accuracy = 0.8255060911178589
    Iter #212480:  Learning rate = 0.004608:   Batch Loss = 0.586609, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1963112354278564, Accuracy = 0.7850202322006226
    Iter #212992:  Learning rate = 0.004608:   Batch Loss = 0.670515, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1676064729690552, Accuracy = 0.7955465316772461
    Iter #213504:  Learning rate = 0.004608:   Batch Loss = 0.654249, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1709481477737427, Accuracy = 0.7939271330833435
    Iter #214016:  Learning rate = 0.004608:   Batch Loss = 0.728431, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1427932977676392, Accuracy = 0.794331967830658
    Iter #214528:  Learning rate = 0.004608:   Batch Loss = 0.581832, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1900687217712402, Accuracy = 0.7862347960472107
    Iter #215040:  Learning rate = 0.004608:   Batch Loss = 0.641598, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1231375932693481, Accuracy = 0.8097165822982788
    Iter #215552:  Learning rate = 0.004608:   Batch Loss = 0.626514, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.459888219833374, Accuracy = 0.7153846025466919
    Iter #216064:  Learning rate = 0.004608:   Batch Loss = 0.565045, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1312501430511475, Accuracy = 0.8097165822982788
    Iter #216576:  Learning rate = 0.004608:   Batch Loss = 0.589717, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0957221984863281, Accuracy = 0.8080971837043762
    Iter #217088:  Learning rate = 0.004608:   Batch Loss = 0.628145, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.137852668762207, Accuracy = 0.8048583269119263
    Iter #217600:  Learning rate = 0.004608:   Batch Loss = 0.623340, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1204710006713867, Accuracy = 0.807692289352417
    Iter #218112:  Learning rate = 0.004608:   Batch Loss = 0.607647, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1243380308151245, Accuracy = 0.8085020184516907
    Iter #218624:  Learning rate = 0.004608:   Batch Loss = 0.784315, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1251561641693115, Accuracy = 0.8097165822982788
    Iter #219136:  Learning rate = 0.004608:   Batch Loss = 0.564254, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1564466953277588, Accuracy = 0.804453432559967
    Iter #219648:  Learning rate = 0.004608:   Batch Loss = 0.634528, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1370258331298828, Accuracy = 0.8089068531990051
    Iter #220160:  Learning rate = 0.004608:   Batch Loss = 0.764507, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1180235147476196, Accuracy = 0.813360333442688
    Iter #220672:  Learning rate = 0.004608:   Batch Loss = 0.584860, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.159077525138855, Accuracy = 0.8020243048667908
    Iter #221184:  Learning rate = 0.004608:   Batch Loss = 0.591322, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1223030090332031, Accuracy = 0.8194332122802734
    Iter #221696:  Learning rate = 0.004608:   Batch Loss = 0.600207, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0924768447875977, Accuracy = 0.8222672343254089
    Iter #222208:  Learning rate = 0.004608:   Batch Loss = 0.572439, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1197810173034668, Accuracy = 0.8206477761268616
    Iter #222720:  Learning rate = 0.004608:   Batch Loss = 0.553923, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0794661045074463, Accuracy = 0.8291497826576233
    Iter #223232:  Learning rate = 0.004608:   Batch Loss = 0.545648, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0579906702041626, Accuracy = 0.829959511756897
    Iter #223744:  Learning rate = 0.004608:   Batch Loss = 0.568754, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0756806135177612, Accuracy = 0.8255060911178589
    Iter #224256:  Learning rate = 0.004608:   Batch Loss = 0.628243, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0605183839797974, Accuracy = 0.8279352188110352
    Iter #224768:  Learning rate = 0.004608:   Batch Loss = 0.520999, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0803284645080566, Accuracy = 0.8246963620185852
    Iter #225280:  Learning rate = 0.004608:   Batch Loss = 0.538988, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0848408937454224, Accuracy = 0.8271254897117615
    Iter #225792:  Learning rate = 0.004608:   Batch Loss = 0.512097, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0565671920776367, Accuracy = 0.8344129323959351
    Iter #226304:  Learning rate = 0.004608:   Batch Loss = 0.523975, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0418702363967896, Accuracy = 0.8437246680259705
    Iter #226816:  Learning rate = 0.004608:   Batch Loss = 0.538800, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.040739893913269, Accuracy = 0.8380566835403442
    Iter #227328:  Learning rate = 0.004608:   Batch Loss = 0.547712, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0699782371520996, Accuracy = 0.8178137540817261
    Iter #227840:  Learning rate = 0.004608:   Batch Loss = 0.557422, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0733227729797363, Accuracy = 0.8287449479103088
    Iter #228352:  Learning rate = 0.004608:   Batch Loss = 0.530297, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0806783437728882, Accuracy = 0.8234817981719971
    Iter #228864:  Learning rate = 0.004608:   Batch Loss = 0.559355, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0888737440109253, Accuracy = 0.8145748972892761
    Iter #229376:  Learning rate = 0.004608:   Batch Loss = 0.538408, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1408131122589111, Accuracy = 0.7963562607765198
    Iter #229888:  Learning rate = 0.004608:   Batch Loss = 0.585752, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.099421739578247, Accuracy = 0.8093117475509644
    Iter #230400:  Learning rate = 0.004608:   Batch Loss = 0.721215, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1372625827789307, Accuracy = 0.7975708246231079
    Iter #230912:  Learning rate = 0.004608:   Batch Loss = 0.605288, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1186755895614624, Accuracy = 0.8020243048667908
    Iter #231424:  Learning rate = 0.004608:   Batch Loss = 0.663369, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1310226917266846, Accuracy = 0.7975708246231079
    Iter #231936:  Learning rate = 0.004608:   Batch Loss = 0.721333, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.082029938697815, Accuracy = 0.8161943554878235
    Iter #232448:  Learning rate = 0.004608:   Batch Loss = 0.581336, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1133829355239868, Accuracy = 0.8008097410202026
    Iter #232960:  Learning rate = 0.004608:   Batch Loss = 0.693412, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0840251445770264, Accuracy = 0.807692289352417
    Iter #233472:  Learning rate = 0.004608:   Batch Loss = 0.662963, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1061432361602783, Accuracy = 0.7935222387313843
    Iter #233984:  Learning rate = 0.004608:   Batch Loss = 0.658615, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1926774978637695, Accuracy = 0.7676113247871399
    Iter #234496:  Learning rate = 0.004608:   Batch Loss = 0.614190, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1295055150985718, Accuracy = 0.7914980053901672
    Iter #235008:  Learning rate = 0.004608:   Batch Loss = 0.638352, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1580041646957397, Accuracy = 0.7959514260292053
    Iter #235520:  Learning rate = 0.004608:   Batch Loss = 0.572183, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1431084871292114, Accuracy = 0.7971659898757935
    Iter #236032:  Learning rate = 0.004608:   Batch Loss = 0.629573, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.083419919013977, Accuracy = 0.8226720690727234
    Iter #236544:  Learning rate = 0.004608:   Batch Loss = 0.547116, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0790398120880127, Accuracy = 0.8222672343254089
    Iter #237056:  Learning rate = 0.004608:   Batch Loss = 0.588624, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.099000096321106, Accuracy = 0.8170040249824524
    Iter #237568:  Learning rate = 0.004608:   Batch Loss = 0.638181, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.09548020362854, Accuracy = 0.8109311461448669
    Iter #238080:  Learning rate = 0.004608:   Batch Loss = 0.565653, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1044363975524902, Accuracy = 0.8230769038200378
    Iter #238592:  Learning rate = 0.004608:   Batch Loss = 0.568068, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1081135272979736, Accuracy = 0.8198380470275879
    Iter #239104:  Learning rate = 0.004608:   Batch Loss = 0.527457, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.078368902206421, Accuracy = 0.8234817981719971
    Iter #239616:  Learning rate = 0.004608:   Batch Loss = 0.567938, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0326199531555176, Accuracy = 0.8417003750801086
    Iter #240128:  Learning rate = 0.004608:   Batch Loss = 0.537365, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0296090841293335, Accuracy = 0.8441295623779297
    Iter #240640:  Learning rate = 0.004608:   Batch Loss = 0.500336, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0348576307296753, Accuracy = 0.829959511756897
    Iter #241152:  Learning rate = 0.004608:   Batch Loss = 0.575674, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.03934645652771, Accuracy = 0.8279352188110352
    Iter #241664:  Learning rate = 0.004608:   Batch Loss = 0.546058, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0485572814941406, Accuracy = 0.8238866329193115
    Iter #242176:  Learning rate = 0.004608:   Batch Loss = 0.588409, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1220890283584595, Accuracy = 0.8097165822982788
    Iter #242688:  Learning rate = 0.004608:   Batch Loss = 0.525224, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1191215515136719, Accuracy = 0.8068826198577881
    Iter #243200:  Learning rate = 0.004608:   Batch Loss = 0.531403, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1146048307418823, Accuracy = 0.8028340339660645
    Iter #243712:  Learning rate = 0.004608:   Batch Loss = 0.711835, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1296051740646362, Accuracy = 0.7951416969299316
    Iter #244224:  Learning rate = 0.004608:   Batch Loss = 0.805770, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1998087167739868, Accuracy = 0.78340083360672
    Iter #244736:  Learning rate = 0.004608:   Batch Loss = 0.700801, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1836169958114624, Accuracy = 0.7874494194984436
    Iter #245248:  Learning rate = 0.004608:   Batch Loss = 0.622184, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1822882890701294, Accuracy = 0.7813765406608582
    Iter #245760:  Learning rate = 0.004608:   Batch Loss = 0.722961, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.20864999294281, Accuracy = 0.7757084965705872
    Iter #246272:  Learning rate = 0.004608:   Batch Loss = 0.559290, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.111382246017456, Accuracy = 0.8080971837043762
    Iter #246784:  Learning rate = 0.004608:   Batch Loss = 0.624723, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1023238897323608, Accuracy = 0.8064777255058289
    Iter #247296:  Learning rate = 0.004608:   Batch Loss = 0.529983, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0925264358520508, Accuracy = 0.8040485978126526
    Iter #247808:  Learning rate = 0.004608:   Batch Loss = 0.668657, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0703890323638916, Accuracy = 0.8109311461448669
    Iter #248320:  Learning rate = 0.004608:   Batch Loss = 0.590747, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1220319271087646, Accuracy = 0.7991902828216553
    Iter #248832:  Learning rate = 0.004608:   Batch Loss = 0.636171, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1050610542297363, Accuracy = 0.813360333442688
    Iter #249344:  Learning rate = 0.004608:   Batch Loss = 0.548447, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1033307313919067, Accuracy = 0.8174089193344116
    Iter #249856:  Learning rate = 0.004608:   Batch Loss = 0.539790, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0939481258392334, Accuracy = 0.8056679964065552
    Iter #250368:  Learning rate = 0.004608:   Batch Loss = 0.652206, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.066609263420105, Accuracy = 0.8246963620185852
    Iter #250880:  Learning rate = 0.004608:   Batch Loss = 0.617657, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0377349853515625, Accuracy = 0.8336032629013062
    Iter #251392:  Learning rate = 0.004608:   Batch Loss = 0.556082, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0217188596725464, Accuracy = 0.8303643465042114
    Iter #251904:  Learning rate = 0.004608:   Batch Loss = 0.526135, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0535509586334229, Accuracy = 0.8238866329193115
    Iter #252416:  Learning rate = 0.004608:   Batch Loss = 0.511595, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0230002403259277, Accuracy = 0.8331983685493469
    Iter #252928:  Learning rate = 0.004608:   Batch Loss = 0.525102, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0110292434692383, Accuracy = 0.8364372253417969
    Iter #253440:  Learning rate = 0.004608:   Batch Loss = 0.520312, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9820454120635986, Accuracy = 0.8510121703147888
    Iter #253952:  Learning rate = 0.004608:   Batch Loss = 0.506028, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.021223783493042, Accuracy = 0.843319833278656
    Iter #254464:  Learning rate = 0.004608:   Batch Loss = 0.532642, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9927793741226196, Accuracy = 0.8380566835403442
    Iter #254976:  Learning rate = 0.004608:   Batch Loss = 0.544509, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0177698135375977, Accuracy = 0.8303643465042114
    Iter #255488:  Learning rate = 0.004608:   Batch Loss = 0.551609, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0104918479919434, Accuracy = 0.8376518487930298
    Iter #256000:  Learning rate = 0.004608:   Batch Loss = 0.501868, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0008145570755005, Accuracy = 0.8404858112335205
    Iter #256512:  Learning rate = 0.004608:   Batch Loss = 0.510037, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9886420965194702, Accuracy = 0.8437246680259705
    Iter #257024:  Learning rate = 0.004608:   Batch Loss = 0.530182, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9750426411628723, Accuracy = 0.8404858112335205
    Iter #257536:  Learning rate = 0.004608:   Batch Loss = 0.578339, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9763212203979492, Accuracy = 0.8481781482696533
    Iter #258048:  Learning rate = 0.004608:   Batch Loss = 0.473709, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9995445013046265, Accuracy = 0.8441295623779297
    Iter #258560:  Learning rate = 0.004608:   Batch Loss = 0.496147, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9938870072364807, Accuracy = 0.8396761417388916
    Iter #259072:  Learning rate = 0.004608:   Batch Loss = 0.481284, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9823373556137085, Accuracy = 0.8453441262245178
    Iter #259584:  Learning rate = 0.004608:   Batch Loss = 0.451037, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9861012697219849, Accuracy = 0.8384615182876587
    Iter #260096:  Learning rate = 0.004608:   Batch Loss = 0.488178, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.013047695159912, Accuracy = 0.8396761417388916
    Iter #260608:  Learning rate = 0.004608:   Batch Loss = 0.457162, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9995871782302856, Accuracy = 0.8311740756034851
    Iter #261120:  Learning rate = 0.004608:   Batch Loss = 0.534629, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0245516300201416, Accuracy = 0.8251011967658997
    Iter #261632:  Learning rate = 0.004608:   Batch Loss = 0.513665, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0225592851638794, Accuracy = 0.8384615182876587
    Iter #262144:  Learning rate = 0.004608:   Batch Loss = 0.495488, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0553336143493652, Accuracy = 0.8230769038200378
    Iter #262656:  Learning rate = 0.004608:   Batch Loss = 0.516729, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0502245426177979, Accuracy = 0.8129554390907288
    Iter #263168:  Learning rate = 0.004608:   Batch Loss = 0.540962, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0395786762237549, Accuracy = 0.8279352188110352
    Iter #263680:  Learning rate = 0.004608:   Batch Loss = 0.497529, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.01521635055542, Accuracy = 0.8230769038200378
    Iter #264192:  Learning rate = 0.004608:   Batch Loss = 0.535729, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.028084397315979, Accuracy = 0.8242915272712708
    Iter #264704:  Learning rate = 0.004608:   Batch Loss = 0.498811, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0184134244918823, Accuracy = 0.826720654964447
    Iter #265216:  Learning rate = 0.004608:   Batch Loss = 0.523810, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0120136737823486, Accuracy = 0.8226720690727234
    Iter #265728:  Learning rate = 0.004608:   Batch Loss = 0.516757, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0300369262695312, Accuracy = 0.8323886394500732
    Iter #266240:  Learning rate = 0.004608:   Batch Loss = 0.474523, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.998360276222229, Accuracy = 0.8376518487930298
    Iter #266752:  Learning rate = 0.004608:   Batch Loss = 0.513689, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9923146963119507, Accuracy = 0.8412955403327942
    Iter #267264:  Learning rate = 0.004608:   Batch Loss = 0.491265, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0171070098876953, Accuracy = 0.8303643465042114
    Iter #267776:  Learning rate = 0.004608:   Batch Loss = 0.501215, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0178625583648682, Accuracy = 0.8287449479103088
    Iter #268288:  Learning rate = 0.004608:   Batch Loss = 0.501098, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9813311100006104, Accuracy = 0.8449392914772034
    Iter #268800:  Learning rate = 0.004608:   Batch Loss = 0.455664, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9885790944099426, Accuracy = 0.8441295623779297
    Iter #269312:  Learning rate = 0.004608:   Batch Loss = 0.468176, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9762309789657593, Accuracy = 0.8445343971252441
    Iter #269824:  Learning rate = 0.004608:   Batch Loss = 0.439255, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.957893967628479, Accuracy = 0.8408907055854797
    Iter #270336:  Learning rate = 0.004608:   Batch Loss = 0.502614, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9956657290458679, Accuracy = 0.8344129323959351
    Iter #270848:  Learning rate = 0.004608:   Batch Loss = 0.492438, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9787944555282593, Accuracy = 0.8376518487930298
    Iter #271360:  Learning rate = 0.004608:   Batch Loss = 0.473096, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9905290603637695, Accuracy = 0.835627555847168
    Iter #271872:  Learning rate = 0.004608:   Batch Loss = 0.502975, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9840099811553955, Accuracy = 0.8445343971252441
    Iter #272384:  Learning rate = 0.004608:   Batch Loss = 0.497347, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.955640435218811, Accuracy = 0.843319833278656
    Iter #272896:  Learning rate = 0.004608:   Batch Loss = 0.471584, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9580353498458862, Accuracy = 0.8441295623779297
    Iter #273408:  Learning rate = 0.004608:   Batch Loss = 0.462351, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9560321569442749, Accuracy = 0.8477732539176941
    Iter #273920:  Learning rate = 0.004608:   Batch Loss = 0.627482, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9790834188461304, Accuracy = 0.8408907055854797
    Iter #274432:  Learning rate = 0.004608:   Batch Loss = 0.466841, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.031611442565918, Accuracy = 0.8165991902351379
    Iter #274944:  Learning rate = 0.004608:   Batch Loss = 0.581099, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0142221450805664, Accuracy = 0.8230769038200378
    Iter #275456:  Learning rate = 0.004608:   Batch Loss = 0.545996, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0749831199645996, Accuracy = 0.8080971837043762
    Iter #275968:  Learning rate = 0.004608:   Batch Loss = 0.597403, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0368390083312988, Accuracy = 0.8157894611358643
    Iter #276480:  Learning rate = 0.004608:   Batch Loss = 0.546657, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.108144760131836, Accuracy = 0.7935222387313843
    Iter #276992:  Learning rate = 0.004608:   Batch Loss = 0.663911, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1258000135421753, Accuracy = 0.7890688180923462
    Iter #277504:  Learning rate = 0.004608:   Batch Loss = 0.580480, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1152567863464355, Accuracy = 0.7931174039840698
    Iter #278016:  Learning rate = 0.004608:   Batch Loss = 0.623648, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0852676630020142, Accuracy = 0.8052631616592407
    Iter #278528:  Learning rate = 0.004608:   Batch Loss = 0.548398, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0710713863372803, Accuracy = 0.8052631616592407
    Iter #279040:  Learning rate = 0.004608:   Batch Loss = 0.551675, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0669703483581543, Accuracy = 0.8068826198577881
    Iter #279552:  Learning rate = 0.004608:   Batch Loss = 0.576380, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0904886722564697, Accuracy = 0.8004048466682434
    Iter #280064:  Learning rate = 0.004608:   Batch Loss = 0.652455, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1010266542434692, Accuracy = 0.7971659898757935
    Iter #280576:  Learning rate = 0.004608:   Batch Loss = 0.629692, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0775244235992432, Accuracy = 0.8024291396141052
    Iter #281088:  Learning rate = 0.004608:   Batch Loss = 0.598880, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0589309930801392, Accuracy = 0.8190283179283142
    Iter #281600:  Learning rate = 0.004608:   Batch Loss = 0.525295, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.089963674545288, Accuracy = 0.7979757189750671
    Iter #282112:  Learning rate = 0.004608:   Batch Loss = 0.616038, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0623998641967773, Accuracy = 0.8153846263885498
    Iter #282624:  Learning rate = 0.004608:   Batch Loss = 0.531464, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1118748188018799, Accuracy = 0.7979757189750671
    Iter #283136:  Learning rate = 0.004608:   Batch Loss = 0.590267, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0837851762771606, Accuracy = 0.8109311461448669
    Iter #283648:  Learning rate = 0.004608:   Batch Loss = 0.582317, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0909980535507202, Accuracy = 0.8060728907585144
    Iter #284160:  Learning rate = 0.004608:   Batch Loss = 0.611063, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.061417579650879, Accuracy = 0.810121476650238
    Iter #284672:  Learning rate = 0.004608:   Batch Loss = 0.647360, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.128114938735962, Accuracy = 0.7939271330833435
    Iter #285184:  Learning rate = 0.004608:   Batch Loss = 0.585503, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0758908987045288, Accuracy = 0.8008097410202026
    Iter #285696:  Learning rate = 0.004608:   Batch Loss = 0.616745, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.041541576385498, Accuracy = 0.8178137540817261
    Iter #286208:  Learning rate = 0.004608:   Batch Loss = 0.585914, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.090591549873352, Accuracy = 0.8040485978126526
    Iter #286720:  Learning rate = 0.004608:   Batch Loss = 0.603959, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0912001132965088, Accuracy = 0.8052631616592407
    Iter #287232:  Learning rate = 0.004608:   Batch Loss = 0.674801, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0773874521255493, Accuracy = 0.8020243048667908
    Iter #287744:  Learning rate = 0.004608:   Batch Loss = 0.568472, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0698060989379883, Accuracy = 0.813360333442688
    Iter #288256:  Learning rate = 0.004608:   Batch Loss = 0.634070, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.09218168258667, Accuracy = 0.8097165822982788
    Iter #288768:  Learning rate = 0.004608:   Batch Loss = 0.628753, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0994473695755005, Accuracy = 0.8157894611358643
    Iter #289280:  Learning rate = 0.004608:   Batch Loss = 0.594989, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1360893249511719, Accuracy = 0.7951416969299316
    Iter #289792:  Learning rate = 0.004608:   Batch Loss = 0.621261, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1425118446350098, Accuracy = 0.8032388687133789
    Iter #290304:  Learning rate = 0.004608:   Batch Loss = 0.665869, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1568771600723267, Accuracy = 0.7878542542457581
    Iter #290816:  Learning rate = 0.004608:   Batch Loss = 0.647192, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0977742671966553, Accuracy = 0.813360333442688
    Iter #291328:  Learning rate = 0.004608:   Batch Loss = 0.577105, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0907312631607056, Accuracy = 0.8052631616592407
    Iter #291840:  Learning rate = 0.004608:   Batch Loss = 0.692147, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0733981132507324, Accuracy = 0.8105263113975525
    Iter #292352:  Learning rate = 0.004608:   Batch Loss = 0.577289, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0662070512771606, Accuracy = 0.8198380470275879
    Iter #292864:  Learning rate = 0.004608:   Batch Loss = 0.564370, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.062003493309021, Accuracy = 0.8259109258651733
    Iter #293376:  Learning rate = 0.004608:   Batch Loss = 0.530618, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.055572509765625, Accuracy = 0.826720654964447
    Iter #293888:  Learning rate = 0.004608:   Batch Loss = 0.468783, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.033958911895752, Accuracy = 0.8291497826576233
    Iter #294400:  Learning rate = 0.004608:   Batch Loss = 0.500717, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0345661640167236, Accuracy = 0.8251011967658997
    Iter #294912:  Learning rate = 0.004608:   Batch Loss = 0.466603, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0091255903244019, Accuracy = 0.8336032629013062
    Iter #295424:  Learning rate = 0.004608:   Batch Loss = 0.485281, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.006303310394287, Accuracy = 0.8315789699554443
    Iter #295936:  Learning rate = 0.004608:   Batch Loss = 0.518489, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.035645842552185, Accuracy = 0.8230769038200378
    Iter #296448:  Learning rate = 0.004608:   Batch Loss = 0.456199, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0817677974700928, Accuracy = 0.8202429413795471
    Iter #296960:  Learning rate = 0.004608:   Batch Loss = 0.498879, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0680270195007324, Accuracy = 0.8105263113975525
    Iter #297472:  Learning rate = 0.004608:   Batch Loss = 0.679076, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0794031620025635, Accuracy = 0.8052631616592407
    Iter #297984:  Learning rate = 0.004608:   Batch Loss = 0.576552, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.040163278579712, Accuracy = 0.8161943554878235
    Iter #298496:  Learning rate = 0.004608:   Batch Loss = 0.495972, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0209920406341553, Accuracy = 0.8226720690727234
    Iter #299008:  Learning rate = 0.004608:   Batch Loss = 0.503515, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.013809323310852, Accuracy = 0.8311740756034851
    Iter #299520:  Learning rate = 0.004608:   Batch Loss = 0.505153, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.031062364578247, Accuracy = 0.8319838047027588
    Iter #300032:  Learning rate = 0.004424:   Batch Loss = 0.459315, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9885802268981934, Accuracy = 0.8344129323959351
    Iter #300544:  Learning rate = 0.004424:   Batch Loss = 0.465991, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0162420272827148, Accuracy = 0.8234817981719971
    Iter #301056:  Learning rate = 0.004424:   Batch Loss = 0.489694, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0080645084381104, Accuracy = 0.8315789699554443
    Iter #301568:  Learning rate = 0.004424:   Batch Loss = 0.534045, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0444142818450928, Accuracy = 0.821052610874176
    Iter #302080:  Learning rate = 0.004424:   Batch Loss = 0.522612, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0502912998199463, Accuracy = 0.8226720690727234
    Iter #302592:  Learning rate = 0.004424:   Batch Loss = 0.537134, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0484118461608887, Accuracy = 0.8271254897117615
    Iter #303104:  Learning rate = 0.004424:   Batch Loss = 0.454848, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.04282808303833, Accuracy = 0.8238866329193115
    Iter #303616:  Learning rate = 0.004424:   Batch Loss = 0.505335, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.02077317237854, Accuracy = 0.8311740756034851
    Iter #304128:  Learning rate = 0.004424:   Batch Loss = 0.536557, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0192004442214966, Accuracy = 0.8348178267478943
    Iter #304640:  Learning rate = 0.004424:   Batch Loss = 0.452292, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9956603050231934, Accuracy = 0.840080976486206
    Iter #305152:  Learning rate = 0.004424:   Batch Loss = 0.464157, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9785493612289429, Accuracy = 0.8425101041793823
    Iter #305664:  Learning rate = 0.004424:   Batch Loss = 0.479485, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9542885422706604, Accuracy = 0.8449392914772034
    Iter #306176:  Learning rate = 0.004424:   Batch Loss = 0.444658, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9590650200843811, Accuracy = 0.8493927121162415
    Iter #306688:  Learning rate = 0.004424:   Batch Loss = 0.441094, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9518420100212097, Accuracy = 0.8485829830169678
    Iter #307200:  Learning rate = 0.004424:   Batch Loss = 0.429649, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.925942063331604, Accuracy = 0.8506072759628296
    Iter #307712:  Learning rate = 0.004424:   Batch Loss = 0.430331, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9344317317008972, Accuracy = 0.8530364632606506
    Iter #308224:  Learning rate = 0.004424:   Batch Loss = 0.428621, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9300645589828491, Accuracy = 0.856680154800415
    Iter #308736:  Learning rate = 0.004424:   Batch Loss = 0.448345, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9173681735992432, Accuracy = 0.8530364632606506
    Iter #309248:  Learning rate = 0.004424:   Batch Loss = 0.442360, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9206064343452454, Accuracy = 0.8550607562065125
    Iter #309760:  Learning rate = 0.004424:   Batch Loss = 0.416532, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.90241938829422, Accuracy = 0.8619433045387268
    Iter #310272:  Learning rate = 0.004424:   Batch Loss = 0.400611, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8955918550491333, Accuracy = 0.8651821613311768
    Iter #310784:  Learning rate = 0.004424:   Batch Loss = 0.438952, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9115344285964966, Accuracy = 0.8578947186470032
    Iter #311296:  Learning rate = 0.004424:   Batch Loss = 0.412632, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9043416976928711, Accuracy = 0.8603239059448242
    Iter #311808:  Learning rate = 0.004424:   Batch Loss = 0.416925, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9112814664840698, Accuracy = 0.8546558618545532
    Iter #312320:  Learning rate = 0.004424:   Batch Loss = 0.427833, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9038841724395752, Accuracy = 0.8587044477462769
    Iter #312832:  Learning rate = 0.004424:   Batch Loss = 0.405435, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8936166763305664, Accuracy = 0.8570850491523743
    Iter #313344:  Learning rate = 0.004424:   Batch Loss = 0.408005, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8894078731536865, Accuracy = 0.8574898838996887
    Iter #313856:  Learning rate = 0.004424:   Batch Loss = 0.413316, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8911588191986084, Accuracy = 0.8587044477462769
    Iter #314368:  Learning rate = 0.004424:   Batch Loss = 0.419856, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.87995445728302, Accuracy = 0.8639675974845886
    Iter #314880:  Learning rate = 0.004424:   Batch Loss = 0.412030, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8758120536804199, Accuracy = 0.8647773265838623
    Iter #315392:  Learning rate = 0.004424:   Batch Loss = 0.402329, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8724272847175598, Accuracy = 0.8587044477462769
    Iter #315904:  Learning rate = 0.004424:   Batch Loss = 0.419522, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8819479942321777, Accuracy = 0.8558704257011414
    Iter #316416:  Learning rate = 0.004424:   Batch Loss = 0.403251, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8859230279922485, Accuracy = 0.859919011592865
    Iter #316928:  Learning rate = 0.004424:   Batch Loss = 0.389883, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.898836612701416, Accuracy = 0.8538461327552795
    Iter #317440:  Learning rate = 0.004424:   Batch Loss = 0.393905, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9022818803787231, Accuracy = 0.8526315689086914
    Iter #317952:  Learning rate = 0.004424:   Batch Loss = 0.406913, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9003064632415771, Accuracy = 0.8526315689086914
    Iter #318464:  Learning rate = 0.004424:   Batch Loss = 0.388970, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9137293696403503, Accuracy = 0.8481781482696533
    Iter #318976:  Learning rate = 0.004424:   Batch Loss = 0.403212, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9173669219017029, Accuracy = 0.8457489609718323
    Iter #319488:  Learning rate = 0.004424:   Batch Loss = 0.412382, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8829892873764038, Accuracy = 0.8562753200531006
    Iter #320000:  Learning rate = 0.004424:   Batch Loss = 0.402277, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8827782869338989, Accuracy = 0.8538461327552795
    Iter #320512:  Learning rate = 0.004424:   Batch Loss = 0.385957, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8993160128593445, Accuracy = 0.8493927121162415
    Iter #321024:  Learning rate = 0.004424:   Batch Loss = 0.416430, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.903052806854248, Accuracy = 0.8461538553237915
    Iter #321536:  Learning rate = 0.004424:   Batch Loss = 0.389074, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9046326875686646, Accuracy = 0.8493927121162415
    Iter #322048:  Learning rate = 0.004424:   Batch Loss = 0.384009, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8911221027374268, Accuracy = 0.846558690071106
    Iter #322560:  Learning rate = 0.004424:   Batch Loss = 0.390905, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8867301344871521, Accuracy = 0.8506072759628296
    Iter #323072:  Learning rate = 0.004424:   Batch Loss = 0.383243, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8938056230545044, Accuracy = 0.8493927121162415
    Iter #323584:  Learning rate = 0.004424:   Batch Loss = 0.382084, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8784715533256531, Accuracy = 0.8550607562065125
    Iter #324096:  Learning rate = 0.004424:   Batch Loss = 0.386068, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8704186081886292, Accuracy = 0.8615384697914124
    Iter #324608:  Learning rate = 0.004424:   Batch Loss = 0.374469, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8598822355270386, Accuracy = 0.8651821613311768
    Iter #325120:  Learning rate = 0.004424:   Batch Loss = 0.366414, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8600344061851501, Accuracy = 0.8582996129989624
    Iter #325632:  Learning rate = 0.004424:   Batch Loss = 0.372104, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8536707162857056, Accuracy = 0.8595141768455505
    Iter #326144:  Learning rate = 0.004424:   Batch Loss = 0.366339, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8513420224189758, Accuracy = 0.8668016195297241
    Iter #326656:  Learning rate = 0.004424:   Batch Loss = 0.365907, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8547015190124512, Accuracy = 0.8611335754394531
    Iter #327168:  Learning rate = 0.004424:   Batch Loss = 0.382598, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.851140558719635, Accuracy = 0.8603239059448242
    Iter #327680:  Learning rate = 0.004424:   Batch Loss = 0.380318, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.883296549320221, Accuracy = 0.8514170050621033
    Iter #328192:  Learning rate = 0.004424:   Batch Loss = 0.374998, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.850621223449707, Accuracy = 0.8672064542770386
    Iter #328704:  Learning rate = 0.004424:   Batch Loss = 0.408819, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8460096716880798, Accuracy = 0.8643724918365479
    Iter #329216:  Learning rate = 0.004424:   Batch Loss = 0.368324, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8688255548477173, Accuracy = 0.8477732539176941
    Iter #329728:  Learning rate = 0.004424:   Batch Loss = 0.375120, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8648420572280884, Accuracy = 0.8514170050621033
    Iter #330240:  Learning rate = 0.004424:   Batch Loss = 0.391575, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.853095293045044, Accuracy = 0.856680154800415
    Iter #330752:  Learning rate = 0.004424:   Batch Loss = 0.368945, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8507637977600098, Accuracy = 0.8651821613311768
    Iter #331264:  Learning rate = 0.004424:   Batch Loss = 0.365509, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8554397821426392, Accuracy = 0.8635627627372742
    Iter #331776:  Learning rate = 0.004424:   Batch Loss = 0.371477, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8422545194625854, Accuracy = 0.8663967847824097
    Iter #332288:  Learning rate = 0.004424:   Batch Loss = 0.356326, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8317500352859497, Accuracy = 0.8676113486289978
    Iter #332800:  Learning rate = 0.004424:   Batch Loss = 0.371613, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8284894227981567, Accuracy = 0.8720647692680359
    Iter #333312:  Learning rate = 0.004424:   Batch Loss = 0.370900, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8259831666946411, Accuracy = 0.8696356415748596
    Iter #333824:  Learning rate = 0.004424:   Batch Loss = 0.347697, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8314791917800903, Accuracy = 0.8647773265838623
    Iter #334336:  Learning rate = 0.004424:   Batch Loss = 0.353046, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8298577070236206, Accuracy = 0.862348198890686
    Iter #334848:  Learning rate = 0.004424:   Batch Loss = 0.348025, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8392689228057861, Accuracy = 0.8603239059448242
    Iter #335360:  Learning rate = 0.004424:   Batch Loss = 0.357915, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8460513353347778, Accuracy = 0.8587044477462769
    Iter #335872:  Learning rate = 0.004424:   Batch Loss = 0.421463, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8588323593139648, Accuracy = 0.8558704257011414
    Iter #336384:  Learning rate = 0.004424:   Batch Loss = 0.423958, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9142900705337524, Accuracy = 0.8360323905944824
    Iter #336896:  Learning rate = 0.004424:   Batch Loss = 0.518114, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9553350210189819, Accuracy = 0.8251011967658997
    Iter #337408:  Learning rate = 0.004424:   Batch Loss = 0.490239, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9704395532608032, Accuracy = 0.8238866329193115
    Iter #337920:  Learning rate = 0.004424:   Batch Loss = 0.528256, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0179073810577393, Accuracy = 0.8036437034606934
    Iter #338432:  Learning rate = 0.004424:   Batch Loss = 0.600734, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9894013404846191, Accuracy = 0.8105263113975525
    Iter #338944:  Learning rate = 0.004424:   Batch Loss = 0.518252, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0604146718978882, Accuracy = 0.7951416969299316
    Iter #339456:  Learning rate = 0.004424:   Batch Loss = 0.850536, Accuracy = 0.8125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2159984111785889, Accuracy = 0.7429149746894836
    Iter #339968:  Learning rate = 0.004424:   Batch Loss = 0.904479, Accuracy = 0.84375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.2709070444107056, Accuracy = 0.7319837808609009
    Iter #340480:  Learning rate = 0.004424:   Batch Loss = 0.765612, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.279071569442749, Accuracy = 0.7368420958518982
    Iter #340992:  Learning rate = 0.004424:   Batch Loss = 0.654128, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1237177848815918, Accuracy = 0.768825888633728
    Iter #341504:  Learning rate = 0.004424:   Batch Loss = 0.812648, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1459977626800537, Accuracy = 0.7684210538864136
    Iter #342016:  Learning rate = 0.004424:   Batch Loss = 0.678236, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.205429196357727, Accuracy = 0.7433198094367981
    Iter #342528:  Learning rate = 0.004424:   Batch Loss = 0.762192, Accuracy = 0.875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.161719799041748, Accuracy = 0.7518218755722046
    Iter #343040:  Learning rate = 0.004424:   Batch Loss = 0.783088, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1186902523040771, Accuracy = 0.7672064900398254
    Iter #343552:  Learning rate = 0.004424:   Batch Loss = 0.560789, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1381374597549438, Accuracy = 0.7668015956878662
    Iter #344064:  Learning rate = 0.004424:   Batch Loss = 0.637334, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1316169500350952, Accuracy = 0.7704453468322754
    Iter #344576:  Learning rate = 0.004424:   Batch Loss = 0.529904, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1247336864471436, Accuracy = 0.7753036618232727
    Iter #345088:  Learning rate = 0.004424:   Batch Loss = 0.576819, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1296395063400269, Accuracy = 0.7716599106788635
    Iter #345600:  Learning rate = 0.004424:   Batch Loss = 0.491705, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.018786072731018, Accuracy = 0.8141700625419617
    Iter #346112:  Learning rate = 0.004424:   Batch Loss = 0.487245, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0614153146743774, Accuracy = 0.8016194105148315
    Iter #346624:  Learning rate = 0.004424:   Batch Loss = 0.624767, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.067143440246582, Accuracy = 0.8020243048667908
    Iter #347136:  Learning rate = 0.004424:   Batch Loss = 0.565140, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0745089054107666, Accuracy = 0.8064777255058289
    Iter #347648:  Learning rate = 0.004424:   Batch Loss = 0.491571, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0656136274337769, Accuracy = 0.8016194105148315
    Iter #348160:  Learning rate = 0.004424:   Batch Loss = 0.497454, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0060667991638184, Accuracy = 0.8157894611358643
    Iter #348672:  Learning rate = 0.004424:   Batch Loss = 0.517588, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.020798683166504, Accuracy = 0.8230769038200378
    Iter #349184:  Learning rate = 0.004424:   Batch Loss = 0.684902, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0115641355514526, Accuracy = 0.8186234831809998
    Iter #349696:  Learning rate = 0.004424:   Batch Loss = 0.571831, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0254604816436768, Accuracy = 0.8198380470275879
    Iter #350208:  Learning rate = 0.004424:   Batch Loss = 0.518582, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1229383945465088, Accuracy = 0.7882590889930725
    Iter #350720:  Learning rate = 0.004424:   Batch Loss = 0.588502, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0506699085235596, Accuracy = 0.8036437034606934
    Iter #351232:  Learning rate = 0.004424:   Batch Loss = 0.593488, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0072124004364014, Accuracy = 0.8214575052261353
    Iter #351744:  Learning rate = 0.004424:   Batch Loss = 0.801647, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.057780146598816, Accuracy = 0.8040485978126526
    Iter #352256:  Learning rate = 0.004424:   Batch Loss = 0.615077, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.042114496231079, Accuracy = 0.8072874546051025
    Iter #352768:  Learning rate = 0.004424:   Batch Loss = 0.624153, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0228402614593506, Accuracy = 0.8190283179283142
    Iter #353280:  Learning rate = 0.004424:   Batch Loss = 0.561744, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.046412706375122, Accuracy = 0.8085020184516907
    Iter #353792:  Learning rate = 0.004424:   Batch Loss = 0.482996, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.027019739151001, Accuracy = 0.8174089193344116
    Iter #354304:  Learning rate = 0.004424:   Batch Loss = 0.479679, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0037195682525635, Accuracy = 0.8287449479103088
    Iter #354816:  Learning rate = 0.004424:   Batch Loss = 0.519900, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.011885166168213, Accuracy = 0.8170040249824524
    Iter #355328:  Learning rate = 0.004424:   Batch Loss = 0.479218, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9938714504241943, Accuracy = 0.8218623399734497
    Iter #355840:  Learning rate = 0.004424:   Batch Loss = 0.490011, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9662439823150635, Accuracy = 0.8352226614952087
    Iter #356352:  Learning rate = 0.004424:   Batch Loss = 0.476293, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9986777305603027, Accuracy = 0.8226720690727234
    Iter #356864:  Learning rate = 0.004424:   Batch Loss = 0.570294, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0122473239898682, Accuracy = 0.8109311461448669
    Iter #357376:  Learning rate = 0.004424:   Batch Loss = 0.550784, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0461100339889526, Accuracy = 0.7995951175689697
    Iter #357888:  Learning rate = 0.004424:   Batch Loss = 0.778503, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0319432020187378, Accuracy = 0.8117408752441406
    Iter #358400:  Learning rate = 0.004424:   Batch Loss = 0.653743, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.047039270401001, Accuracy = 0.8056679964065552
    Iter #358912:  Learning rate = 0.004424:   Batch Loss = 0.691928, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0423177480697632, Accuracy = 0.8165991902351379
    Iter #359424:  Learning rate = 0.004424:   Batch Loss = 0.527814, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9820802807807922, Accuracy = 0.8182186484336853
    Iter #359936:  Learning rate = 0.004424:   Batch Loss = 0.463870, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9630374908447266, Accuracy = 0.8238866329193115
    Iter #360448:  Learning rate = 0.004424:   Batch Loss = 0.531210, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9526230096817017, Accuracy = 0.8307692408561707
    Iter #360960:  Learning rate = 0.004424:   Batch Loss = 0.483955, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9684048891067505, Accuracy = 0.8283400535583496
    Iter #361472:  Learning rate = 0.004424:   Batch Loss = 0.576878, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9726576209068298, Accuracy = 0.8238866329193115
    Iter #361984:  Learning rate = 0.004424:   Batch Loss = 0.467172, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.982957124710083, Accuracy = 0.8283400535583496
    Iter #362496:  Learning rate = 0.004424:   Batch Loss = 0.455857, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.947145402431488, Accuracy = 0.8380566835403442
    Iter #363008:  Learning rate = 0.004424:   Batch Loss = 0.495906, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9356993436813354, Accuracy = 0.8493927121162415
    Iter #363520:  Learning rate = 0.004424:   Batch Loss = 0.509487, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.993665337562561, Accuracy = 0.8319838047027588
    Iter #364032:  Learning rate = 0.004424:   Batch Loss = 0.489452, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9654826521873474, Accuracy = 0.8246963620185852
    Iter #364544:  Learning rate = 0.004424:   Batch Loss = 0.479506, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9703713059425354, Accuracy = 0.8319838047027588
    Iter #365056:  Learning rate = 0.004424:   Batch Loss = 0.442795, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9534486532211304, Accuracy = 0.8348178267478943
    Iter #365568:  Learning rate = 0.004424:   Batch Loss = 0.439191, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9726619720458984, Accuracy = 0.8279352188110352
    Iter #366080:  Learning rate = 0.004424:   Batch Loss = 0.464176, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9276165962219238, Accuracy = 0.8445343971252441
    Iter #366592:  Learning rate = 0.004424:   Batch Loss = 0.460548, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8988890647888184, Accuracy = 0.8558704257011414
    Iter #367104:  Learning rate = 0.004424:   Batch Loss = 0.553445, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9753104448318481, Accuracy = 0.8315789699554443
    Iter #367616:  Learning rate = 0.004424:   Batch Loss = 0.419773, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.901528537273407, Accuracy = 0.8493927121162415
    Iter #368128:  Learning rate = 0.004424:   Batch Loss = 0.435175, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8899072408676147, Accuracy = 0.8526315689086914
    Iter #368640:  Learning rate = 0.004424:   Batch Loss = 0.486813, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8806542158126831, Accuracy = 0.848987877368927
    Iter #369152:  Learning rate = 0.004424:   Batch Loss = 0.533407, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9162634611129761, Accuracy = 0.8473684191703796
    Iter #369664:  Learning rate = 0.004424:   Batch Loss = 0.524396, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9376922845840454, Accuracy = 0.8473684191703796
    Iter #370176:  Learning rate = 0.004424:   Batch Loss = 0.438491, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9474358558654785, Accuracy = 0.8380566835403442
    Iter #370688:  Learning rate = 0.004424:   Batch Loss = 0.459393, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9443776607513428, Accuracy = 0.8425101041793823
    Iter #371200:  Learning rate = 0.004424:   Batch Loss = 0.438380, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9569860696792603, Accuracy = 0.8380566835403442
    Iter #371712:  Learning rate = 0.004424:   Batch Loss = 0.423866, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9605813026428223, Accuracy = 0.840080976486206
    Iter #372224:  Learning rate = 0.004424:   Batch Loss = 0.408432, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9239287376403809, Accuracy = 0.8473684191703796
    Iter #372736:  Learning rate = 0.004424:   Batch Loss = 0.419144, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9062488079071045, Accuracy = 0.856680154800415
    Iter #373248:  Learning rate = 0.004424:   Batch Loss = 0.409183, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8962571620941162, Accuracy = 0.8550607562065125
    Iter #373760:  Learning rate = 0.004424:   Batch Loss = 0.425519, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8842176795005798, Accuracy = 0.8554655909538269
    Iter #374272:  Learning rate = 0.004424:   Batch Loss = 0.396024, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9068498611450195, Accuracy = 0.852226734161377
    Iter #374784:  Learning rate = 0.004424:   Batch Loss = 0.409816, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8931671380996704, Accuracy = 0.8526315689086914
    Iter #375296:  Learning rate = 0.004424:   Batch Loss = 0.408013, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.940041184425354, Accuracy = 0.8364372253417969
    Iter #375808:  Learning rate = 0.004424:   Batch Loss = 0.404114, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8672125339508057, Accuracy = 0.8651821613311768
    Iter #376320:  Learning rate = 0.004424:   Batch Loss = 0.404333, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8840186595916748, Accuracy = 0.8506072759628296
    Iter #376832:  Learning rate = 0.004424:   Batch Loss = 0.394342, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9384324550628662, Accuracy = 0.840080976486206
    Iter #377344:  Learning rate = 0.004424:   Batch Loss = 0.381828, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9070744514465332, Accuracy = 0.846558690071106
    Iter #377856:  Learning rate = 0.004424:   Batch Loss = 0.400250, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9067282676696777, Accuracy = 0.8388664126396179
    Iter #378368:  Learning rate = 0.004424:   Batch Loss = 0.385332, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8935010433197021, Accuracy = 0.8514170050621033
    Iter #378880:  Learning rate = 0.004424:   Batch Loss = 0.380062, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.898440957069397, Accuracy = 0.8481781482696533
    Iter #379392:  Learning rate = 0.004424:   Batch Loss = 0.435251, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8762079477310181, Accuracy = 0.8538461327552795
    Iter #379904:  Learning rate = 0.004424:   Batch Loss = 0.394438, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8777728080749512, Accuracy = 0.8534412980079651
    Iter #380416:  Learning rate = 0.004424:   Batch Loss = 0.367474, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8688442707061768, Accuracy = 0.8550607562065125
    Iter #380928:  Learning rate = 0.004424:   Batch Loss = 0.401642, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8847221732139587, Accuracy = 0.8473684191703796
    Iter #381440:  Learning rate = 0.004424:   Batch Loss = 0.377753, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8816210031509399, Accuracy = 0.8542510271072388
    Iter #381952:  Learning rate = 0.004424:   Batch Loss = 0.386280, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8582228422164917, Accuracy = 0.8611335754394531
    Iter #382464:  Learning rate = 0.004424:   Batch Loss = 0.424246, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.870702862739563, Accuracy = 0.8591092824935913
    Iter #382976:  Learning rate = 0.004424:   Batch Loss = 0.384437, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8681108355522156, Accuracy = 0.8554655909538269
    Iter #383488:  Learning rate = 0.004424:   Batch Loss = 0.362524, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.871391773223877, Accuracy = 0.8578947186470032
    Iter #384000:  Learning rate = 0.004424:   Batch Loss = 0.358240, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8560656905174255, Accuracy = 0.8574898838996887
    Iter #384512:  Learning rate = 0.004424:   Batch Loss = 0.376974, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8378716707229614, Accuracy = 0.865587055683136
    Iter #385024:  Learning rate = 0.004424:   Batch Loss = 0.365560, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8767629265785217, Accuracy = 0.8461538553237915
    Iter #385536:  Learning rate = 0.004424:   Batch Loss = 0.376246, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8397018909454346, Accuracy = 0.8635627627372742
    Iter #386048:  Learning rate = 0.004424:   Batch Loss = 0.361172, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8481276035308838, Accuracy = 0.8647773265838623
    Iter #386560:  Learning rate = 0.004424:   Batch Loss = 0.356687, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8349229693412781, Accuracy = 0.8627530336380005
    Iter #387072:  Learning rate = 0.004424:   Batch Loss = 0.373412, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8329992890357971, Accuracy = 0.865587055683136
    Iter #387584:  Learning rate = 0.004424:   Batch Loss = 0.345512, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8300981521606445, Accuracy = 0.8639675974845886
    Iter #388096:  Learning rate = 0.004424:   Batch Loss = 0.364402, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8165435791015625, Accuracy = 0.8704453706741333
    Iter #388608:  Learning rate = 0.004424:   Batch Loss = 0.344140, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8082798719406128, Accuracy = 0.8744939565658569
    Iter #389120:  Learning rate = 0.004424:   Batch Loss = 0.351665, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8218061923980713, Accuracy = 0.8696356415748596
    Iter #389632:  Learning rate = 0.004424:   Batch Loss = 0.408387, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.825415313243866, Accuracy = 0.8595141768455505
    Iter #390144:  Learning rate = 0.004424:   Batch Loss = 0.391362, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8572255969047546, Accuracy = 0.8562753200531006
    Iter #390656:  Learning rate = 0.004424:   Batch Loss = 0.343879, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8362301588058472, Accuracy = 0.8615384697914124
    Iter #391168:  Learning rate = 0.004424:   Batch Loss = 0.344693, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8164335489273071, Accuracy = 0.8680161833763123
    Iter #391680:  Learning rate = 0.004424:   Batch Loss = 0.348721, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8259398341178894, Accuracy = 0.862348198890686
    Iter #392192:  Learning rate = 0.004424:   Batch Loss = 0.375299, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8287081718444824, Accuracy = 0.862348198890686
    Iter #392704:  Learning rate = 0.004424:   Batch Loss = 0.342413, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8124328851699829, Accuracy = 0.865587055683136
    Iter #393216:  Learning rate = 0.004424:   Batch Loss = 0.345129, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8123466968536377, Accuracy = 0.8615384697914124
    Iter #393728:  Learning rate = 0.004424:   Batch Loss = 0.344997, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8113852739334106, Accuracy = 0.8615384697914124
    Iter #394240:  Learning rate = 0.004424:   Batch Loss = 0.340662, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8171975612640381, Accuracy = 0.8619433045387268
    Iter #394752:  Learning rate = 0.004424:   Batch Loss = 0.366500, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8208938837051392, Accuracy = 0.8647773265838623
    Iter #395264:  Learning rate = 0.004424:   Batch Loss = 0.342144, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8231435418128967, Accuracy = 0.8591092824935913
    Iter #395776:  Learning rate = 0.004424:   Batch Loss = 0.334013, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.80961012840271, Accuracy = 0.8611335754394531
    Iter #396288:  Learning rate = 0.004424:   Batch Loss = 0.362999, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8091781139373779, Accuracy = 0.8704453706741333
    Iter #396800:  Learning rate = 0.004424:   Batch Loss = 0.344761, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8581792116165161, Accuracy = 0.8497975468635559
    Iter #397312:  Learning rate = 0.004424:   Batch Loss = 0.355229, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.806883692741394, Accuracy = 0.8647773265838623
    Iter #397824:  Learning rate = 0.004424:   Batch Loss = 0.345354, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8629264831542969, Accuracy = 0.8526315689086914
    Iter #398336:  Learning rate = 0.004424:   Batch Loss = 0.369109, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.869048535823822, Accuracy = 0.8497975468635559
    Iter #398848:  Learning rate = 0.004424:   Batch Loss = 0.350239, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8417965173721313, Accuracy = 0.8526315689086914
    Iter #399360:  Learning rate = 0.004424:   Batch Loss = 0.367009, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8451970219612122, Accuracy = 0.8550607562065125
    Iter #399872:  Learning rate = 0.004424:   Batch Loss = 0.348010, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8713749647140503, Accuracy = 0.843319833278656
    Iter #400384:  Learning rate = 0.004247:   Batch Loss = 0.388902, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9731937646865845, Accuracy = 0.8153846263885498
    Iter #400896:  Learning rate = 0.004247:   Batch Loss = 0.500430, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9936968684196472, Accuracy = 0.807692289352417
    Iter #401408:  Learning rate = 0.004247:   Batch Loss = 0.481107, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0265147686004639, Accuracy = 0.8040485978126526
    Iter #401920:  Learning rate = 0.004247:   Batch Loss = 0.461573, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9926802515983582, Accuracy = 0.8089068531990051
    Iter #402432:  Learning rate = 0.004247:   Batch Loss = 0.507915, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0672825574874878, Accuracy = 0.7935222387313843
    Iter #402944:  Learning rate = 0.004247:   Batch Loss = 0.698445, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.058563232421875, Accuracy = 0.7923076748847961
    Iter #403456:  Learning rate = 0.004247:   Batch Loss = 0.660909, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.107590675354004, Accuracy = 0.7821862101554871
    Iter #403968:  Learning rate = 0.004247:   Batch Loss = 0.585202, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.024707555770874, Accuracy = 0.7951416969299316
    Iter #404480:  Learning rate = 0.004247:   Batch Loss = 0.499218, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9467739462852478, Accuracy = 0.8206477761268616
    Iter #404992:  Learning rate = 0.004247:   Batch Loss = 0.477957, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0015006065368652, Accuracy = 0.8125506043434143
    Iter #405504:  Learning rate = 0.004247:   Batch Loss = 0.538492, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9435508251190186, Accuracy = 0.8194332122802734
    Iter #406016:  Learning rate = 0.004247:   Batch Loss = 0.529700, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0264109373092651, Accuracy = 0.8016194105148315
    Iter #406528:  Learning rate = 0.004247:   Batch Loss = 0.544611, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0194357633590698, Accuracy = 0.810121476650238
    Iter #407040:  Learning rate = 0.004247:   Batch Loss = 0.483722, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9832237958908081, Accuracy = 0.8182186484336853
    Iter #407552:  Learning rate = 0.004247:   Batch Loss = 0.531773, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9858845472335815, Accuracy = 0.8125506043434143
    Iter #408064:  Learning rate = 0.004247:   Batch Loss = 0.445566, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9494419097900391, Accuracy = 0.8275303840637207
    Iter #408576:  Learning rate = 0.004247:   Batch Loss = 0.591338, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.957136869430542, Accuracy = 0.8352226614952087
    Iter #409088:  Learning rate = 0.004247:   Batch Loss = 0.465006, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9398735761642456, Accuracy = 0.8319838047027588
    Iter #409600:  Learning rate = 0.004247:   Batch Loss = 0.486818, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9402907490730286, Accuracy = 0.8234817981719971
    Iter #410112:  Learning rate = 0.004247:   Batch Loss = 0.480332, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9576888084411621, Accuracy = 0.8275303840637207
    Iter #410624:  Learning rate = 0.004247:   Batch Loss = 0.587193, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9668155908584595, Accuracy = 0.8246963620185852
    Iter #411136:  Learning rate = 0.004247:   Batch Loss = 0.493908, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0080196857452393, Accuracy = 0.8097165822982788
    Iter #411648:  Learning rate = 0.004247:   Batch Loss = 0.489254, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0125683546066284, Accuracy = 0.8149797320365906
    Iter #412160:  Learning rate = 0.004247:   Batch Loss = 0.544051, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9764869213104248, Accuracy = 0.8161943554878235
    Iter #412672:  Learning rate = 0.004247:   Batch Loss = 0.497806, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0003907680511475, Accuracy = 0.8028340339660645
    Iter #413184:  Learning rate = 0.004247:   Batch Loss = 0.532635, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9660407900810242, Accuracy = 0.821052610874176
    Iter #413696:  Learning rate = 0.004247:   Batch Loss = 0.452992, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8900865912437439, Accuracy = 0.8376518487930298
    Iter #414208:  Learning rate = 0.004247:   Batch Loss = 0.589343, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9116855263710022, Accuracy = 0.840080976486206
    Iter #414720:  Learning rate = 0.004247:   Batch Loss = 0.432466, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9318511486053467, Accuracy = 0.8449392914772034
    Iter #415232:  Learning rate = 0.004247:   Batch Loss = 0.438705, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9076068997383118, Accuracy = 0.8437246680259705
    Iter #415744:  Learning rate = 0.004247:   Batch Loss = 0.411590, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8714607954025269, Accuracy = 0.8582996129989624
    Iter #416256:  Learning rate = 0.004247:   Batch Loss = 0.466932, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8866037726402283, Accuracy = 0.8506072759628296
    Iter #416768:  Learning rate = 0.004247:   Batch Loss = 0.409086, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.895384669303894, Accuracy = 0.8453441262245178
    Iter #417280:  Learning rate = 0.004247:   Batch Loss = 0.419435, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8928934931755066, Accuracy = 0.8425101041793823
    Iter #417792:  Learning rate = 0.004247:   Batch Loss = 0.412595, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9018632173538208, Accuracy = 0.852226734161377
    Iter #418304:  Learning rate = 0.004247:   Batch Loss = 0.465097, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9493027925491333, Accuracy = 0.8364372253417969
    Iter #418816:  Learning rate = 0.004247:   Batch Loss = 0.531515, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9616116285324097, Accuracy = 0.826720654964447
    Iter #419328:  Learning rate = 0.004247:   Batch Loss = 0.467434, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9355182647705078, Accuracy = 0.8376518487930298
    Iter #419840:  Learning rate = 0.004247:   Batch Loss = 0.458930, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9481110572814941, Accuracy = 0.8344129323959351
    Iter #420352:  Learning rate = 0.004247:   Batch Loss = 0.418102, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9749259948730469, Accuracy = 0.8234817981719971
    Iter #420864:  Learning rate = 0.004247:   Batch Loss = 0.410101, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9233042597770691, Accuracy = 0.8392712473869324
    Iter #421376:  Learning rate = 0.004247:   Batch Loss = 0.385356, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9454336762428284, Accuracy = 0.835627555847168
    Iter #421888:  Learning rate = 0.004247:   Batch Loss = 0.428544, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9327970743179321, Accuracy = 0.8352226614952087
    Iter #422400:  Learning rate = 0.004247:   Batch Loss = 0.584956, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.900456428527832, Accuracy = 0.843319833278656
    Iter #422912:  Learning rate = 0.004247:   Batch Loss = 0.403456, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9241470694541931, Accuracy = 0.8368421196937561
    Iter #423424:  Learning rate = 0.004247:   Batch Loss = 0.405611, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9028900861740112, Accuracy = 0.8477732539176941
    Iter #423936:  Learning rate = 0.004247:   Batch Loss = 0.431875, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8792833089828491, Accuracy = 0.8493927121162415
    Iter #424448:  Learning rate = 0.004247:   Batch Loss = 0.446011, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9078953862190247, Accuracy = 0.8360323905944824
    Iter #424960:  Learning rate = 0.004247:   Batch Loss = 0.465226, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9086672067642212, Accuracy = 0.8408907055854797
    Iter #425472:  Learning rate = 0.004247:   Batch Loss = 0.375222, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9343684911727905, Accuracy = 0.8307692408561707
    Iter #425984:  Learning rate = 0.004247:   Batch Loss = 0.387088, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9254871606826782, Accuracy = 0.8417003750801086
    Iter #426496:  Learning rate = 0.004247:   Batch Loss = 0.366153, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8902319073677063, Accuracy = 0.8469635844230652
    Iter #427008:  Learning rate = 0.004247:   Batch Loss = 0.416544, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8889095783233643, Accuracy = 0.8477732539176941
    Iter #427520:  Learning rate = 0.004247:   Batch Loss = 0.409613, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9229698181152344, Accuracy = 0.829959511756897
    Iter #428032:  Learning rate = 0.004247:   Batch Loss = 0.405423, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9220318794250488, Accuracy = 0.8376518487930298
    Iter #428544:  Learning rate = 0.004247:   Batch Loss = 0.566954, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9367866516113281, Accuracy = 0.8234817981719971
    Iter #429056:  Learning rate = 0.004247:   Batch Loss = 0.438064, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9675741791725159, Accuracy = 0.8242915272712708
    Iter #429568:  Learning rate = 0.004247:   Batch Loss = 0.493211, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9133601188659668, Accuracy = 0.843319833278656
    Iter #430080:  Learning rate = 0.004247:   Batch Loss = 0.403870, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.932815432548523, Accuracy = 0.8291497826576233
    Iter #430592:  Learning rate = 0.004247:   Batch Loss = 0.446986, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.931830644607544, Accuracy = 0.8352226614952087
    Iter #431104:  Learning rate = 0.004247:   Batch Loss = 0.387894, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9090594053268433, Accuracy = 0.8412955403327942
    Iter #431616:  Learning rate = 0.004247:   Batch Loss = 0.397619, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9484384059906006, Accuracy = 0.8323886394500732
    Iter #432128:  Learning rate = 0.004247:   Batch Loss = 0.398177, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8911950588226318, Accuracy = 0.8485829830169678
    Iter #432640:  Learning rate = 0.004247:   Batch Loss = 0.428100, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.882803201675415, Accuracy = 0.8493927121162415
    Iter #433152:  Learning rate = 0.004247:   Batch Loss = 0.412898, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9069815874099731, Accuracy = 0.8408907055854797
    Iter #433664:  Learning rate = 0.004247:   Batch Loss = 0.392624, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8986721038818359, Accuracy = 0.840080976486206
    Iter #434176:  Learning rate = 0.004247:   Batch Loss = 0.382275, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8749196529388428, Accuracy = 0.8514170050621033
    Iter #434688:  Learning rate = 0.004247:   Batch Loss = 0.375946, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8755776286125183, Accuracy = 0.8481781482696533
    Iter #435200:  Learning rate = 0.004247:   Batch Loss = 0.365706, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8568358421325684, Accuracy = 0.8530364632606506
    Iter #435712:  Learning rate = 0.004247:   Batch Loss = 0.388159, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8248775005340576, Accuracy = 0.8672064542770386
    Iter #436224:  Learning rate = 0.004247:   Batch Loss = 0.379710, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8683193922042847, Accuracy = 0.8530364632606506
    Iter #436736:  Learning rate = 0.004247:   Batch Loss = 0.365665, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8596463203430176, Accuracy = 0.8607287406921387
    Iter #437248:  Learning rate = 0.004247:   Batch Loss = 0.373178, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8391038775444031, Accuracy = 0.8619433045387268
    Iter #437760:  Learning rate = 0.004247:   Batch Loss = 0.434803, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8765088319778442, Accuracy = 0.8538461327552795
    Iter #438272:  Learning rate = 0.004247:   Batch Loss = 0.400660, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9423169493675232, Accuracy = 0.8380566835403442
    Iter #438784:  Learning rate = 0.004247:   Batch Loss = 0.462041, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8922715187072754, Accuracy = 0.8388664126396179
    Iter #439296:  Learning rate = 0.004247:   Batch Loss = 0.428821, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9276984930038452, Accuracy = 0.829959511756897
    Iter #439808:  Learning rate = 0.004247:   Batch Loss = 0.427047, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9432421922683716, Accuracy = 0.8251011967658997
    Iter #440320:  Learning rate = 0.004247:   Batch Loss = 0.448216, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9168692231178284, Accuracy = 0.8283400535583496
    Iter #440832:  Learning rate = 0.004247:   Batch Loss = 0.390926, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9520247578620911, Accuracy = 0.8246963620185852
    Iter #441344:  Learning rate = 0.004247:   Batch Loss = 0.433293, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.015254259109497, Accuracy = 0.8085020184516907
    Iter #441856:  Learning rate = 0.004247:   Batch Loss = 0.443793, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0126619338989258, Accuracy = 0.8064777255058289
    Iter #442368:  Learning rate = 0.004247:   Batch Loss = 0.604822, Accuracy = 0.875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9998070597648621, Accuracy = 0.8149797320365906
    Iter #442880:  Learning rate = 0.004247:   Batch Loss = 0.453418, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9662511348724365, Accuracy = 0.8230769038200378
    Iter #443392:  Learning rate = 0.004247:   Batch Loss = 0.434337, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9512420892715454, Accuracy = 0.8348178267478943
    Iter #443904:  Learning rate = 0.004247:   Batch Loss = 0.543703, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.070814847946167, Accuracy = 0.7963562607765198
    Iter #444416:  Learning rate = 0.004247:   Batch Loss = 0.468839, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9991804361343384, Accuracy = 0.8085020184516907
    Iter #444928:  Learning rate = 0.004247:   Batch Loss = 0.691998, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9956021308898926, Accuracy = 0.8125506043434143
    Iter #445440:  Learning rate = 0.004247:   Batch Loss = 0.485823, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0212732553482056, Accuracy = 0.8040485978126526
    Iter #445952:  Learning rate = 0.004247:   Batch Loss = 0.482319, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0210487842559814, Accuracy = 0.8109311461448669
    Iter #446464:  Learning rate = 0.004247:   Batch Loss = 0.524835, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9787307977676392, Accuracy = 0.8093117475509644
    Iter #446976:  Learning rate = 0.004247:   Batch Loss = 0.399776, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9433467388153076, Accuracy = 0.8198380470275879
    Iter #447488:  Learning rate = 0.004247:   Batch Loss = 0.460762, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9637855291366577, Accuracy = 0.8259109258651733
    Iter #448000:  Learning rate = 0.004247:   Batch Loss = 0.569091, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9875705242156982, Accuracy = 0.8121457695960999
    Iter #448512:  Learning rate = 0.004247:   Batch Loss = 0.574851, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.955741286277771, Accuracy = 0.8307692408561707
    Iter #449024:  Learning rate = 0.004247:   Batch Loss = 0.467566, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9506199359893799, Accuracy = 0.8263157606124878
    Iter #449536:  Learning rate = 0.004247:   Batch Loss = 0.464089, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9362286925315857, Accuracy = 0.8344129323959351
    Iter #450048:  Learning rate = 0.004247:   Batch Loss = 0.401376, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.879759669303894, Accuracy = 0.8485829830169678
    Iter #450560:  Learning rate = 0.004247:   Batch Loss = 0.370031, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9171731472015381, Accuracy = 0.8352226614952087
    Iter #451072:  Learning rate = 0.004247:   Batch Loss = 0.400338, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8789665102958679, Accuracy = 0.8506072759628296
    Iter #451584:  Learning rate = 0.004247:   Batch Loss = 0.383793, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9112236499786377, Accuracy = 0.8380566835403442
    Iter #452096:  Learning rate = 0.004247:   Batch Loss = 0.410262, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8736909627914429, Accuracy = 0.8485829830169678
    Iter #452608:  Learning rate = 0.004247:   Batch Loss = 0.394341, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9178493618965149, Accuracy = 0.8421052694320679
    Iter #453120:  Learning rate = 0.004247:   Batch Loss = 0.442712, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8945838212966919, Accuracy = 0.8445343971252441
    Iter #453632:  Learning rate = 0.004247:   Batch Loss = 0.388402, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8765257596969604, Accuracy = 0.8477732539176941
    Iter #454144:  Learning rate = 0.004247:   Batch Loss = 0.418013, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8809977769851685, Accuracy = 0.8502024412155151
    Iter #454656:  Learning rate = 0.004247:   Batch Loss = 0.361883, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8845382928848267, Accuracy = 0.8445343971252441
    Iter #455168:  Learning rate = 0.004247:   Batch Loss = 0.408565, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8862057328224182, Accuracy = 0.8457489609718323
    Iter #455680:  Learning rate = 0.004247:   Batch Loss = 0.374301, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8566770553588867, Accuracy = 0.8550607562065125
    Iter #456192:  Learning rate = 0.004247:   Batch Loss = 0.365489, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8542734384536743, Accuracy = 0.8518218398094177
    Iter #456704:  Learning rate = 0.004247:   Batch Loss = 0.362733, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8600630164146423, Accuracy = 0.8554655909538269
    Iter #457216:  Learning rate = 0.004247:   Batch Loss = 0.379695, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8357345461845398, Accuracy = 0.8603239059448242
    Iter #457728:  Learning rate = 0.004247:   Batch Loss = 0.357218, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.821500837802887, Accuracy = 0.8639675974845886
    Iter #458240:  Learning rate = 0.004247:   Batch Loss = 0.350939, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8054896593093872, Accuracy = 0.865587055683136
    Iter #458752:  Learning rate = 0.004247:   Batch Loss = 0.349337, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8224581480026245, Accuracy = 0.8651821613311768
    Iter #459264:  Learning rate = 0.004247:   Batch Loss = 0.338312, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8055952787399292, Accuracy = 0.8668016195297241
    Iter #459776:  Learning rate = 0.004247:   Batch Loss = 0.343982, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7995074987411499, Accuracy = 0.8663967847824097
    Iter #460288:  Learning rate = 0.004247:   Batch Loss = 0.343882, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7878198623657227, Accuracy = 0.8688259124755859
    Iter #460800:  Learning rate = 0.004247:   Batch Loss = 0.333650, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7703734636306763, Accuracy = 0.8716599345207214
    Iter #461312:  Learning rate = 0.004247:   Batch Loss = 0.367072, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7722742557525635, Accuracy = 0.873279333114624
    Iter #461824:  Learning rate = 0.004247:   Batch Loss = 0.352349, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7776000499725342, Accuracy = 0.8712550401687622
    Iter #462336:  Learning rate = 0.004247:   Batch Loss = 0.326301, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7697950005531311, Accuracy = 0.8753036260604858
    Iter #462848:  Learning rate = 0.004247:   Batch Loss = 0.327526, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.772391676902771, Accuracy = 0.8785424828529358
    Iter #463360:  Learning rate = 0.004247:   Batch Loss = 0.315752, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7703367471694946, Accuracy = 0.8797571063041687
    Iter #463872:  Learning rate = 0.004247:   Batch Loss = 0.323671, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7640265226364136, Accuracy = 0.878947377204895
    Iter #464384:  Learning rate = 0.004247:   Batch Loss = 0.323219, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7591100931167603, Accuracy = 0.8724696636199951
    Iter #464896:  Learning rate = 0.004247:   Batch Loss = 0.317795, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7573158740997314, Accuracy = 0.873279333114624
    Iter #465408:  Learning rate = 0.004247:   Batch Loss = 0.311102, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7595708966255188, Accuracy = 0.8769230842590332
    Iter #465920:  Learning rate = 0.004247:   Batch Loss = 0.320361, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7509466409683228, Accuracy = 0.8744939565658569
    Iter #466432:  Learning rate = 0.004247:   Batch Loss = 0.314718, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7502342462539673, Accuracy = 0.8785424828529358
    Iter #466944:  Learning rate = 0.004247:   Batch Loss = 0.318263, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7433396577835083, Accuracy = 0.8773279190063477
    Iter #467456:  Learning rate = 0.004247:   Batch Loss = 0.315629, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7336370944976807, Accuracy = 0.8805667757987976
    Iter #467968:  Learning rate = 0.004247:   Batch Loss = 0.317299, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7330269813537598, Accuracy = 0.8846153616905212
    Iter #468480:  Learning rate = 0.004247:   Batch Loss = 0.311040, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7306572198867798, Accuracy = 0.8813765048980713
    Iter #468992:  Learning rate = 0.004247:   Batch Loss = 0.310488, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7311428785324097, Accuracy = 0.8829959630966187
    Iter #469504:  Learning rate = 0.004247:   Batch Loss = 0.308709, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7327255010604858, Accuracy = 0.8838056921958923
    Iter #470016:  Learning rate = 0.004247:   Batch Loss = 0.303791, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7291254997253418, Accuracy = 0.8850202560424805
    Iter #470528:  Learning rate = 0.004247:   Batch Loss = 0.307411, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7334864139556885, Accuracy = 0.8805667757987976
    Iter #471040:  Learning rate = 0.004247:   Batch Loss = 0.304216, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.731699526309967, Accuracy = 0.8773279190063477
    Iter #471552:  Learning rate = 0.004247:   Batch Loss = 0.301062, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7350315451622009, Accuracy = 0.8744939565658569
    Iter #472064:  Learning rate = 0.004247:   Batch Loss = 0.297753, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7319093942642212, Accuracy = 0.8793522119522095
    Iter #472576:  Learning rate = 0.004247:   Batch Loss = 0.295616, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.73032146692276, Accuracy = 0.8793522119522095
    Iter #473088:  Learning rate = 0.004247:   Batch Loss = 0.301321, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7288590669631958, Accuracy = 0.8805667757987976
    Iter #473600:  Learning rate = 0.004247:   Batch Loss = 0.295738, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7245368957519531, Accuracy = 0.8797571063041687
    Iter #474112:  Learning rate = 0.004247:   Batch Loss = 0.292904, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7312518358230591, Accuracy = 0.8781376481056213
    Iter #474624:  Learning rate = 0.004247:   Batch Loss = 0.301146, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7378551363945007, Accuracy = 0.8728744983673096
    Iter #475136:  Learning rate = 0.004247:   Batch Loss = 0.300693, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7318992018699646, Accuracy = 0.8736842274665833
    Iter #475648:  Learning rate = 0.004247:   Batch Loss = 0.292305, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7304244041442871, Accuracy = 0.8716599345207214
    Iter #476160:  Learning rate = 0.004247:   Batch Loss = 0.290071, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7267853617668152, Accuracy = 0.8769230842590332
    Iter #476672:  Learning rate = 0.004247:   Batch Loss = 0.281599, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7295745611190796, Accuracy = 0.8773279190063477
    Iter #477184:  Learning rate = 0.004247:   Batch Loss = 0.285286, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7200229167938232, Accuracy = 0.8801619410514832
    Iter #477696:  Learning rate = 0.004247:   Batch Loss = 0.290667, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7144140005111694, Accuracy = 0.8773279190063477
    Iter #478208:  Learning rate = 0.004247:   Batch Loss = 0.286448, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7181574106216431, Accuracy = 0.878947377204895
    Iter #478720:  Learning rate = 0.004247:   Batch Loss = 0.285964, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7171542644500732, Accuracy = 0.8785424828529358
    Iter #479232:  Learning rate = 0.004247:   Batch Loss = 0.289732, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7136094570159912, Accuracy = 0.8809716701507568
    Iter #479744:  Learning rate = 0.004247:   Batch Loss = 0.285158, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.724098801612854, Accuracy = 0.8728744983673096
    Iter #480256:  Learning rate = 0.004247:   Batch Loss = 0.287142, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7179343700408936, Accuracy = 0.873279333114624
    Iter #480768:  Learning rate = 0.004247:   Batch Loss = 0.282923, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7172360420227051, Accuracy = 0.8744939565658569
    Iter #481280:  Learning rate = 0.004247:   Batch Loss = 0.284514, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7142056226730347, Accuracy = 0.873279333114624
    Iter #481792:  Learning rate = 0.004247:   Batch Loss = 0.284102, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7128819227218628, Accuracy = 0.8769230842590332
    Iter #482304:  Learning rate = 0.004247:   Batch Loss = 0.281181, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7199099063873291, Accuracy = 0.8748987913131714
    Iter #482816:  Learning rate = 0.004247:   Batch Loss = 0.280738, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7196599245071411, Accuracy = 0.8720647692680359
    Iter #483328:  Learning rate = 0.004247:   Batch Loss = 0.279733, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7068909406661987, Accuracy = 0.876518189907074
    Iter #483840:  Learning rate = 0.004247:   Batch Loss = 0.277048, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7093510031700134, Accuracy = 0.8813765048980713
    Iter #484352:  Learning rate = 0.004247:   Batch Loss = 0.275781, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7094393968582153, Accuracy = 0.8813765048980713
    Iter #484864:  Learning rate = 0.004247:   Batch Loss = 0.279777, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7108491063117981, Accuracy = 0.8781376481056213
    Iter #485376:  Learning rate = 0.004247:   Batch Loss = 0.278243, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7102649211883545, Accuracy = 0.8801619410514832
    Iter #485888:  Learning rate = 0.004247:   Batch Loss = 0.273005, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7084256410598755, Accuracy = 0.8748987913131714
    Iter #486400:  Learning rate = 0.004247:   Batch Loss = 0.277909, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7069318890571594, Accuracy = 0.8736842274665833
    Iter #486912:  Learning rate = 0.004247:   Batch Loss = 0.271485, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7117388844490051, Accuracy = 0.8700404763221741
    Iter #487424:  Learning rate = 0.004247:   Batch Loss = 0.271391, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7166793942451477, Accuracy = 0.8692307472229004
    Iter #487936:  Learning rate = 0.004247:   Batch Loss = 0.277393, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7067890167236328, Accuracy = 0.8753036260604858
    Iter #488448:  Learning rate = 0.004247:   Batch Loss = 0.273192, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7079997062683105, Accuracy = 0.876518189907074
    Iter #488960:  Learning rate = 0.004247:   Batch Loss = 0.267335, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7049100995063782, Accuracy = 0.8757085204124451
    Iter #489472:  Learning rate = 0.004247:   Batch Loss = 0.273234, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7027318477630615, Accuracy = 0.8761133551597595
    Iter #489984:  Learning rate = 0.004247:   Batch Loss = 0.264747, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7002891302108765, Accuracy = 0.878947377204895
    Iter #490496:  Learning rate = 0.004247:   Batch Loss = 0.262435, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7111892700195312, Accuracy = 0.8761133551597595
    Iter #491008:  Learning rate = 0.004247:   Batch Loss = 0.266490, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7034366726875305, Accuracy = 0.8769230842590332
    Iter #491520:  Learning rate = 0.004247:   Batch Loss = 0.273643, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7097873687744141, Accuracy = 0.8696356415748596
    Iter #492032:  Learning rate = 0.004247:   Batch Loss = 0.267614, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7133756279945374, Accuracy = 0.873279333114624
    Iter #492544:  Learning rate = 0.004247:   Batch Loss = 0.267329, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7084590792655945, Accuracy = 0.8712550401687622
    Iter #493056:  Learning rate = 0.004247:   Batch Loss = 0.275160, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7116050720214844, Accuracy = 0.873279333114624
    Iter #493568:  Learning rate = 0.004247:   Batch Loss = 0.261130, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7133979797363281, Accuracy = 0.8720647692680359
    Iter #494080:  Learning rate = 0.004247:   Batch Loss = 0.258397, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7049367427825928, Accuracy = 0.8773279190063477
    Iter #494592:  Learning rate = 0.004247:   Batch Loss = 0.264871, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7092781662940979, Accuracy = 0.8748987913131714
    Iter #495104:  Learning rate = 0.004247:   Batch Loss = 0.262000, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7214134931564331, Accuracy = 0.8692307472229004
    Iter #495616:  Learning rate = 0.004247:   Batch Loss = 0.257589, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7208959460258484, Accuracy = 0.8663967847824097
    Iter #496128:  Learning rate = 0.004247:   Batch Loss = 0.282296, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7174409627914429, Accuracy = 0.8716599345207214
    Iter #496640:  Learning rate = 0.004247:   Batch Loss = 0.339157, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8076707124710083, Accuracy = 0.8437246680259705
    Iter #497152:  Learning rate = 0.004247:   Batch Loss = 0.341528, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8555126190185547, Accuracy = 0.8303643465042114
    Iter #497664:  Learning rate = 0.004247:   Batch Loss = 0.504687, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9147456884384155, Accuracy = 0.8085020184516907
    Iter #498176:  Learning rate = 0.004247:   Batch Loss = 0.636798, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0463013648986816, Accuracy = 0.7635627388954163
    Iter #498688:  Learning rate = 0.004247:   Batch Loss = 0.688287, Accuracy = 0.84375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9570355415344238, Accuracy = 0.7931174039840698
    Iter #499200:  Learning rate = 0.004247:   Batch Loss = 0.569228, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0157537460327148, Accuracy = 0.7740890979766846
    Iter #499712:  Learning rate = 0.004247:   Batch Loss = 0.840456, Accuracy = 0.859375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0402615070343018, Accuracy = 0.7765182256698608
    Iter #500224:  Learning rate = 0.004077:   Batch Loss = 0.633743, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0815633535385132, Accuracy = 0.7599190473556519
    Iter #500736:  Learning rate = 0.004077:   Batch Loss = 0.631457, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0899101495742798, Accuracy = 0.752226710319519
    Iter #501248:  Learning rate = 0.004077:   Batch Loss = 0.500722, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.029130458831787, Accuracy = 0.7753036618232727
    Iter #501760:  Learning rate = 0.004077:   Batch Loss = 0.810799, Accuracy = 0.8125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0237962007522583, Accuracy = 0.7959514260292053
    Iter #502272:  Learning rate = 0.004077:   Batch Loss = 0.571071, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0588269233703613, Accuracy = 0.7655870318412781
    Iter #502784:  Learning rate = 0.004077:   Batch Loss = 0.550520, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.041515827178955, Accuracy = 0.7781376242637634
    Iter #503296:  Learning rate = 0.004077:   Batch Loss = 0.766745, Accuracy = 0.875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.994389533996582, Accuracy = 0.7935222387313843
    Iter #503808:  Learning rate = 0.004077:   Batch Loss = 0.511957, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0510272979736328, Accuracy = 0.7797570824623108
    Iter #504320:  Learning rate = 0.004077:   Batch Loss = 0.623622, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0816874504089355, Accuracy = 0.7797570824623108
    Iter #504832:  Learning rate = 0.004077:   Batch Loss = 0.666093, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0114474296569824, Accuracy = 0.7959514260292053
    Iter #505344:  Learning rate = 0.004077:   Batch Loss = 0.592687, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9667838215827942, Accuracy = 0.807692289352417
    Iter #505856:  Learning rate = 0.004077:   Batch Loss = 0.726697, Accuracy = 0.875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1126536130905151, Accuracy = 0.7530364394187927
    Iter #506368:  Learning rate = 0.004077:   Batch Loss = 0.741767, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0111148357391357, Accuracy = 0.8016194105148315
    Iter #506880:  Learning rate = 0.004077:   Batch Loss = 0.534018, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9945718050003052, Accuracy = 0.7919028401374817
    Iter #507392:  Learning rate = 0.004077:   Batch Loss = 0.553428, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0470101833343506, Accuracy = 0.7951416969299316
    Iter #507904:  Learning rate = 0.004077:   Batch Loss = 0.531084, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0218695402145386, Accuracy = 0.7886639833450317
    Iter #508416:  Learning rate = 0.004077:   Batch Loss = 0.605810, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0167299509048462, Accuracy = 0.8020243048667908
    Iter #508928:  Learning rate = 0.004077:   Batch Loss = 0.665210, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0645205974578857, Accuracy = 0.7838056683540344
    Iter #509440:  Learning rate = 0.004077:   Batch Loss = 0.533041, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0424152612686157, Accuracy = 0.7902833819389343
    Iter #509952:  Learning rate = 0.004077:   Batch Loss = 0.529570, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9548870325088501, Accuracy = 0.8028340339660645
    Iter #510464:  Learning rate = 0.004077:   Batch Loss = 0.545959, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9446580410003662, Accuracy = 0.8068826198577881
    Iter #510976:  Learning rate = 0.004077:   Batch Loss = 0.552274, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9188857078552246, Accuracy = 0.821052610874176
    Iter #511488:  Learning rate = 0.004077:   Batch Loss = 0.547865, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9036061763763428, Accuracy = 0.8218623399734497
    Iter #512000:  Learning rate = 0.004077:   Batch Loss = 0.413453, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9147021174430847, Accuracy = 0.8218623399734497
    Iter #512512:  Learning rate = 0.004077:   Batch Loss = 0.422101, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9208749532699585, Accuracy = 0.8340080976486206
    Iter #513024:  Learning rate = 0.004077:   Batch Loss = 0.446450, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.865320086479187, Accuracy = 0.8493927121162415
    Iter #513536:  Learning rate = 0.004077:   Batch Loss = 0.383332, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8633266091346741, Accuracy = 0.843319833278656
    Iter #514048:  Learning rate = 0.004077:   Batch Loss = 0.364882, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.84514319896698, Accuracy = 0.8534412980079651
    Iter #514560:  Learning rate = 0.004077:   Batch Loss = 0.390325, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8650644421577454, Accuracy = 0.8421052694320679
    Iter #515072:  Learning rate = 0.004077:   Batch Loss = 0.443229, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8572050333023071, Accuracy = 0.8384615182876587
    Iter #515584:  Learning rate = 0.004077:   Batch Loss = 0.384838, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8666403293609619, Accuracy = 0.848987877368927
    Iter #516096:  Learning rate = 0.004077:   Batch Loss = 0.384071, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8653013706207275, Accuracy = 0.840080976486206
    Iter #516608:  Learning rate = 0.004077:   Batch Loss = 0.464570, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8705877065658569, Accuracy = 0.8417003750801086
    Iter #517120:  Learning rate = 0.004077:   Batch Loss = 0.387449, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.859021008014679, Accuracy = 0.8408907055854797
    Iter #517632:  Learning rate = 0.004077:   Batch Loss = 0.475595, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8573263883590698, Accuracy = 0.8429149985313416
    Iter #518144:  Learning rate = 0.004077:   Batch Loss = 0.485273, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8838244676589966, Accuracy = 0.8360323905944824
    Iter #518656:  Learning rate = 0.004077:   Batch Loss = 0.394812, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8574075698852539, Accuracy = 0.8384615182876587
    Iter #519168:  Learning rate = 0.004077:   Batch Loss = 0.405198, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8744000196456909, Accuracy = 0.8376518487930298
    Iter #519680:  Learning rate = 0.004077:   Batch Loss = 0.376523, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8628218770027161, Accuracy = 0.8514170050621033
    Iter #520192:  Learning rate = 0.004077:   Batch Loss = 0.357902, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.849694013595581, Accuracy = 0.8595141768455505
    Iter #520704:  Learning rate = 0.004077:   Batch Loss = 0.481996, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8946593999862671, Accuracy = 0.8421052694320679
    Iter #521216:  Learning rate = 0.004077:   Batch Loss = 0.427529, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8612633943557739, Accuracy = 0.8493927121162415
    Iter #521728:  Learning rate = 0.004077:   Batch Loss = 0.348089, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8160991668701172, Accuracy = 0.8607287406921387
    Iter #522240:  Learning rate = 0.004077:   Batch Loss = 0.334597, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8148449659347534, Accuracy = 0.8562753200531006
    Iter #522752:  Learning rate = 0.004077:   Batch Loss = 0.346158, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.799070417881012, Accuracy = 0.8716599345207214
    Iter #523264:  Learning rate = 0.004077:   Batch Loss = 0.336730, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8587307929992676, Accuracy = 0.8534412980079651
    Iter #523776:  Learning rate = 0.004077:   Batch Loss = 0.360330, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8287509083747864, Accuracy = 0.8514170050621033
    Iter #524288:  Learning rate = 0.004077:   Batch Loss = 0.338291, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8446685075759888, Accuracy = 0.8469635844230652
    Iter #524800:  Learning rate = 0.004077:   Batch Loss = 0.343073, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8622217178344727, Accuracy = 0.8481781482696533
    Iter #525312:  Learning rate = 0.004077:   Batch Loss = 0.375333, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.852615237236023, Accuracy = 0.852226734161377
    Iter #525824:  Learning rate = 0.004077:   Batch Loss = 0.336888, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8281556963920593, Accuracy = 0.8582996129989624
    Iter #526336:  Learning rate = 0.004077:   Batch Loss = 0.429954, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7894459962844849, Accuracy = 0.8647773265838623
    Iter #526848:  Learning rate = 0.004077:   Batch Loss = 0.339113, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8391488790512085, Accuracy = 0.846558690071106
    Iter #527360:  Learning rate = 0.004077:   Batch Loss = 0.362166, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.974912166595459, Accuracy = 0.8157894611358643
    Iter #527872:  Learning rate = 0.004077:   Batch Loss = 0.370033, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8241985440254211, Accuracy = 0.8481781482696533
    Iter #528384:  Learning rate = 0.004077:   Batch Loss = 0.419771, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8256375789642334, Accuracy = 0.8530364632606506
    Iter #528896:  Learning rate = 0.004077:   Batch Loss = 0.354933, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8491480350494385, Accuracy = 0.840080976486206
    Iter #529408:  Learning rate = 0.004077:   Batch Loss = 0.382492, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.853866457939148, Accuracy = 0.8396761417388916
    Iter #529920:  Learning rate = 0.004077:   Batch Loss = 0.486227, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9355783462524414, Accuracy = 0.8242915272712708
    Iter #530432:  Learning rate = 0.004077:   Batch Loss = 0.391751, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9567803144454956, Accuracy = 0.8259109258651733
    Iter #530944:  Learning rate = 0.004077:   Batch Loss = 0.372168, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.891573429107666, Accuracy = 0.8340080976486206
    Iter #531456:  Learning rate = 0.004077:   Batch Loss = 0.416624, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8627914786338806, Accuracy = 0.8477732539176941
    Iter #531968:  Learning rate = 0.004077:   Batch Loss = 0.400991, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8735674619674683, Accuracy = 0.8368421196937561
    Iter #532480:  Learning rate = 0.004077:   Batch Loss = 0.466605, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8831381797790527, Accuracy = 0.8453441262245178
    Iter #532992:  Learning rate = 0.004077:   Batch Loss = 0.437198, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0030311346054077, Accuracy = 0.794331967830658
    Iter #533504:  Learning rate = 0.004077:   Batch Loss = 0.401290, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9051326513290405, Accuracy = 0.8259109258651733
    Iter #534016:  Learning rate = 0.004077:   Batch Loss = 0.429364, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9422644376754761, Accuracy = 0.835627555847168
    Iter #534528:  Learning rate = 0.004077:   Batch Loss = 0.440879, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9172713160514832, Accuracy = 0.8198380470275879
    Iter #535040:  Learning rate = 0.004077:   Batch Loss = 0.438319, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8688316345214844, Accuracy = 0.8372469544410706
    Iter #535552:  Learning rate = 0.004077:   Batch Loss = 0.505303, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.899003267288208, Accuracy = 0.8242915272712708
    Iter #536064:  Learning rate = 0.004077:   Batch Loss = 0.458894, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.915076494216919, Accuracy = 0.8315789699554443
    Iter #536576:  Learning rate = 0.004077:   Batch Loss = 0.515795, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9910626411437988, Accuracy = 0.8141700625419617
    Iter #537088:  Learning rate = 0.004077:   Batch Loss = 0.352727, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8819465637207031, Accuracy = 0.8352226614952087
    Iter #537600:  Learning rate = 0.004077:   Batch Loss = 0.432066, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8811864852905273, Accuracy = 0.8376518487930298
    Iter #538112:  Learning rate = 0.004077:   Batch Loss = 0.372475, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8752646446228027, Accuracy = 0.8380566835403442
    Iter #538624:  Learning rate = 0.004077:   Batch Loss = 0.422086, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8998948335647583, Accuracy = 0.8323886394500732
    Iter #539136:  Learning rate = 0.004077:   Batch Loss = 0.377077, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8604769110679626, Accuracy = 0.8457489609718323
    Iter #539648:  Learning rate = 0.004077:   Batch Loss = 0.378474, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8732361793518066, Accuracy = 0.8408907055854797
    Iter #540160:  Learning rate = 0.004077:   Batch Loss = 0.346249, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8823692798614502, Accuracy = 0.8404858112335205
    Iter #540672:  Learning rate = 0.004077:   Batch Loss = 0.372937, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8758158683776855, Accuracy = 0.8473684191703796
    Iter #541184:  Learning rate = 0.004077:   Batch Loss = 0.392278, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8392003774642944, Accuracy = 0.8473684191703796
    Iter #541696:  Learning rate = 0.004077:   Batch Loss = 0.363656, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8174329996109009, Accuracy = 0.8595141768455505
    Iter #542208:  Learning rate = 0.004077:   Batch Loss = 0.356082, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8636394143104553, Accuracy = 0.8352226614952087
    Iter #542720:  Learning rate = 0.004077:   Batch Loss = 0.358596, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8262900114059448, Accuracy = 0.8546558618545532
    Iter #543232:  Learning rate = 0.004077:   Batch Loss = 0.385198, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8009505867958069, Accuracy = 0.8615384697914124
    Iter #543744:  Learning rate = 0.004077:   Batch Loss = 0.369895, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8595844507217407, Accuracy = 0.8392712473869324
    Iter #544256:  Learning rate = 0.004077:   Batch Loss = 0.358383, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8497804403305054, Accuracy = 0.8421052694320679
    Iter #544768:  Learning rate = 0.004077:   Batch Loss = 0.347545, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8231844902038574, Accuracy = 0.8493927121162415
    Iter #545280:  Learning rate = 0.004077:   Batch Loss = 0.416568, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8458510637283325, Accuracy = 0.8449392914772034
    Iter #545792:  Learning rate = 0.004077:   Batch Loss = 0.332887, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8260726928710938, Accuracy = 0.8546558618545532
    Iter #546304:  Learning rate = 0.004077:   Batch Loss = 0.339769, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8037933707237244, Accuracy = 0.8659918904304504
    Iter #546816:  Learning rate = 0.004077:   Batch Loss = 0.319991, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7854591608047485, Accuracy = 0.8676113486289978
    Iter #547328:  Learning rate = 0.004077:   Batch Loss = 0.322254, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7732581496238708, Accuracy = 0.8712550401687622
    Iter #547840:  Learning rate = 0.004077:   Batch Loss = 0.327196, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7598258852958679, Accuracy = 0.8736842274665833
    Iter #548352:  Learning rate = 0.004077:   Batch Loss = 0.319636, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7533547282218933, Accuracy = 0.8761133551597595
    Iter #548864:  Learning rate = 0.004077:   Batch Loss = 0.306557, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7544915676116943, Accuracy = 0.8757085204124451
    Iter #549376:  Learning rate = 0.004077:   Batch Loss = 0.310354, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7564959526062012, Accuracy = 0.8728744983673096
    Iter #549888:  Learning rate = 0.004077:   Batch Loss = 0.311543, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7460289001464844, Accuracy = 0.8773279190063477
    Iter #550400:  Learning rate = 0.004077:   Batch Loss = 0.296307, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7378143072128296, Accuracy = 0.8797571063041687
    Iter #550912:  Learning rate = 0.004077:   Batch Loss = 0.297934, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7320829629898071, Accuracy = 0.8785424828529358
    Iter #551424:  Learning rate = 0.004077:   Batch Loss = 0.291790, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7275932431221008, Accuracy = 0.8797571063041687
    Iter #551936:  Learning rate = 0.004077:   Batch Loss = 0.297105, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7329884767532349, Accuracy = 0.8785424828529358
    Iter #552448:  Learning rate = 0.004077:   Batch Loss = 0.294079, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7321609854698181, Accuracy = 0.8801619410514832
    Iter #552960:  Learning rate = 0.004077:   Batch Loss = 0.290899, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7308269739151001, Accuracy = 0.8761133551597595
    Iter #553472:  Learning rate = 0.004077:   Batch Loss = 0.288554, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7217152714729309, Accuracy = 0.8785424828529358
    Iter #553984:  Learning rate = 0.004077:   Batch Loss = 0.285603, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7161266207695007, Accuracy = 0.8785424828529358
    Iter #554496:  Learning rate = 0.004077:   Batch Loss = 0.284348, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7127292156219482, Accuracy = 0.8797571063041687
    Iter #555008:  Learning rate = 0.004077:   Batch Loss = 0.288454, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.711232602596283, Accuracy = 0.8797571063041687
    Iter #555520:  Learning rate = 0.004077:   Batch Loss = 0.281811, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7094242572784424, Accuracy = 0.8817813992500305
    Iter #556032:  Learning rate = 0.004077:   Batch Loss = 0.283186, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.709842324256897, Accuracy = 0.8785424828529358
    Iter #556544:  Learning rate = 0.004077:   Batch Loss = 0.276508, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7108458280563354, Accuracy = 0.8769230842590332
    Iter #557056:  Learning rate = 0.004077:   Batch Loss = 0.280871, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7089085578918457, Accuracy = 0.8777328133583069
    Iter #557568:  Learning rate = 0.004077:   Batch Loss = 0.273289, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7065662741661072, Accuracy = 0.8769230842590332
    Iter #558080:  Learning rate = 0.004077:   Batch Loss = 0.273416, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7073032855987549, Accuracy = 0.8777328133583069
    Iter #558592:  Learning rate = 0.004077:   Batch Loss = 0.272652, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7050309181213379, Accuracy = 0.8809716701507568
    Iter #559104:  Learning rate = 0.004077:   Batch Loss = 0.272353, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7113813161849976, Accuracy = 0.8769230842590332
    Iter #559616:  Learning rate = 0.004077:   Batch Loss = 0.271595, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7090163230895996, Accuracy = 0.8805667757987976
    Iter #560128:  Learning rate = 0.004077:   Batch Loss = 0.271646, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7221219539642334, Accuracy = 0.8781376481056213
    Iter #560640:  Learning rate = 0.004077:   Batch Loss = 0.268657, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7033337950706482, Accuracy = 0.8773279190063477
    Iter #561152:  Learning rate = 0.004077:   Batch Loss = 0.273540, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7094773054122925, Accuracy = 0.8753036260604858
    Iter #561664:  Learning rate = 0.004077:   Batch Loss = 0.268065, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7063950896263123, Accuracy = 0.8785424828529358
    Iter #562176:  Learning rate = 0.004077:   Batch Loss = 0.265196, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7005293369293213, Accuracy = 0.876518189907074
    Iter #562688:  Learning rate = 0.004077:   Batch Loss = 0.262960, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7072543501853943, Accuracy = 0.8785424828529358
    Iter #563200:  Learning rate = 0.004077:   Batch Loss = 0.267217, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6954082250595093, Accuracy = 0.876518189907074
    Iter #563712:  Learning rate = 0.004077:   Batch Loss = 0.267549, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6896581649780273, Accuracy = 0.878947377204895
    Iter #564224:  Learning rate = 0.004077:   Batch Loss = 0.260955, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6936594247817993, Accuracy = 0.8801619410514832
    Iter #564736:  Learning rate = 0.004077:   Batch Loss = 0.257330, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.683647632598877, Accuracy = 0.882186233997345
    Iter #565248:  Learning rate = 0.004077:   Batch Loss = 0.266911, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6850357055664062, Accuracy = 0.8773279190063477
    Iter #565760:  Learning rate = 0.004077:   Batch Loss = 0.259413, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.692096471786499, Accuracy = 0.8781376481056213
    Iter #566272:  Learning rate = 0.004077:   Batch Loss = 0.255928, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6807011365890503, Accuracy = 0.8797571063041687
    Iter #566784:  Learning rate = 0.004077:   Batch Loss = 0.255041, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6842004060745239, Accuracy = 0.8805667757987976
    Iter #567296:  Learning rate = 0.004077:   Batch Loss = 0.260984, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6900376677513123, Accuracy = 0.8797571063041687
    Iter #567808:  Learning rate = 0.004077:   Batch Loss = 0.256751, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6842965483665466, Accuracy = 0.8769230842590332
    Iter #568320:  Learning rate = 0.004077:   Batch Loss = 0.253982, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6867748498916626, Accuracy = 0.8757085204124451
    Iter #568832:  Learning rate = 0.004077:   Batch Loss = 0.253648, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6787393093109131, Accuracy = 0.8777328133583069
    Iter #569344:  Learning rate = 0.004077:   Batch Loss = 0.255124, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.676833987236023, Accuracy = 0.8769230842590332
    Iter #569856:  Learning rate = 0.004077:   Batch Loss = 0.256620, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6844736933708191, Accuracy = 0.8769230842590332
    Iter #570368:  Learning rate = 0.004077:   Batch Loss = 0.251373, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.679649829864502, Accuracy = 0.8744939565658569
    Iter #570880:  Learning rate = 0.004077:   Batch Loss = 0.247914, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6862822771072388, Accuracy = 0.878947377204895
    Iter #571392:  Learning rate = 0.004077:   Batch Loss = 0.275156, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6943258047103882, Accuracy = 0.8720647692680359
    Iter #571904:  Learning rate = 0.004077:   Batch Loss = 0.255389, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6982403993606567, Accuracy = 0.8708502054214478
    Iter #572416:  Learning rate = 0.004077:   Batch Loss = 0.267557, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7159847021102905, Accuracy = 0.8684210777282715
    Iter #572928:  Learning rate = 0.004077:   Batch Loss = 0.296804, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7493817210197449, Accuracy = 0.8562753200531006
    Iter #573440:  Learning rate = 0.004077:   Batch Loss = 0.289023, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7851510047912598, Accuracy = 0.8493927121162415
    Iter #573952:  Learning rate = 0.004077:   Batch Loss = 0.383659, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7816981077194214, Accuracy = 0.8514170050621033
    Iter #574464:  Learning rate = 0.004077:   Batch Loss = 0.369074, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8599399328231812, Accuracy = 0.8303643465042114
    Iter #574976:  Learning rate = 0.004077:   Batch Loss = 0.559173, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8926703929901123, Accuracy = 0.8218623399734497
    Iter #575488:  Learning rate = 0.004077:   Batch Loss = 0.515632, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9159711599349976, Accuracy = 0.8085020184516907
    Iter #576000:  Learning rate = 0.004077:   Batch Loss = 0.460228, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9382089376449585, Accuracy = 0.8085020184516907
    Iter #576512:  Learning rate = 0.004077:   Batch Loss = 0.458216, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9469316005706787, Accuracy = 0.7939271330833435
    Iter #577024:  Learning rate = 0.004077:   Batch Loss = 0.423428, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9548153877258301, Accuracy = 0.7979757189750671
    Iter #577536:  Learning rate = 0.004077:   Batch Loss = 0.658182, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0276014804840088, Accuracy = 0.774493932723999
    Iter #578048:  Learning rate = 0.004077:   Batch Loss = 0.465623, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.005614995956421, Accuracy = 0.7829959392547607
    Iter #578560:  Learning rate = 0.004077:   Batch Loss = 0.458386, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9536992311477661, Accuracy = 0.796761155128479
    Iter #579072:  Learning rate = 0.004077:   Batch Loss = 0.524866, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9502629637718201, Accuracy = 0.8016194105148315
    Iter #579584:  Learning rate = 0.004077:   Batch Loss = 0.535650, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.93175208568573, Accuracy = 0.8105263113975525
    Iter #580096:  Learning rate = 0.004077:   Batch Loss = 0.565423, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9094251394271851, Accuracy = 0.8206477761268616
    Iter #580608:  Learning rate = 0.004077:   Batch Loss = 0.509357, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9628220796585083, Accuracy = 0.7971659898757935
    Iter #581120:  Learning rate = 0.004077:   Batch Loss = 0.528206, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9519544839859009, Accuracy = 0.8004048466682434
    Iter #581632:  Learning rate = 0.004077:   Batch Loss = 0.544327, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8971148133277893, Accuracy = 0.8157894611358643
    Iter #582144:  Learning rate = 0.004077:   Batch Loss = 0.461852, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.897139310836792, Accuracy = 0.8295546770095825
    Iter #582656:  Learning rate = 0.004077:   Batch Loss = 0.409665, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8636289834976196, Accuracy = 0.8396761417388916
    Iter #583168:  Learning rate = 0.004077:   Batch Loss = 0.440523, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8873497843742371, Accuracy = 0.8384615182876587
    Iter #583680:  Learning rate = 0.004077:   Batch Loss = 0.471524, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8833210468292236, Accuracy = 0.835627555847168
    Iter #584192:  Learning rate = 0.004077:   Batch Loss = 0.502205, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9662968516349792, Accuracy = 0.8028340339660645
    Iter #584704:  Learning rate = 0.004077:   Batch Loss = 0.463243, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.940064549446106, Accuracy = 0.8121457695960999
    Iter #585216:  Learning rate = 0.004077:   Batch Loss = 0.508132, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8956552743911743, Accuracy = 0.8263157606124878
    Iter #585728:  Learning rate = 0.004077:   Batch Loss = 0.425130, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9527662396430969, Accuracy = 0.8093117475509644
    Iter #586240:  Learning rate = 0.004077:   Batch Loss = 0.607510, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9773313999176025, Accuracy = 0.807692289352417
    Iter #586752:  Learning rate = 0.004077:   Batch Loss = 0.478281, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9536212086677551, Accuracy = 0.8093117475509644
    Iter #587264:  Learning rate = 0.004077:   Batch Loss = 0.525533, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9432265758514404, Accuracy = 0.810121476650238
    Iter #587776:  Learning rate = 0.004077:   Batch Loss = 0.482015, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0055649280548096, Accuracy = 0.7979757189750671
    Iter #588288:  Learning rate = 0.004077:   Batch Loss = 0.572821, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9261223673820496, Accuracy = 0.8275303840637207
    Iter #588800:  Learning rate = 0.004077:   Batch Loss = 0.427988, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8882032632827759, Accuracy = 0.8340080976486206
    Iter #589312:  Learning rate = 0.004077:   Batch Loss = 0.424632, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.877819299697876, Accuracy = 0.8348178267478943
    Iter #589824:  Learning rate = 0.004077:   Batch Loss = 0.401832, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9051353931427002, Accuracy = 0.8283400535583496
    Iter #590336:  Learning rate = 0.004077:   Batch Loss = 0.424338, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8911091089248657, Accuracy = 0.8449392914772034
    Iter #590848:  Learning rate = 0.004077:   Batch Loss = 0.452236, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8999569416046143, Accuracy = 0.8392712473869324
    Iter #591360:  Learning rate = 0.004077:   Batch Loss = 0.403771, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9201942086219788, Accuracy = 0.8287449479103088
    Iter #591872:  Learning rate = 0.004077:   Batch Loss = 0.464315, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9134623408317566, Accuracy = 0.8295546770095825
    Iter #592384:  Learning rate = 0.004077:   Batch Loss = 0.535597, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9335049390792847, Accuracy = 0.8165991902351379
    Iter #592896:  Learning rate = 0.004077:   Batch Loss = 0.438665, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8927872180938721, Accuracy = 0.8392712473869324
    Iter #593408:  Learning rate = 0.004077:   Batch Loss = 0.384270, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8946158289909363, Accuracy = 0.8396761417388916
    Iter #593920:  Learning rate = 0.004077:   Batch Loss = 0.419568, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8919434547424316, Accuracy = 0.8408907055854797
    Iter #594432:  Learning rate = 0.004077:   Batch Loss = 0.374474, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8601119518280029, Accuracy = 0.8485829830169678
    Iter #594944:  Learning rate = 0.004077:   Batch Loss = 0.382126, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8959900140762329, Accuracy = 0.8368421196937561
    Iter #595456:  Learning rate = 0.004077:   Batch Loss = 0.380554, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9063366651535034, Accuracy = 0.8376518487930298
    Iter #595968:  Learning rate = 0.004077:   Batch Loss = 0.363867, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8665634989738464, Accuracy = 0.8473684191703796
    Iter #596480:  Learning rate = 0.004077:   Batch Loss = 0.334961, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.836418867111206, Accuracy = 0.8530364632606506
    Iter #596992:  Learning rate = 0.004077:   Batch Loss = 0.366325, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8468621969223022, Accuracy = 0.8506072759628296
    Iter #597504:  Learning rate = 0.004077:   Batch Loss = 0.369589, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8363214731216431, Accuracy = 0.8558704257011414
    Iter #598016:  Learning rate = 0.004077:   Batch Loss = 0.354158, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8456095457077026, Accuracy = 0.8538461327552795
    Iter #598528:  Learning rate = 0.004077:   Batch Loss = 0.350082, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7955716848373413, Accuracy = 0.8643724918365479
    Iter #599040:  Learning rate = 0.004077:   Batch Loss = 0.330059, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8426209688186646, Accuracy = 0.8510121703147888
    Iter #599552:  Learning rate = 0.004077:   Batch Loss = 0.323734, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7905998229980469, Accuracy = 0.8639675974845886
    Iter #600064:  Learning rate = 0.003914:   Batch Loss = 0.326534, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7821517586708069, Accuracy = 0.8696356415748596
    Iter #600576:  Learning rate = 0.003914:   Batch Loss = 0.311168, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7967824935913086, Accuracy = 0.862348198890686
    Iter #601088:  Learning rate = 0.003914:   Batch Loss = 0.322584, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8224198818206787, Accuracy = 0.8558704257011414
    Iter #601600:  Learning rate = 0.003914:   Batch Loss = 0.301225, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8065929412841797, Accuracy = 0.8538461327552795
    Iter #602112:  Learning rate = 0.003914:   Batch Loss = 0.305448, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7881120443344116, Accuracy = 0.8615384697914124
    Iter #602624:  Learning rate = 0.003914:   Batch Loss = 0.299408, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7884238958358765, Accuracy = 0.8712550401687622
    Iter #603136:  Learning rate = 0.003914:   Batch Loss = 0.300009, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7866873741149902, Accuracy = 0.8700404763221741
    Iter #603648:  Learning rate = 0.003914:   Batch Loss = 0.300880, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7889822721481323, Accuracy = 0.8627530336380005
    Iter #604160:  Learning rate = 0.003914:   Batch Loss = 0.291184, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7694458365440369, Accuracy = 0.8676113486289978
    Iter #604672:  Learning rate = 0.003914:   Batch Loss = 0.293062, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7518943548202515, Accuracy = 0.8688259124755859
    Iter #605184:  Learning rate = 0.003914:   Batch Loss = 0.298207, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7437371611595154, Accuracy = 0.8716599345207214
    Iter #605696:  Learning rate = 0.003914:   Batch Loss = 0.292092, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.741468071937561, Accuracy = 0.8716599345207214
    Iter #606208:  Learning rate = 0.003914:   Batch Loss = 0.291636, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7416242957115173, Accuracy = 0.8708502054214478
    Iter #606720:  Learning rate = 0.003914:   Batch Loss = 0.288782, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7495293617248535, Accuracy = 0.8712550401687622
    Iter #607232:  Learning rate = 0.003914:   Batch Loss = 0.288535, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7415781021118164, Accuracy = 0.8761133551597595
    Iter #607744:  Learning rate = 0.003914:   Batch Loss = 0.288797, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.734271228313446, Accuracy = 0.8753036260604858
    Iter #608256:  Learning rate = 0.003914:   Batch Loss = 0.282985, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7379553318023682, Accuracy = 0.8757085204124451
    Iter #608768:  Learning rate = 0.003914:   Batch Loss = 0.277775, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7262274622917175, Accuracy = 0.8781376481056213
    Iter #609280:  Learning rate = 0.003914:   Batch Loss = 0.283096, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7228704690933228, Accuracy = 0.8793522119522095
    Iter #609792:  Learning rate = 0.003914:   Batch Loss = 0.272605, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7188913822174072, Accuracy = 0.876518189907074
    Iter #610304:  Learning rate = 0.003914:   Batch Loss = 0.271320, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7287687063217163, Accuracy = 0.8744939565658569
    Iter #610816:  Learning rate = 0.003914:   Batch Loss = 0.272037, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7297110557556152, Accuracy = 0.8712550401687622
    Iter #611328:  Learning rate = 0.003914:   Batch Loss = 0.271818, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7234479784965515, Accuracy = 0.8736842274665833
    Iter #611840:  Learning rate = 0.003914:   Batch Loss = 0.269402, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7141544222831726, Accuracy = 0.8761133551597595
    Iter #612352:  Learning rate = 0.003914:   Batch Loss = 0.270418, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7141519784927368, Accuracy = 0.8769230842590332
    Iter #612864:  Learning rate = 0.003914:   Batch Loss = 0.274407, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.717238187789917, Accuracy = 0.8748987913131714
    Iter #613376:  Learning rate = 0.003914:   Batch Loss = 0.262665, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7184298038482666, Accuracy = 0.8720647692680359
    Iter #613888:  Learning rate = 0.003914:   Batch Loss = 0.270398, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7077070474624634, Accuracy = 0.8761133551597595
    Iter #614400:  Learning rate = 0.003914:   Batch Loss = 0.267882, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7060335278511047, Accuracy = 0.8785424828529358
    Iter #614912:  Learning rate = 0.003914:   Batch Loss = 0.271148, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7043789029121399, Accuracy = 0.8781376481056213
    Iter #615424:  Learning rate = 0.003914:   Batch Loss = 0.258296, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6998950242996216, Accuracy = 0.876518189907074
    Iter #615936:  Learning rate = 0.003914:   Batch Loss = 0.264658, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7016006708145142, Accuracy = 0.8773279190063477
    Iter #616448:  Learning rate = 0.003914:   Batch Loss = 0.267425, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7003987431526184, Accuracy = 0.8753036260604858
    Iter #616960:  Learning rate = 0.003914:   Batch Loss = 0.262266, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6977291107177734, Accuracy = 0.8736842274665833
    Iter #617472:  Learning rate = 0.003914:   Batch Loss = 0.261587, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.695716142654419, Accuracy = 0.876518189907074
    Iter #617984:  Learning rate = 0.003914:   Batch Loss = 0.259244, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6992225050926208, Accuracy = 0.8753036260604858
    Iter #618496:  Learning rate = 0.003914:   Batch Loss = 0.254973, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6955258846282959, Accuracy = 0.8785424828529358
    Iter #619008:  Learning rate = 0.003914:   Batch Loss = 0.252703, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6936686635017395, Accuracy = 0.876518189907074
    Iter #619520:  Learning rate = 0.003914:   Batch Loss = 0.256482, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6982165575027466, Accuracy = 0.8736842274665833
    Iter #620032:  Learning rate = 0.003914:   Batch Loss = 0.262834, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6972717046737671, Accuracy = 0.8777328133583069
    Iter #620544:  Learning rate = 0.003914:   Batch Loss = 0.250445, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6960269808769226, Accuracy = 0.8757085204124451
    Iter #621056:  Learning rate = 0.003914:   Batch Loss = 0.255003, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6936894655227661, Accuracy = 0.8769230842590332
    Iter #621568:  Learning rate = 0.003914:   Batch Loss = 0.255768, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6936243772506714, Accuracy = 0.8785424828529358
    Iter #622080:  Learning rate = 0.003914:   Batch Loss = 0.249169, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6923222541809082, Accuracy = 0.8761133551597595
    Iter #622592:  Learning rate = 0.003914:   Batch Loss = 0.253126, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6877171397209167, Accuracy = 0.8757085204124451
    Iter #623104:  Learning rate = 0.003914:   Batch Loss = 0.248883, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6831881403923035, Accuracy = 0.8793522119522095
    Iter #623616:  Learning rate = 0.003914:   Batch Loss = 0.247220, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.682876706123352, Accuracy = 0.8805667757987976
    Iter #624128:  Learning rate = 0.003914:   Batch Loss = 0.247445, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6776930689811707, Accuracy = 0.8785424828529358
    Iter #624640:  Learning rate = 0.003914:   Batch Loss = 0.251299, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6836833357810974, Accuracy = 0.876518189907074
    Iter #625152:  Learning rate = 0.003914:   Batch Loss = 0.247832, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6878266930580139, Accuracy = 0.8736842274665833
    Iter #625664:  Learning rate = 0.003914:   Batch Loss = 0.246820, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6872732043266296, Accuracy = 0.8781376481056213
    Iter #626176:  Learning rate = 0.003914:   Batch Loss = 0.241871, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6896094083786011, Accuracy = 0.8744939565658569
    Iter #626688:  Learning rate = 0.003914:   Batch Loss = 0.246692, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6772140264511108, Accuracy = 0.8801619410514832
    Iter #627200:  Learning rate = 0.003914:   Batch Loss = 0.243584, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6723541617393494, Accuracy = 0.8805667757987976
    Iter #627712:  Learning rate = 0.003914:   Batch Loss = 0.241673, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6830098628997803, Accuracy = 0.8761133551597595
    Iter #628224:  Learning rate = 0.003914:   Batch Loss = 0.243643, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6913102865219116, Accuracy = 0.8740890622138977
    Iter #628736:  Learning rate = 0.003914:   Batch Loss = 0.235047, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6826369166374207, Accuracy = 0.8785424828529358
    Iter #629248:  Learning rate = 0.003914:   Batch Loss = 0.238736, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.681494414806366, Accuracy = 0.8773279190063477
    Iter #629760:  Learning rate = 0.003914:   Batch Loss = 0.240265, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6897320747375488, Accuracy = 0.873279333114624
    Iter #630272:  Learning rate = 0.003914:   Batch Loss = 0.235616, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6882032155990601, Accuracy = 0.8728744983673096
    Iter #630784:  Learning rate = 0.003914:   Batch Loss = 0.236906, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.682612419128418, Accuracy = 0.8748987913131714
    Iter #631296:  Learning rate = 0.003914:   Batch Loss = 0.240215, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.675653874874115, Accuracy = 0.8757085204124451
    Iter #631808:  Learning rate = 0.003914:   Batch Loss = 0.235618, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6673072576522827, Accuracy = 0.8805667757987976
    Iter #632320:  Learning rate = 0.003914:   Batch Loss = 0.236166, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6723963022232056, Accuracy = 0.8801619410514832
    Iter #632832:  Learning rate = 0.003914:   Batch Loss = 0.234682, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6789965629577637, Accuracy = 0.8744939565658569
    Iter #633344:  Learning rate = 0.003914:   Batch Loss = 0.229640, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6756324768066406, Accuracy = 0.8769230842590332
    Iter #633856:  Learning rate = 0.003914:   Batch Loss = 0.233304, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6706677079200745, Accuracy = 0.8777328133583069
    Iter #634368:  Learning rate = 0.003914:   Batch Loss = 0.239632, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6753736734390259, Accuracy = 0.8785424828529358
    Iter #634880:  Learning rate = 0.003914:   Batch Loss = 0.232687, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6744548678398132, Accuracy = 0.8773279190063477
    Iter #635392:  Learning rate = 0.003914:   Batch Loss = 0.232623, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.670535147190094, Accuracy = 0.8773279190063477
    Iter #635904:  Learning rate = 0.003914:   Batch Loss = 0.231050, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6761423349380493, Accuracy = 0.8761133551597595
    Iter #636416:  Learning rate = 0.003914:   Batch Loss = 0.233495, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6801729202270508, Accuracy = 0.8740890622138977
    Iter #636928:  Learning rate = 0.003914:   Batch Loss = 0.230435, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6721057295799255, Accuracy = 0.8748987913131714
    Iter #637440:  Learning rate = 0.003914:   Batch Loss = 0.227443, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.671644926071167, Accuracy = 0.8740890622138977
    Iter #637952:  Learning rate = 0.003914:   Batch Loss = 0.228571, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6737772226333618, Accuracy = 0.873279333114624
    Iter #638464:  Learning rate = 0.003914:   Batch Loss = 0.231340, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6686567664146423, Accuracy = 0.8748987913131714
    Iter #638976:  Learning rate = 0.003914:   Batch Loss = 0.223079, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6624741554260254, Accuracy = 0.8757085204124451
    Iter #639488:  Learning rate = 0.003914:   Batch Loss = 0.228002, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6675451993942261, Accuracy = 0.878947377204895
    Iter #640000:  Learning rate = 0.003914:   Batch Loss = 0.230272, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6801905035972595, Accuracy = 0.8757085204124451
    Iter #640512:  Learning rate = 0.003914:   Batch Loss = 0.228276, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.674712061882019, Accuracy = 0.8761133551597595
    Iter #641024:  Learning rate = 0.003914:   Batch Loss = 0.226213, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6740695238113403, Accuracy = 0.8684210777282715
    Iter #641536:  Learning rate = 0.003914:   Batch Loss = 0.222816, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6716394424438477, Accuracy = 0.8728744983673096
    Iter #642048:  Learning rate = 0.003914:   Batch Loss = 0.226529, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6724101901054382, Accuracy = 0.8724696636199951
    Iter #642560:  Learning rate = 0.003914:   Batch Loss = 0.222439, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6689215302467346, Accuracy = 0.8740890622138977
    Iter #643072:  Learning rate = 0.003914:   Batch Loss = 0.227116, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6674497127532959, Accuracy = 0.8785424828529358
    Iter #643584:  Learning rate = 0.003914:   Batch Loss = 0.218794, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6683182716369629, Accuracy = 0.8744939565658569
    Iter #644096:  Learning rate = 0.003914:   Batch Loss = 0.225325, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6755033135414124, Accuracy = 0.8736842274665833
    Iter #644608:  Learning rate = 0.003914:   Batch Loss = 0.226822, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6962841749191284, Accuracy = 0.865587055683136
    Iter #645120:  Learning rate = 0.003914:   Batch Loss = 0.288338, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7453727722167969, Accuracy = 0.852226734161377
    Iter #645632:  Learning rate = 0.003914:   Batch Loss = 0.246772, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7620598673820496, Accuracy = 0.8510121703147888
    Iter #646144:  Learning rate = 0.003914:   Batch Loss = 0.317879, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7965864539146423, Accuracy = 0.8429149985313416
    Iter #646656:  Learning rate = 0.003914:   Batch Loss = 0.609163, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9654516577720642, Accuracy = 0.794331967830658
    Iter #647168:  Learning rate = 0.003914:   Batch Loss = 0.627922, Accuracy = 0.875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.1255462169647217, Accuracy = 0.7307692170143127
    Iter #647680:  Learning rate = 0.003914:   Batch Loss = 0.697548, Accuracy = 0.859375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.037571668624878, Accuracy = 0.7526316046714783
    Iter #648192:  Learning rate = 0.003914:   Batch Loss = 0.685805, Accuracy = 0.875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9468127489089966, Accuracy = 0.7878542542457581
    Iter #648704:  Learning rate = 0.003914:   Batch Loss = 0.650971, Accuracy = 0.859375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.96346515417099, Accuracy = 0.7748987674713135
    Iter #649216:  Learning rate = 0.003914:   Batch Loss = 0.580825, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0409152507781982, Accuracy = 0.7740890979766846
    Iter #649728:  Learning rate = 0.003914:   Batch Loss = 0.529075, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0022881031036377, Accuracy = 0.7858299612998962
    Iter #650240:  Learning rate = 0.003914:   Batch Loss = 0.510877, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.027618169784546, Accuracy = 0.768825888633728
    Iter #650752:  Learning rate = 0.003914:   Batch Loss = 0.486724, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0018028020858765, Accuracy = 0.7939271330833435
    Iter #651264:  Learning rate = 0.003914:   Batch Loss = 0.521439, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9206545352935791, Accuracy = 0.800000011920929
    Iter #651776:  Learning rate = 0.003914:   Batch Loss = 0.685563, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8981671929359436, Accuracy = 0.8145748972892761
    Iter #652288:  Learning rate = 0.003914:   Batch Loss = 0.477296, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9419882893562317, Accuracy = 0.7951416969299316
    Iter #652800:  Learning rate = 0.003914:   Batch Loss = 0.584315, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9214555025100708, Accuracy = 0.813360333442688
    Iter #653312:  Learning rate = 0.003914:   Batch Loss = 0.682473, Accuracy = 0.875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9943463802337646, Accuracy = 0.7931174039840698
    Iter #653824:  Learning rate = 0.003914:   Batch Loss = 0.576074, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9596412777900696, Accuracy = 0.8052631616592407
    Iter #654336:  Learning rate = 0.003914:   Batch Loss = 0.525228, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9552200436592102, Accuracy = 0.7971659898757935
    Iter #654848:  Learning rate = 0.003914:   Batch Loss = 0.498015, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9664665460586548, Accuracy = 0.7987854480743408
    Iter #655360:  Learning rate = 0.003914:   Batch Loss = 0.469249, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9304498434066772, Accuracy = 0.8072874546051025
    Iter #655872:  Learning rate = 0.003914:   Batch Loss = 0.403267, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9932681322097778, Accuracy = 0.7866396903991699
    Iter #656384:  Learning rate = 0.003914:   Batch Loss = 0.709561, Accuracy = 0.875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0122675895690918, Accuracy = 0.7890688180923462
    Iter #656896:  Learning rate = 0.003914:   Batch Loss = 0.461827, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9007564783096313, Accuracy = 0.8230769038200378
    Iter #657408:  Learning rate = 0.003914:   Batch Loss = 0.513701, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9560648202896118, Accuracy = 0.8113360404968262
    Iter #657920:  Learning rate = 0.003914:   Batch Loss = 0.485812, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9404669404029846, Accuracy = 0.8105263113975525
    Iter #658432:  Learning rate = 0.003914:   Batch Loss = 0.470799, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8561183214187622, Accuracy = 0.8376518487930298
    Iter #658944:  Learning rate = 0.003914:   Batch Loss = 0.417642, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8837774991989136, Accuracy = 0.8186234831809998
    Iter #659456:  Learning rate = 0.003914:   Batch Loss = 0.408262, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8224051594734192, Accuracy = 0.8514170050621033
    Iter #659968:  Learning rate = 0.003914:   Batch Loss = 0.386345, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8309370279312134, Accuracy = 0.8425101041793823
    Iter #660480:  Learning rate = 0.003914:   Batch Loss = 0.388910, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8288789391517639, Accuracy = 0.8404858112335205
    Iter #660992:  Learning rate = 0.003914:   Batch Loss = 0.343691, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8439486026763916, Accuracy = 0.8421052694320679
    Iter #661504:  Learning rate = 0.003914:   Batch Loss = 0.378421, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8647439479827881, Accuracy = 0.8319838047027588
    Iter #662016:  Learning rate = 0.003914:   Batch Loss = 0.435353, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8495358228683472, Accuracy = 0.8412955403327942
    Iter #662528:  Learning rate = 0.003914:   Batch Loss = 0.404366, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.877292275428772, Accuracy = 0.8287449479103088
    Iter #663040:  Learning rate = 0.003914:   Batch Loss = 0.414883, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8583791255950928, Accuracy = 0.8392712473869324
    Iter #663552:  Learning rate = 0.003914:   Batch Loss = 0.374501, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9009180068969727, Accuracy = 0.8226720690727234
    Iter #664064:  Learning rate = 0.003914:   Batch Loss = 0.389153, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8432780504226685, Accuracy = 0.8469635844230652
    Iter #664576:  Learning rate = 0.003914:   Batch Loss = 0.397437, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8445547819137573, Accuracy = 0.8461538553237915
    Iter #665088:  Learning rate = 0.003914:   Batch Loss = 0.366436, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.863684892654419, Accuracy = 0.8323886394500732
    Iter #665600:  Learning rate = 0.003914:   Batch Loss = 0.371307, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8419585227966309, Accuracy = 0.8449392914772034
    Iter #666112:  Learning rate = 0.003914:   Batch Loss = 0.407748, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8360298871994019, Accuracy = 0.8469635844230652
    Iter #666624:  Learning rate = 0.003914:   Batch Loss = 0.382377, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8356695175170898, Accuracy = 0.8481781482696533
    Iter #667136:  Learning rate = 0.003914:   Batch Loss = 0.511107, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8125253915786743, Accuracy = 0.8485829830169678
    Iter #667648:  Learning rate = 0.003914:   Batch Loss = 0.437379, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8360147476196289, Accuracy = 0.8429149985313416
    Iter #668160:  Learning rate = 0.003914:   Batch Loss = 0.364540, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8759703636169434, Accuracy = 0.8327935338020325
    Iter #668672:  Learning rate = 0.003914:   Batch Loss = 0.352146, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9423433542251587, Accuracy = 0.8121457695960999
    Iter #669184:  Learning rate = 0.003914:   Batch Loss = 0.380666, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8699795007705688, Accuracy = 0.8255060911178589
    Iter #669696:  Learning rate = 0.003914:   Batch Loss = 0.396414, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8542706966400146, Accuracy = 0.8429149985313416
    Iter #670208:  Learning rate = 0.003914:   Batch Loss = 0.420988, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8604357242584229, Accuracy = 0.8360323905944824
    Iter #670720:  Learning rate = 0.003914:   Batch Loss = 0.350443, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8301442265510559, Accuracy = 0.8453441262245178
    Iter #671232:  Learning rate = 0.003914:   Batch Loss = 0.422186, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8378995656967163, Accuracy = 0.8421052694320679
    Iter #671744:  Learning rate = 0.003914:   Batch Loss = 0.355334, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8301317691802979, Accuracy = 0.8510121703147888
    Iter #672256:  Learning rate = 0.003914:   Batch Loss = 0.300937, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7862088680267334, Accuracy = 0.8712550401687622
    Iter #672768:  Learning rate = 0.003914:   Batch Loss = 0.358129, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8100670576095581, Accuracy = 0.8578947186470032
    Iter #673280:  Learning rate = 0.003914:   Batch Loss = 0.387918, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8077017068862915, Accuracy = 0.856680154800415
    Iter #673792:  Learning rate = 0.003914:   Batch Loss = 0.328020, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7848078012466431, Accuracy = 0.8603239059448242
    Iter #674304:  Learning rate = 0.003914:   Batch Loss = 0.341296, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7841719388961792, Accuracy = 0.8550607562065125
    Iter #674816:  Learning rate = 0.003914:   Batch Loss = 0.306489, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8721605539321899, Accuracy = 0.8372469544410706
    Iter #675328:  Learning rate = 0.003914:   Batch Loss = 0.379726, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8648191690444946, Accuracy = 0.840080976486206
    Iter #675840:  Learning rate = 0.003914:   Batch Loss = 0.369618, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8154712915420532, Accuracy = 0.8558704257011414
    Iter #676352:  Learning rate = 0.003914:   Batch Loss = 0.332397, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8257921934127808, Accuracy = 0.8497975468635559
    Iter #676864:  Learning rate = 0.003914:   Batch Loss = 0.323919, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8211448192596436, Accuracy = 0.8441295623779297
    Iter #677376:  Learning rate = 0.003914:   Batch Loss = 0.327748, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8191570043563843, Accuracy = 0.8469635844230652
    Iter #677888:  Learning rate = 0.003914:   Batch Loss = 0.364712, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8152728080749512, Accuracy = 0.8473684191703796
    Iter #678400:  Learning rate = 0.003914:   Batch Loss = 0.349756, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8138676881790161, Accuracy = 0.8469635844230652
    Iter #678912:  Learning rate = 0.003914:   Batch Loss = 0.339339, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8204222321510315, Accuracy = 0.8485829830169678
    Iter #679424:  Learning rate = 0.003914:   Batch Loss = 0.358179, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8049724102020264, Accuracy = 0.8510121703147888
    Iter #679936:  Learning rate = 0.003914:   Batch Loss = 0.337976, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8335843086242676, Accuracy = 0.8388664126396179
    Iter #680448:  Learning rate = 0.003914:   Batch Loss = 0.307584, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8122006058692932, Accuracy = 0.8469635844230652
    Iter #680960:  Learning rate = 0.003914:   Batch Loss = 0.299766, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7917877435684204, Accuracy = 0.8591092824935913
    Iter #681472:  Learning rate = 0.003914:   Batch Loss = 0.317354, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8194869756698608, Accuracy = 0.8502024412155151
    Iter #681984:  Learning rate = 0.003914:   Batch Loss = 0.299001, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7700743079185486, Accuracy = 0.8619433045387268
    Iter #682496:  Learning rate = 0.003914:   Batch Loss = 0.311126, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7732102870941162, Accuracy = 0.8587044477462769
    Iter #683008:  Learning rate = 0.003914:   Batch Loss = 0.304868, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7581645250320435, Accuracy = 0.8639675974845886
    Iter #683520:  Learning rate = 0.003914:   Batch Loss = 0.296190, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7677475214004517, Accuracy = 0.8627530336380005
    Iter #684032:  Learning rate = 0.003914:   Batch Loss = 0.314463, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7765034437179565, Accuracy = 0.8554655909538269
    Iter #684544:  Learning rate = 0.003914:   Batch Loss = 0.310019, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7842754125595093, Accuracy = 0.8651821613311768
    Iter #685056:  Learning rate = 0.003914:   Batch Loss = 0.366903, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7706405520439148, Accuracy = 0.8712550401687622
    Iter #685568:  Learning rate = 0.003914:   Batch Loss = 0.315571, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7577203512191772, Accuracy = 0.8651821613311768
    Iter #686080:  Learning rate = 0.003914:   Batch Loss = 0.293854, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7491791844367981, Accuracy = 0.8716599345207214
    Iter #686592:  Learning rate = 0.003914:   Batch Loss = 0.289630, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7345671653747559, Accuracy = 0.8716599345207214
    Iter #687104:  Learning rate = 0.003914:   Batch Loss = 0.274974, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7306426763534546, Accuracy = 0.8740890622138977
    Iter #687616:  Learning rate = 0.003914:   Batch Loss = 0.277996, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7243785858154297, Accuracy = 0.8753036260604858
    Iter #688128:  Learning rate = 0.003914:   Batch Loss = 0.277296, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7109758853912354, Accuracy = 0.878947377204895
    Iter #688640:  Learning rate = 0.003914:   Batch Loss = 0.282568, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7092267870903015, Accuracy = 0.8748987913131714
    Iter #689152:  Learning rate = 0.003914:   Batch Loss = 0.271670, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6985520124435425, Accuracy = 0.8813765048980713
    Iter #689664:  Learning rate = 0.003914:   Batch Loss = 0.268920, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6957955360412598, Accuracy = 0.8773279190063477
    Iter #690176:  Learning rate = 0.003914:   Batch Loss = 0.271734, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6931694149971008, Accuracy = 0.878947377204895
    Iter #690688:  Learning rate = 0.003914:   Batch Loss = 0.263790, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6863043904304504, Accuracy = 0.8817813992500305
    Iter #691200:  Learning rate = 0.003914:   Batch Loss = 0.267397, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6925979256629944, Accuracy = 0.8801619410514832
    Iter #691712:  Learning rate = 0.003914:   Batch Loss = 0.263099, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6755381226539612, Accuracy = 0.8846153616905212
    Iter #692224:  Learning rate = 0.003914:   Batch Loss = 0.260778, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6642105579376221, Accuracy = 0.887854278087616
    Iter #692736:  Learning rate = 0.003914:   Batch Loss = 0.269265, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6577572226524353, Accuracy = 0.8914979696273804
    Iter #693248:  Learning rate = 0.003914:   Batch Loss = 0.261698, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6584353446960449, Accuracy = 0.8874493837356567
    Iter #693760:  Learning rate = 0.003914:   Batch Loss = 0.259821, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6598964929580688, Accuracy = 0.8882591128349304
    Iter #694272:  Learning rate = 0.003914:   Batch Loss = 0.256047, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6607793569564819, Accuracy = 0.8862348198890686
    Iter #694784:  Learning rate = 0.003914:   Batch Loss = 0.253707, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6599028706550598, Accuracy = 0.8870445489883423
    Iter #695296:  Learning rate = 0.003914:   Batch Loss = 0.260071, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6649070382118225, Accuracy = 0.8846153616905212
    Iter #695808:  Learning rate = 0.003914:   Batch Loss = 0.269606, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6804808378219604, Accuracy = 0.8769230842590332
    Iter #696320:  Learning rate = 0.003914:   Batch Loss = 0.251478, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6775392889976501, Accuracy = 0.8797571063041687
    Iter #696832:  Learning rate = 0.003914:   Batch Loss = 0.255332, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6668789982795715, Accuracy = 0.8809716701507568
    Iter #697344:  Learning rate = 0.003914:   Batch Loss = 0.251724, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6568760275840759, Accuracy = 0.8854250907897949
    Iter #697856:  Learning rate = 0.003914:   Batch Loss = 0.248901, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6691262125968933, Accuracy = 0.8777328133583069
    Iter #698368:  Learning rate = 0.003914:   Batch Loss = 0.250578, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.671928882598877, Accuracy = 0.8748987913131714
    Iter #698880:  Learning rate = 0.003914:   Batch Loss = 0.245971, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6691195368766785, Accuracy = 0.8797571063041687
    Iter #699392:  Learning rate = 0.003914:   Batch Loss = 0.242777, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6635733842849731, Accuracy = 0.8797571063041687
    Iter #699904:  Learning rate = 0.003914:   Batch Loss = 0.241033, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6633502244949341, Accuracy = 0.8805667757987976
    Iter #700416:  Learning rate = 0.003757:   Batch Loss = 0.237843, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.654554545879364, Accuracy = 0.8850202560424805
    Iter #700928:  Learning rate = 0.003757:   Batch Loss = 0.240750, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6455017924308777, Accuracy = 0.8870445489883423
    Iter #701440:  Learning rate = 0.003757:   Batch Loss = 0.237529, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6418333649635315, Accuracy = 0.8866396546363831
    Iter #701952:  Learning rate = 0.003757:   Batch Loss = 0.237636, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6590441465377808, Accuracy = 0.8813765048980713
    Iter #702464:  Learning rate = 0.003757:   Batch Loss = 0.235007, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6538228988647461, Accuracy = 0.8838056921958923
    Iter #702976:  Learning rate = 0.003757:   Batch Loss = 0.235984, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6477323770523071, Accuracy = 0.8866396546363831
    Iter #703488:  Learning rate = 0.003757:   Batch Loss = 0.237044, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6453174352645874, Accuracy = 0.882186233997345
    Iter #704000:  Learning rate = 0.003757:   Batch Loss = 0.241137, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6509603261947632, Accuracy = 0.878947377204895
    Iter #704512:  Learning rate = 0.003757:   Batch Loss = 0.237317, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6413549780845642, Accuracy = 0.8858299851417542
    Iter #705024:  Learning rate = 0.003757:   Batch Loss = 0.239412, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6418421268463135, Accuracy = 0.8870445489883423
    Iter #705536:  Learning rate = 0.003757:   Batch Loss = 0.241308, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6446171998977661, Accuracy = 0.8842105269432068
    Iter #706048:  Learning rate = 0.003757:   Batch Loss = 0.230404, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6445266008377075, Accuracy = 0.8797571063041687
    Iter #706560:  Learning rate = 0.003757:   Batch Loss = 0.232370, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6480920314788818, Accuracy = 0.8850202560424805
    Iter #707072:  Learning rate = 0.003757:   Batch Loss = 0.235728, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6378982067108154, Accuracy = 0.8874493837356567
    Iter #707584:  Learning rate = 0.003757:   Batch Loss = 0.230089, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6361441612243652, Accuracy = 0.8890688419342041
    Iter #708096:  Learning rate = 0.003757:   Batch Loss = 0.229388, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6340609788894653, Accuracy = 0.8842105269432068
    Iter #708608:  Learning rate = 0.003757:   Batch Loss = 0.233173, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6317761540412903, Accuracy = 0.8874493837356567
    Iter #709120:  Learning rate = 0.003757:   Batch Loss = 0.229667, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6300032138824463, Accuracy = 0.8890688419342041
    Iter #709632:  Learning rate = 0.003757:   Batch Loss = 0.232432, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6327003240585327, Accuracy = 0.882186233997345
    Iter #710144:  Learning rate = 0.003757:   Batch Loss = 0.227326, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6366276144981384, Accuracy = 0.8793522119522095
    Iter #710656:  Learning rate = 0.003757:   Batch Loss = 0.233607, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6572300791740417, Accuracy = 0.8720647692680359
    Iter #711168:  Learning rate = 0.003757:   Batch Loss = 0.237106, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6519719958305359, Accuracy = 0.8842105269432068
    Iter #711680:  Learning rate = 0.003757:   Batch Loss = 0.227221, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6555429697036743, Accuracy = 0.8858299851417542
    Iter #712192:  Learning rate = 0.003757:   Batch Loss = 0.233243, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6588232517242432, Accuracy = 0.8777328133583069
    Iter #712704:  Learning rate = 0.003757:   Batch Loss = 0.279868, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6992185115814209, Accuracy = 0.8651821613311768
    Iter #713216:  Learning rate = 0.003757:   Batch Loss = 0.265982, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6958230137825012, Accuracy = 0.8631578683853149
    Iter #713728:  Learning rate = 0.003757:   Batch Loss = 0.257935, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6942260265350342, Accuracy = 0.8672064542770386
    Iter #714240:  Learning rate = 0.003757:   Batch Loss = 0.338405, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8159785270690918, Accuracy = 0.8295546770095825
    Iter #714752:  Learning rate = 0.003757:   Batch Loss = 0.456693, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8339781761169434, Accuracy = 0.8242915272712708
    Iter #715264:  Learning rate = 0.003757:   Batch Loss = 0.480573, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8500131964683533, Accuracy = 0.8186234831809998
    Iter #715776:  Learning rate = 0.003757:   Batch Loss = 0.390170, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9024971127510071, Accuracy = 0.8093117475509644
    Iter #716288:  Learning rate = 0.003757:   Batch Loss = 0.483880, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.881499707698822, Accuracy = 0.8190283179283142
    Iter #716800:  Learning rate = 0.003757:   Batch Loss = 0.369042, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9094955325126648, Accuracy = 0.8056679964065552
    Iter #717312:  Learning rate = 0.003757:   Batch Loss = 0.533080, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9611569046974182, Accuracy = 0.794331967830658
    Iter #717824:  Learning rate = 0.003757:   Batch Loss = 0.456471, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9554816484451294, Accuracy = 0.7939271330833435
    Iter #718336:  Learning rate = 0.003757:   Batch Loss = 0.488181, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8621528148651123, Accuracy = 0.8279352188110352
    Iter #718848:  Learning rate = 0.003757:   Batch Loss = 0.681164, Accuracy = 0.875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9860389828681946, Accuracy = 0.7894737124443054
    Iter #719360:  Learning rate = 0.003757:   Batch Loss = 0.771874, Accuracy = 0.828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8758969306945801, Accuracy = 0.8165991902351379
    Iter #719872:  Learning rate = 0.003757:   Batch Loss = 0.440762, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.858992338180542, Accuracy = 0.826720654964447
    Iter #720384:  Learning rate = 0.003757:   Batch Loss = 0.505063, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8235447406768799, Accuracy = 0.8340080976486206
    Iter #720896:  Learning rate = 0.003757:   Batch Loss = 0.468054, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8579158782958984, Accuracy = 0.8311740756034851
    Iter #721408:  Learning rate = 0.003757:   Batch Loss = 0.422029, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8610062599182129, Accuracy = 0.8287449479103088
    Iter #721920:  Learning rate = 0.003757:   Batch Loss = 0.378493, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.929669976234436, Accuracy = 0.8068826198577881
    Iter #722432:  Learning rate = 0.003757:   Batch Loss = 0.382206, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8535854816436768, Accuracy = 0.8194332122802734
    Iter #722944:  Learning rate = 0.003757:   Batch Loss = 0.397571, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8753252029418945, Accuracy = 0.8287449479103088
    Iter #723456:  Learning rate = 0.003757:   Batch Loss = 0.427981, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8570036888122559, Accuracy = 0.8259109258651733
    Iter #723968:  Learning rate = 0.003757:   Batch Loss = 0.372146, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8518659472465515, Accuracy = 0.8344129323959351
    Iter #724480:  Learning rate = 0.003757:   Batch Loss = 0.444329, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.938210129737854, Accuracy = 0.8141700625419617
    Iter #724992:  Learning rate = 0.003757:   Batch Loss = 0.432657, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8269907832145691, Accuracy = 0.8380566835403442
    Iter #725504:  Learning rate = 0.003757:   Batch Loss = 0.357101, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8421283960342407, Accuracy = 0.8396761417388916
    Iter #726016:  Learning rate = 0.003757:   Batch Loss = 0.377373, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8351311683654785, Accuracy = 0.8473684191703796
    Iter #726528:  Learning rate = 0.003757:   Batch Loss = 0.381563, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8227555751800537, Accuracy = 0.8412955403327942
    Iter #727040:  Learning rate = 0.003757:   Batch Loss = 0.424801, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8181520104408264, Accuracy = 0.8473684191703796
    Iter #727552:  Learning rate = 0.003757:   Batch Loss = 0.332021, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8082183003425598, Accuracy = 0.840080976486206
    Iter #728064:  Learning rate = 0.003757:   Batch Loss = 0.379857, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7970559597015381, Accuracy = 0.8502024412155151
    Iter #728576:  Learning rate = 0.003757:   Batch Loss = 0.352555, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7913398146629333, Accuracy = 0.8530364632606506
    Iter #729088:  Learning rate = 0.003757:   Batch Loss = 0.356552, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7687584161758423, Accuracy = 0.8578947186470032
    Iter #729600:  Learning rate = 0.003757:   Batch Loss = 0.403867, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7608994245529175, Accuracy = 0.8578947186470032
    Iter #730112:  Learning rate = 0.003757:   Batch Loss = 0.416589, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8159552812576294, Accuracy = 0.8445343971252441
    Iter #730624:  Learning rate = 0.003757:   Batch Loss = 0.337387, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7638736963272095, Accuracy = 0.8570850491523743
    Iter #731136:  Learning rate = 0.003757:   Batch Loss = 0.353865, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7740039229393005, Accuracy = 0.8554655909538269
    Iter #731648:  Learning rate = 0.003757:   Batch Loss = 0.356905, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7754802703857422, Accuracy = 0.8615384697914124
    Iter #732160:  Learning rate = 0.003757:   Batch Loss = 0.290405, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.776265025138855, Accuracy = 0.8542510271072388
    Iter #732672:  Learning rate = 0.003757:   Batch Loss = 0.300732, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8162105083465576, Accuracy = 0.8502024412155151
    Iter #733184:  Learning rate = 0.003757:   Batch Loss = 0.367629, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7898929119110107, Accuracy = 0.8554655909538269
    Iter #733696:  Learning rate = 0.003757:   Batch Loss = 0.369792, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8254783749580383, Accuracy = 0.8392712473869324
    Iter #734208:  Learning rate = 0.003757:   Batch Loss = 0.418398, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7813211679458618, Accuracy = 0.8570850491523743
    Iter #734720:  Learning rate = 0.003757:   Batch Loss = 0.390169, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8496299386024475, Accuracy = 0.8412955403327942
    Iter #735232:  Learning rate = 0.003757:   Batch Loss = 0.386857, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8377628326416016, Accuracy = 0.8461538553237915
    Iter #735744:  Learning rate = 0.003757:   Batch Loss = 0.363342, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.871290922164917, Accuracy = 0.8331983685493469
    Iter #736256:  Learning rate = 0.003757:   Batch Loss = 0.320229, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7977771759033203, Accuracy = 0.8538461327552795
    Iter #736768:  Learning rate = 0.003757:   Batch Loss = 0.345435, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.795152485370636, Accuracy = 0.8595141768455505
    Iter #737280:  Learning rate = 0.003757:   Batch Loss = 0.328654, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7858881950378418, Accuracy = 0.8562753200531006
    Iter #737792:  Learning rate = 0.003757:   Batch Loss = 0.320672, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7550283074378967, Accuracy = 0.862348198890686
    Iter #738304:  Learning rate = 0.003757:   Batch Loss = 0.308211, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.802423357963562, Accuracy = 0.8493927121162415
    Iter #738816:  Learning rate = 0.003757:   Batch Loss = 0.320476, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8063052296638489, Accuracy = 0.8526315689086914
    Iter #739328:  Learning rate = 0.003757:   Batch Loss = 0.411670, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8209180235862732, Accuracy = 0.8510121703147888
    Iter #739840:  Learning rate = 0.003757:   Batch Loss = 0.352336, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8222994804382324, Accuracy = 0.8392712473869324
    Iter #740352:  Learning rate = 0.003757:   Batch Loss = 0.370593, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8132201433181763, Accuracy = 0.852226734161377
    Iter #740864:  Learning rate = 0.003757:   Batch Loss = 0.340682, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7790569067001343, Accuracy = 0.8591092824935913
    Iter #741376:  Learning rate = 0.003757:   Batch Loss = 0.312339, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7576290369033813, Accuracy = 0.8603239059448242
    Iter #741888:  Learning rate = 0.003757:   Batch Loss = 0.360620, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8148737549781799, Accuracy = 0.8449392914772034
    Iter #742400:  Learning rate = 0.003757:   Batch Loss = 0.449849, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7971171140670776, Accuracy = 0.8611335754394531
    Iter #742912:  Learning rate = 0.003757:   Batch Loss = 0.332339, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8135614395141602, Accuracy = 0.846558690071106
    Iter #743424:  Learning rate = 0.003757:   Batch Loss = 0.360520, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8011326789855957, Accuracy = 0.8554655909538269
    Iter #743936:  Learning rate = 0.003757:   Batch Loss = 0.296876, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7487486004829407, Accuracy = 0.8643724918365479
    Iter #744448:  Learning rate = 0.003757:   Batch Loss = 0.306686, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7345458269119263, Accuracy = 0.8639675974845886
    Iter #744960:  Learning rate = 0.003757:   Batch Loss = 0.298522, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7139612436294556, Accuracy = 0.8785424828529358
    Iter #745472:  Learning rate = 0.003757:   Batch Loss = 0.314779, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6970475316047668, Accuracy = 0.8761133551597595
    Iter #745984:  Learning rate = 0.003757:   Batch Loss = 0.303887, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7260333895683289, Accuracy = 0.8659918904304504
    Iter #746496:  Learning rate = 0.003757:   Batch Loss = 0.305923, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7060314416885376, Accuracy = 0.8793522119522095
    Iter #747008:  Learning rate = 0.003757:   Batch Loss = 0.284620, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7293823957443237, Accuracy = 0.8712550401687622
    Iter #747520:  Learning rate = 0.003757:   Batch Loss = 0.281143, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6997907161712646, Accuracy = 0.8744939565658569
    Iter #748032:  Learning rate = 0.003757:   Batch Loss = 0.270322, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6853853464126587, Accuracy = 0.8781376481056213
    Iter #748544:  Learning rate = 0.003757:   Batch Loss = 0.283991, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6746770143508911, Accuracy = 0.8825910687446594
    Iter #749056:  Learning rate = 0.003757:   Batch Loss = 0.267850, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6783120632171631, Accuracy = 0.8850202560424805
    Iter #749568:  Learning rate = 0.003757:   Batch Loss = 0.270974, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6849547624588013, Accuracy = 0.8785424828529358
    Iter #750080:  Learning rate = 0.003757:   Batch Loss = 0.282914, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6650800704956055, Accuracy = 0.8862348198890686
    Iter #750592:  Learning rate = 0.003757:   Batch Loss = 0.263451, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6851712465286255, Accuracy = 0.8862348198890686
    Iter #751104:  Learning rate = 0.003757:   Batch Loss = 0.265011, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6795159578323364, Accuracy = 0.8825910687446594
    Iter #751616:  Learning rate = 0.003757:   Batch Loss = 0.265415, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6718060374259949, Accuracy = 0.8842105269432068
    Iter #752128:  Learning rate = 0.003757:   Batch Loss = 0.254657, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6831412315368652, Accuracy = 0.8809716701507568
    Iter #752640:  Learning rate = 0.003757:   Batch Loss = 0.255857, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6667368412017822, Accuracy = 0.8801619410514832
    Iter #753152:  Learning rate = 0.003757:   Batch Loss = 0.253651, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6595913767814636, Accuracy = 0.8801619410514832
    Iter #753664:  Learning rate = 0.003757:   Batch Loss = 0.250579, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6502822637557983, Accuracy = 0.8850202560424805
    Iter #754176:  Learning rate = 0.003757:   Batch Loss = 0.248729, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6421338319778442, Accuracy = 0.887854278087616
    Iter #754688:  Learning rate = 0.003757:   Batch Loss = 0.244933, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6380800008773804, Accuracy = 0.8882591128349304
    Iter #755200:  Learning rate = 0.003757:   Batch Loss = 0.250668, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6360863447189331, Accuracy = 0.8874493837356567
    Iter #755712:  Learning rate = 0.003757:   Batch Loss = 0.246980, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6369143128395081, Accuracy = 0.8890688419342041
    Iter #756224:  Learning rate = 0.003757:   Batch Loss = 0.242799, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6361658573150635, Accuracy = 0.8870445489883423
    Iter #756736:  Learning rate = 0.003757:   Batch Loss = 0.245299, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.632386326789856, Accuracy = 0.887854278087616
    Iter #757248:  Learning rate = 0.003757:   Batch Loss = 0.241968, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.631014347076416, Accuracy = 0.8870445489883423
    Iter #757760:  Learning rate = 0.003757:   Batch Loss = 0.242980, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6314221620559692, Accuracy = 0.8886639475822449
    Iter #758272:  Learning rate = 0.003757:   Batch Loss = 0.241202, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6304039359092712, Accuracy = 0.8882591128349304
    Iter #758784:  Learning rate = 0.003757:   Batch Loss = 0.240902, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6303367614746094, Accuracy = 0.8890688419342041
    Iter #759296:  Learning rate = 0.003757:   Batch Loss = 0.238029, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.624609112739563, Accuracy = 0.8894736766815186
    Iter #759808:  Learning rate = 0.003757:   Batch Loss = 0.238986, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6275467872619629, Accuracy = 0.8902834057807922
    Iter #760320:  Learning rate = 0.003757:   Batch Loss = 0.243309, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6317980885505676, Accuracy = 0.8882591128349304
    Iter #760832:  Learning rate = 0.003757:   Batch Loss = 0.237078, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6301249265670776, Accuracy = 0.8866396546363831
    Iter #761344:  Learning rate = 0.003757:   Batch Loss = 0.237732, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6273505091667175, Accuracy = 0.8866396546363831
    Iter #761856:  Learning rate = 0.003757:   Batch Loss = 0.234784, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.627593994140625, Accuracy = 0.8890688419342041
    Iter #762368:  Learning rate = 0.003757:   Batch Loss = 0.235530, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6289695501327515, Accuracy = 0.8870445489883423
    Iter #762880:  Learning rate = 0.003757:   Batch Loss = 0.236288, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6245456337928772, Accuracy = 0.8890688419342041
    Iter #763392:  Learning rate = 0.003757:   Batch Loss = 0.230052, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6231855154037476, Accuracy = 0.8906882405281067
    Iter #763904:  Learning rate = 0.003757:   Batch Loss = 0.232824, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6375826597213745, Accuracy = 0.8858299851417542
    Iter #764416:  Learning rate = 0.003757:   Batch Loss = 0.233223, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6442294120788574, Accuracy = 0.8834007978439331
    Iter #764928:  Learning rate = 0.003757:   Batch Loss = 0.228936, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6335111856460571, Accuracy = 0.8854250907897949
    Iter #765440:  Learning rate = 0.003757:   Batch Loss = 0.237904, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6229877471923828, Accuracy = 0.8882591128349304
    Iter #765952:  Learning rate = 0.003757:   Batch Loss = 0.229990, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6155754923820496, Accuracy = 0.8898785710334778
    Iter #766464:  Learning rate = 0.003757:   Batch Loss = 0.225221, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6178855299949646, Accuracy = 0.8886639475822449
    Iter #766976:  Learning rate = 0.003757:   Batch Loss = 0.229219, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6228352785110474, Accuracy = 0.8854250907897949
    Iter #767488:  Learning rate = 0.003757:   Batch Loss = 0.225800, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6218585968017578, Accuracy = 0.887854278087616
    Iter #768000:  Learning rate = 0.003757:   Batch Loss = 0.225675, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6180142164230347, Accuracy = 0.8902834057807922
    Iter #768512:  Learning rate = 0.003757:   Batch Loss = 0.232257, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6219320893287659, Accuracy = 0.8854250907897949
    Iter #769024:  Learning rate = 0.003757:   Batch Loss = 0.225383, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6189579963684082, Accuracy = 0.887854278087616
    Iter #769536:  Learning rate = 0.003757:   Batch Loss = 0.221125, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6210482120513916, Accuracy = 0.8862348198890686
    Iter #770048:  Learning rate = 0.003757:   Batch Loss = 0.224207, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6258230209350586, Accuracy = 0.8842105269432068
    Iter #770560:  Learning rate = 0.003757:   Batch Loss = 0.220146, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6234079599380493, Accuracy = 0.8894736766815186
    Iter #771072:  Learning rate = 0.003757:   Batch Loss = 0.222565, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6241489052772522, Accuracy = 0.8882591128349304
    Iter #771584:  Learning rate = 0.003757:   Batch Loss = 0.221589, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6235386729240417, Accuracy = 0.8825910687446594
    Iter #772096:  Learning rate = 0.003757:   Batch Loss = 0.215018, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6198094487190247, Accuracy = 0.8862348198890686
    Iter #772608:  Learning rate = 0.003757:   Batch Loss = 0.219437, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.617557942867279, Accuracy = 0.8902834057807922
    Iter #773120:  Learning rate = 0.003757:   Batch Loss = 0.220743, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6191353797912598, Accuracy = 0.8886639475822449
    Iter #773632:  Learning rate = 0.003757:   Batch Loss = 0.217791, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.610750138759613, Accuracy = 0.8890688419342041
    Iter #774144:  Learning rate = 0.003757:   Batch Loss = 0.213751, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6132993698120117, Accuracy = 0.8874493837356567
    Iter #774656:  Learning rate = 0.003757:   Batch Loss = 0.215760, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6190652251243591, Accuracy = 0.8850202560424805
    Iter #775168:  Learning rate = 0.003757:   Batch Loss = 0.214487, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6155601739883423, Accuracy = 0.8866396546363831
    Iter #775680:  Learning rate = 0.003757:   Batch Loss = 0.211770, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6148077249526978, Accuracy = 0.8862348198890686
    Iter #776192:  Learning rate = 0.003757:   Batch Loss = 0.220023, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6168610453605652, Accuracy = 0.8886639475822449
    Iter #776704:  Learning rate = 0.003757:   Batch Loss = 0.209595, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6271085739135742, Accuracy = 0.8850202560424805
    Iter #777216:  Learning rate = 0.003757:   Batch Loss = 0.212123, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6287964582443237, Accuracy = 0.8866396546363831
    Iter #777728:  Learning rate = 0.003757:   Batch Loss = 0.213718, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6233230829238892, Accuracy = 0.8850202560424805
    Iter #778240:  Learning rate = 0.003757:   Batch Loss = 0.211260, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6207026243209839, Accuracy = 0.8854250907897949
    Iter #778752:  Learning rate = 0.003757:   Batch Loss = 0.211383, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6250487565994263, Accuracy = 0.8838056921958923
    Iter #779264:  Learning rate = 0.003757:   Batch Loss = 0.210031, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6168184280395508, Accuracy = 0.8874493837356567
    Iter #779776:  Learning rate = 0.003757:   Batch Loss = 0.210520, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6146635413169861, Accuracy = 0.8890688419342041
    Iter #780288:  Learning rate = 0.003757:   Batch Loss = 0.214375, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6254042983055115, Accuracy = 0.8858299851417542
    Iter #780800:  Learning rate = 0.003757:   Batch Loss = 0.206348, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6221330761909485, Accuracy = 0.8838056921958923
    Iter #781312:  Learning rate = 0.003757:   Batch Loss = 0.207334, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6164985299110413, Accuracy = 0.8862348198890686
    Iter #781824:  Learning rate = 0.003757:   Batch Loss = 0.206652, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6250088214874268, Accuracy = 0.8854250907897949
    Iter #782336:  Learning rate = 0.003757:   Batch Loss = 0.205726, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6162602305412292, Accuracy = 0.8874493837356567
    Iter #782848:  Learning rate = 0.003757:   Batch Loss = 0.203838, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6151883602142334, Accuracy = 0.8846153616905212
    Iter #783360:  Learning rate = 0.003757:   Batch Loss = 0.207544, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6152450442314148, Accuracy = 0.8854250907897949
    Iter #783872:  Learning rate = 0.003757:   Batch Loss = 0.209959, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6170600056648254, Accuracy = 0.8842105269432068
    Iter #784384:  Learning rate = 0.003757:   Batch Loss = 0.204222, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6172679662704468, Accuracy = 0.8838056921958923
    Iter #784896:  Learning rate = 0.003757:   Batch Loss = 0.202709, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6177816390991211, Accuracy = 0.8825910687446594
    Iter #785408:  Learning rate = 0.003757:   Batch Loss = 0.201557, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6154298782348633, Accuracy = 0.8870445489883423
    Iter #785920:  Learning rate = 0.003757:   Batch Loss = 0.202097, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.613923966884613, Accuracy = 0.887854278087616
    Iter #786432:  Learning rate = 0.003757:   Batch Loss = 0.206204, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6173082590103149, Accuracy = 0.8850202560424805
    Iter #786944:  Learning rate = 0.003757:   Batch Loss = 0.199743, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6214779019355774, Accuracy = 0.8858299851417542
    Iter #787456:  Learning rate = 0.003757:   Batch Loss = 0.198721, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6313866376876831, Accuracy = 0.8801619410514832
    Iter #787968:  Learning rate = 0.003757:   Batch Loss = 0.197215, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6231392025947571, Accuracy = 0.8834007978439331
    Iter #788480:  Learning rate = 0.003757:   Batch Loss = 0.198249, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6319882273674011, Accuracy = 0.8785424828529358
    Iter #788992:  Learning rate = 0.003757:   Batch Loss = 0.201094, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6396669149398804, Accuracy = 0.8781376481056213
    Iter #789504:  Learning rate = 0.003757:   Batch Loss = 0.198140, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6216875910758972, Accuracy = 0.8850202560424805
    Iter #790016:  Learning rate = 0.003757:   Batch Loss = 0.202890, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.627461850643158, Accuracy = 0.8834007978439331
    Iter #790528:  Learning rate = 0.003757:   Batch Loss = 0.198641, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6308206915855408, Accuracy = 0.8801619410514832
    Iter #791040:  Learning rate = 0.003757:   Batch Loss = 0.201459, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6224995851516724, Accuracy = 0.8797571063041687
    Iter #791552:  Learning rate = 0.003757:   Batch Loss = 0.203818, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6246780753135681, Accuracy = 0.8817813992500305
    Iter #792064:  Learning rate = 0.003757:   Batch Loss = 0.197494, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6271606683731079, Accuracy = 0.8813765048980713
    Iter #792576:  Learning rate = 0.003757:   Batch Loss = 0.196137, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.628017783164978, Accuracy = 0.8825910687446594
    Iter #793088:  Learning rate = 0.003757:   Batch Loss = 0.200222, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6199381351470947, Accuracy = 0.8838056921958923
    Iter #793600:  Learning rate = 0.003757:   Batch Loss = 0.204313, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6630791425704956, Accuracy = 0.8696356415748596
    Iter #794112:  Learning rate = 0.003757:   Batch Loss = 0.201628, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6602073907852173, Accuracy = 0.8744939565658569
    Iter #794624:  Learning rate = 0.003757:   Batch Loss = 0.223041, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7308076024055481, Accuracy = 0.8538461327552795
    Iter #795136:  Learning rate = 0.003757:   Batch Loss = 0.241788, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7213209867477417, Accuracy = 0.8457489609718323
    Iter #795648:  Learning rate = 0.003757:   Batch Loss = 0.297419, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7734795212745667, Accuracy = 0.8376518487930298
    Iter #796160:  Learning rate = 0.003757:   Batch Loss = 0.373955, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8789154291152954, Accuracy = 0.8008097410202026
    Iter #796672:  Learning rate = 0.003757:   Batch Loss = 0.581187, Accuracy = 0.875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8861551880836487, Accuracy = 0.7931174039840698
    Iter #797184:  Learning rate = 0.003757:   Batch Loss = 0.618082, Accuracy = 0.84375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9052373766899109, Accuracy = 0.7846153974533081
    Iter #797696:  Learning rate = 0.003757:   Batch Loss = 0.486657, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8800419569015503, Accuracy = 0.8008097410202026
    Iter #798208:  Learning rate = 0.003757:   Batch Loss = 0.570243, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9545247554779053, Accuracy = 0.794331967830658
    Iter #798720:  Learning rate = 0.003757:   Batch Loss = 0.531251, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8973947167396545, Accuracy = 0.8072874546051025
    Iter #799232:  Learning rate = 0.003757:   Batch Loss = 0.465983, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.915595293045044, Accuracy = 0.7963562607765198
    Iter #799744:  Learning rate = 0.003757:   Batch Loss = 0.390607, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9982441067695618, Accuracy = 0.78340083360672
    Iter #800256:  Learning rate = 0.003607:   Batch Loss = 0.539608, Accuracy = 0.859375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9762871265411377, Accuracy = 0.7846153974533081
    Iter #800768:  Learning rate = 0.003607:   Batch Loss = 0.469366, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0115044116973877, Accuracy = 0.768825888633728
    Iter #801280:  Learning rate = 0.003607:   Batch Loss = 0.640861, Accuracy = 0.875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9025483131408691, Accuracy = 0.7955465316772461
    Iter #801792:  Learning rate = 0.003607:   Batch Loss = 0.481626, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8667024374008179, Accuracy = 0.8085020184516907
    Iter #802304:  Learning rate = 0.003607:   Batch Loss = 0.443496, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9335747361183167, Accuracy = 0.7902833819389343
    Iter #802816:  Learning rate = 0.003607:   Batch Loss = 0.499717, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.871681809425354, Accuracy = 0.8230769038200378
    Iter #803328:  Learning rate = 0.003607:   Batch Loss = 0.355582, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.848281741142273, Accuracy = 0.8190283179283142
    Iter #803840:  Learning rate = 0.003607:   Batch Loss = 0.528178, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9585357904434204, Accuracy = 0.7898785471916199
    Iter #804352:  Learning rate = 0.003607:   Batch Loss = 0.382463, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8577735424041748, Accuracy = 0.8109311461448669
    Iter #804864:  Learning rate = 0.003607:   Batch Loss = 0.452681, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.865795373916626, Accuracy = 0.8275303840637207
    Iter #805376:  Learning rate = 0.003607:   Batch Loss = 0.365482, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8323297500610352, Accuracy = 0.8384615182876587
    Iter #805888:  Learning rate = 0.003607:   Batch Loss = 0.351022, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8590368032455444, Accuracy = 0.8222672343254089
    Iter #806400:  Learning rate = 0.003607:   Batch Loss = 0.391889, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8238739967346191, Accuracy = 0.8348178267478943
    Iter #806912:  Learning rate = 0.003607:   Batch Loss = 0.388432, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.794823169708252, Accuracy = 0.8457489609718323
    Iter #807424:  Learning rate = 0.003607:   Batch Loss = 0.334433, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7717304229736328, Accuracy = 0.846558690071106
    Iter #807936:  Learning rate = 0.003607:   Batch Loss = 0.309675, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7633694410324097, Accuracy = 0.8570850491523743
    Iter #808448:  Learning rate = 0.003607:   Batch Loss = 0.333930, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7899601459503174, Accuracy = 0.8497975468635559
    Iter #808960:  Learning rate = 0.003607:   Batch Loss = 0.315611, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7857534885406494, Accuracy = 0.8574898838996887
    Iter #809472:  Learning rate = 0.003607:   Batch Loss = 0.373399, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8062624931335449, Accuracy = 0.8461538553237915
    Iter #809984:  Learning rate = 0.003607:   Batch Loss = 0.351449, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7622522115707397, Accuracy = 0.8562753200531006
    Iter #810496:  Learning rate = 0.003607:   Batch Loss = 0.366618, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7950131297111511, Accuracy = 0.8469635844230652
    Iter #811008:  Learning rate = 0.003607:   Batch Loss = 0.339402, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8024811744689941, Accuracy = 0.8441295623779297
    Iter #811520:  Learning rate = 0.003607:   Batch Loss = 0.349529, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7762205004692078, Accuracy = 0.8514170050621033
    Iter #812032:  Learning rate = 0.003607:   Batch Loss = 0.334984, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7360385656356812, Accuracy = 0.8720647692680359
    Iter #812544:  Learning rate = 0.003607:   Batch Loss = 0.386911, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7813782691955566, Accuracy = 0.8514170050621033
    Iter #813056:  Learning rate = 0.003607:   Batch Loss = 0.393235, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7793273329734802, Accuracy = 0.852226734161377
    Iter #813568:  Learning rate = 0.003607:   Batch Loss = 0.320512, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7729201316833496, Accuracy = 0.8530364632606506
    Iter #814080:  Learning rate = 0.003607:   Batch Loss = 0.364122, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7696847915649414, Accuracy = 0.8550607562065125
    Iter #814592:  Learning rate = 0.003607:   Batch Loss = 0.345354, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.785892903804779, Accuracy = 0.846558690071106
    Iter #815104:  Learning rate = 0.003607:   Batch Loss = 0.392266, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7812771797180176, Accuracy = 0.8518218398094177
    Iter #815616:  Learning rate = 0.003607:   Batch Loss = 0.312899, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7740342020988464, Accuracy = 0.8550607562065125
    Iter #816128:  Learning rate = 0.003607:   Batch Loss = 0.391191, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7756383419036865, Accuracy = 0.862348198890686
    Iter #816640:  Learning rate = 0.003607:   Batch Loss = 0.306249, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7501733303070068, Accuracy = 0.8651821613311768
    Iter #817152:  Learning rate = 0.003607:   Batch Loss = 0.294355, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7430112361907959, Accuracy = 0.8639675974845886
    Iter #817664:  Learning rate = 0.003607:   Batch Loss = 0.270876, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7092064023017883, Accuracy = 0.8797571063041687
    Iter #818176:  Learning rate = 0.003607:   Batch Loss = 0.366150, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7594487071037292, Accuracy = 0.8647773265838623
    Iter #818688:  Learning rate = 0.003607:   Batch Loss = 0.280361, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7327235341072083, Accuracy = 0.8712550401687622
    Iter #819200:  Learning rate = 0.003607:   Batch Loss = 0.276530, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7220193147659302, Accuracy = 0.8700404763221741
    Iter #819712:  Learning rate = 0.003607:   Batch Loss = 0.317562, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7162739038467407, Accuracy = 0.8696356415748596
    Iter #820224:  Learning rate = 0.003607:   Batch Loss = 0.291442, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7115088701248169, Accuracy = 0.8708502054214478
    Iter #820736:  Learning rate = 0.003607:   Batch Loss = 0.312370, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7225046753883362, Accuracy = 0.8724696636199951
    Iter #821248:  Learning rate = 0.003607:   Batch Loss = 0.397362, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6997374296188354, Accuracy = 0.8740890622138977
    Iter #821760:  Learning rate = 0.003607:   Batch Loss = 0.276322, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7428208589553833, Accuracy = 0.8668016195297241
    Iter #822272:  Learning rate = 0.003607:   Batch Loss = 0.314829, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7287434339523315, Accuracy = 0.8680161833763123
    Iter #822784:  Learning rate = 0.003607:   Batch Loss = 0.330977, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6985458135604858, Accuracy = 0.8757085204124451
    Iter #823296:  Learning rate = 0.003607:   Batch Loss = 0.312950, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6975985169410706, Accuracy = 0.8744939565658569
    Iter #823808:  Learning rate = 0.003607:   Batch Loss = 0.291603, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6915740966796875, Accuracy = 0.8720647692680359
    Iter #824320:  Learning rate = 0.003607:   Batch Loss = 0.263693, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6943587064743042, Accuracy = 0.8757085204124451
    Iter #824832:  Learning rate = 0.003607:   Batch Loss = 0.290952, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6734256148338318, Accuracy = 0.8781376481056213
    Iter #825344:  Learning rate = 0.003607:   Batch Loss = 0.266350, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6814810037612915, Accuracy = 0.8773279190063477
    Iter #825856:  Learning rate = 0.003607:   Batch Loss = 0.277040, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7003605365753174, Accuracy = 0.8716599345207214
    Iter #826368:  Learning rate = 0.003607:   Batch Loss = 0.265278, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.704251229763031, Accuracy = 0.8736842274665833
    Iter #826880:  Learning rate = 0.003607:   Batch Loss = 0.257701, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6687899827957153, Accuracy = 0.8797571063041687
    Iter #827392:  Learning rate = 0.003607:   Batch Loss = 0.247026, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6689802408218384, Accuracy = 0.8813765048980713
    Iter #827904:  Learning rate = 0.003607:   Batch Loss = 0.252218, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6652617454528809, Accuracy = 0.8809716701507568
    Iter #828416:  Learning rate = 0.003607:   Batch Loss = 0.261900, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6567939519882202, Accuracy = 0.882186233997345
    Iter #828928:  Learning rate = 0.003607:   Batch Loss = 0.243749, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.653287410736084, Accuracy = 0.8838056921958923
    Iter #829440:  Learning rate = 0.003607:   Batch Loss = 0.241463, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6526479125022888, Accuracy = 0.8825910687446594
    Iter #829952:  Learning rate = 0.003607:   Batch Loss = 0.248057, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6414490938186646, Accuracy = 0.8894736766815186
    Iter #830464:  Learning rate = 0.003607:   Batch Loss = 0.242302, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6384048461914062, Accuracy = 0.8890688419342041
    Iter #830976:  Learning rate = 0.003607:   Batch Loss = 0.242752, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6270191669464111, Accuracy = 0.8919028043746948
    Iter #831488:  Learning rate = 0.003607:   Batch Loss = 0.241699, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6250320672988892, Accuracy = 0.8919028043746948
    Iter #832000:  Learning rate = 0.003607:   Batch Loss = 0.235554, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6290730834007263, Accuracy = 0.8910931348800659
    Iter #832512:  Learning rate = 0.003607:   Batch Loss = 0.237791, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6292489767074585, Accuracy = 0.8898785710334778
    Iter #833024:  Learning rate = 0.003607:   Batch Loss = 0.234318, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6255685091018677, Accuracy = 0.887854278087616
    Iter #833536:  Learning rate = 0.003607:   Batch Loss = 0.233670, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6201149821281433, Accuracy = 0.8902834057807922
    Iter #834048:  Learning rate = 0.003607:   Batch Loss = 0.236637, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6176936626434326, Accuracy = 0.8914979696273804
    Iter #834560:  Learning rate = 0.003607:   Batch Loss = 0.231130, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6184903383255005, Accuracy = 0.8882591128349304
    Iter #835072:  Learning rate = 0.003607:   Batch Loss = 0.232587, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6267009377479553, Accuracy = 0.8850202560424805
    Iter #835584:  Learning rate = 0.003607:   Batch Loss = 0.230169, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6272765398025513, Accuracy = 0.8862348198890686
    Iter #836096:  Learning rate = 0.003607:   Batch Loss = 0.227249, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6233192086219788, Accuracy = 0.887854278087616
    Iter #836608:  Learning rate = 0.003607:   Batch Loss = 0.230786, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6199302673339844, Accuracy = 0.8894736766815186
    Iter #837120:  Learning rate = 0.003607:   Batch Loss = 0.221691, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6222269535064697, Accuracy = 0.8890688419342041
    Iter #837632:  Learning rate = 0.003607:   Batch Loss = 0.226328, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6245827078819275, Accuracy = 0.8850202560424805
    Iter #838144:  Learning rate = 0.003607:   Batch Loss = 0.225582, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6187497973442078, Accuracy = 0.8866396546363831
    Iter #838656:  Learning rate = 0.003607:   Batch Loss = 0.224597, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.612754225730896, Accuracy = 0.8886639475822449
    Iter #839168:  Learning rate = 0.003607:   Batch Loss = 0.222621, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.612097978591919, Accuracy = 0.8906882405281067
    Iter #839680:  Learning rate = 0.003607:   Batch Loss = 0.216285, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6183184385299683, Accuracy = 0.8882591128349304
    Iter #840192:  Learning rate = 0.003607:   Batch Loss = 0.221226, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.622380793094635, Accuracy = 0.8829959630966187
    Iter #840704:  Learning rate = 0.003607:   Batch Loss = 0.218181, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6202399730682373, Accuracy = 0.8842105269432068
    Iter #841216:  Learning rate = 0.003607:   Batch Loss = 0.220507, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.628217339515686, Accuracy = 0.8829959630966187
    Iter #841728:  Learning rate = 0.003607:   Batch Loss = 0.222258, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6293359994888306, Accuracy = 0.8829959630966187
    Iter #842240:  Learning rate = 0.003607:   Batch Loss = 0.217816, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6206743121147156, Accuracy = 0.882186233997345
    Iter #842752:  Learning rate = 0.003607:   Batch Loss = 0.216800, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6179869174957275, Accuracy = 0.8817813992500305
    Iter #843264:  Learning rate = 0.003607:   Batch Loss = 0.217660, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6137446165084839, Accuracy = 0.8858299851417542
    Iter #843776:  Learning rate = 0.003607:   Batch Loss = 0.214466, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6228638291358948, Accuracy = 0.8846153616905212
    Iter #844288:  Learning rate = 0.003607:   Batch Loss = 0.215272, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6182065010070801, Accuracy = 0.8854250907897949
    Iter #844800:  Learning rate = 0.003607:   Batch Loss = 0.214742, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6116174459457397, Accuracy = 0.8882591128349304
    Iter #845312:  Learning rate = 0.003607:   Batch Loss = 0.214227, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6134204864501953, Accuracy = 0.8874493837356567
    Iter #845824:  Learning rate = 0.003607:   Batch Loss = 0.219093, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6143366694450378, Accuracy = 0.8850202560424805
    Iter #846336:  Learning rate = 0.003607:   Batch Loss = 0.218069, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6171005964279175, Accuracy = 0.8809716701507568
    Iter #846848:  Learning rate = 0.003607:   Batch Loss = 0.214391, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6215915679931641, Accuracy = 0.8829959630966187
    Iter #847360:  Learning rate = 0.003607:   Batch Loss = 0.214128, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6157245635986328, Accuracy = 0.8850202560424805
    Iter #847872:  Learning rate = 0.003607:   Batch Loss = 0.210143, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.612060546875, Accuracy = 0.8842105269432068
    Iter #848384:  Learning rate = 0.003607:   Batch Loss = 0.207393, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.613193929195404, Accuracy = 0.8842105269432068
    Iter #848896:  Learning rate = 0.003607:   Batch Loss = 0.206152, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6148611903190613, Accuracy = 0.8842105269432068
    Iter #849408:  Learning rate = 0.003607:   Batch Loss = 0.214180, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6200504302978516, Accuracy = 0.882186233997345
    Iter #849920:  Learning rate = 0.003607:   Batch Loss = 0.203657, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6245855689048767, Accuracy = 0.8809716701507568
    Iter #850432:  Learning rate = 0.003607:   Batch Loss = 0.208085, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6114668250083923, Accuracy = 0.8829959630966187
    Iter #850944:  Learning rate = 0.003607:   Batch Loss = 0.213738, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6062864065170288, Accuracy = 0.882186233997345
    Iter #851456:  Learning rate = 0.003607:   Batch Loss = 0.200190, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6153634786605835, Accuracy = 0.8809716701507568
    Iter #851968:  Learning rate = 0.003607:   Batch Loss = 0.203323, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6149640083312988, Accuracy = 0.8829959630966187
    Iter #852480:  Learning rate = 0.003607:   Batch Loss = 0.203106, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6099096536636353, Accuracy = 0.8817813992500305
    Iter #852992:  Learning rate = 0.003607:   Batch Loss = 0.203989, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6063962578773499, Accuracy = 0.8870445489883423
    Iter #853504:  Learning rate = 0.003607:   Batch Loss = 0.200913, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6090623140335083, Accuracy = 0.8846153616905212
    Iter #854016:  Learning rate = 0.003607:   Batch Loss = 0.205632, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6138007640838623, Accuracy = 0.8813765048980713
    Iter #854528:  Learning rate = 0.003607:   Batch Loss = 0.200323, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6114435791969299, Accuracy = 0.8809716701507568
    Iter #855040:  Learning rate = 0.003607:   Batch Loss = 0.200009, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6057816743850708, Accuracy = 0.8846153616905212
    Iter #855552:  Learning rate = 0.003607:   Batch Loss = 0.202230, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6092525124549866, Accuracy = 0.8825910687446594
    Iter #856064:  Learning rate = 0.003607:   Batch Loss = 0.201796, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6037726402282715, Accuracy = 0.8870445489883423
    Iter #856576:  Learning rate = 0.003607:   Batch Loss = 0.201796, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6096647381782532, Accuracy = 0.8858299851417542
    Iter #857088:  Learning rate = 0.003607:   Batch Loss = 0.202870, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6052485704421997, Accuracy = 0.8842105269432068
    Iter #857600:  Learning rate = 0.003607:   Batch Loss = 0.199999, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6041116714477539, Accuracy = 0.8858299851417542
    Iter #858112:  Learning rate = 0.003607:   Batch Loss = 0.198344, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6092435121536255, Accuracy = 0.8874493837356567
    Iter #858624:  Learning rate = 0.003607:   Batch Loss = 0.197051, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6158232688903809, Accuracy = 0.8846153616905212
    Iter #859136:  Learning rate = 0.003607:   Batch Loss = 0.200183, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6095410585403442, Accuracy = 0.8838056921958923
    Iter #859648:  Learning rate = 0.003607:   Batch Loss = 0.193787, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6184332966804504, Accuracy = 0.8817813992500305
    Iter #860160:  Learning rate = 0.003607:   Batch Loss = 0.196920, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.619225025177002, Accuracy = 0.8797571063041687
    Iter #860672:  Learning rate = 0.003607:   Batch Loss = 0.196280, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6119633913040161, Accuracy = 0.8817813992500305
    Iter #861184:  Learning rate = 0.003607:   Batch Loss = 0.193796, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6051980257034302, Accuracy = 0.887854278087616
    Iter #861696:  Learning rate = 0.003607:   Batch Loss = 0.195527, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6075482964515686, Accuracy = 0.8829959630966187
    Iter #862208:  Learning rate = 0.003607:   Batch Loss = 0.192167, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6044723987579346, Accuracy = 0.8846153616905212
    Iter #862720:  Learning rate = 0.003607:   Batch Loss = 0.192616, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6086450815200806, Accuracy = 0.8842105269432068
    Iter #863232:  Learning rate = 0.003607:   Batch Loss = 0.198236, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6132035255432129, Accuracy = 0.8809716701507568
    Iter #863744:  Learning rate = 0.003607:   Batch Loss = 0.196908, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6084775924682617, Accuracy = 0.8838056921958923
    Iter #864256:  Learning rate = 0.003607:   Batch Loss = 0.195675, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6040326356887817, Accuracy = 0.8870445489883423
    Iter #864768:  Learning rate = 0.003607:   Batch Loss = 0.195934, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6073468923568726, Accuracy = 0.8846153616905212
    Iter #865280:  Learning rate = 0.003607:   Batch Loss = 0.192175, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6086357831954956, Accuracy = 0.8854250907897949
    Iter #865792:  Learning rate = 0.003607:   Batch Loss = 0.196717, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6082093119621277, Accuracy = 0.8854250907897949
    Iter #866304:  Learning rate = 0.003607:   Batch Loss = 0.191984, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6188507676124573, Accuracy = 0.8846153616905212
    Iter #866816:  Learning rate = 0.003607:   Batch Loss = 0.192166, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6166751384735107, Accuracy = 0.8797571063041687
    Iter #867328:  Learning rate = 0.003607:   Batch Loss = 0.189097, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6330574154853821, Accuracy = 0.8724696636199951
    Iter #867840:  Learning rate = 0.003607:   Batch Loss = 0.195130, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6200677156448364, Accuracy = 0.8817813992500305
    Iter #868352:  Learning rate = 0.003607:   Batch Loss = 0.208334, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6224200129508972, Accuracy = 0.8817813992500305
    Iter #868864:  Learning rate = 0.003607:   Batch Loss = 0.219116, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6369656324386597, Accuracy = 0.8728744983673096
    Iter #869376:  Learning rate = 0.003607:   Batch Loss = 0.239938, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6517415046691895, Accuracy = 0.8720647692680359
    Iter #869888:  Learning rate = 0.003607:   Batch Loss = 0.256329, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7601866722106934, Accuracy = 0.840080976486206
    Iter #870400:  Learning rate = 0.003607:   Batch Loss = 0.344987, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7915321588516235, Accuracy = 0.8202429413795471
    Iter #870912:  Learning rate = 0.003607:   Batch Loss = 0.695044, Accuracy = 0.796875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9447256326675415, Accuracy = 0.7680162191390991
    Iter #871424:  Learning rate = 0.003607:   Batch Loss = 0.671189, Accuracy = 0.84375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8937883973121643, Accuracy = 0.8028340339660645
    Iter #871936:  Learning rate = 0.003607:   Batch Loss = 0.559898, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9042686223983765, Accuracy = 0.796761155128479
    Iter #872448:  Learning rate = 0.003607:   Batch Loss = 0.480266, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9240737557411194, Accuracy = 0.794331967830658
    Iter #872960:  Learning rate = 0.003607:   Batch Loss = 0.507760, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9535059928894043, Accuracy = 0.7797570824623108
    Iter #873472:  Learning rate = 0.003607:   Batch Loss = 0.578930, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9166699647903442, Accuracy = 0.7858299612998962
    Iter #873984:  Learning rate = 0.003607:   Batch Loss = 0.636405, Accuracy = 0.84375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8781771659851074, Accuracy = 0.8117408752441406
    Iter #874496:  Learning rate = 0.003607:   Batch Loss = 0.672298, Accuracy = 0.8125
    PERFORMANCE ON TEST SET:             Batch Loss = 1.008084774017334, Accuracy = 0.7761133313179016
    Iter #875008:  Learning rate = 0.003607:   Batch Loss = 0.564294, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 1.016222357749939, Accuracy = 0.758704423904419
    Iter #875520:  Learning rate = 0.003607:   Batch Loss = 0.689442, Accuracy = 0.875
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0576562881469727, Accuracy = 0.7615384459495544
    Iter #876032:  Learning rate = 0.003607:   Batch Loss = 0.529300, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.894970178604126, Accuracy = 0.8032388687133789
    Iter #876544:  Learning rate = 0.003607:   Batch Loss = 0.546039, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8639676570892334, Accuracy = 0.8214575052261353
    Iter #877056:  Learning rate = 0.003607:   Batch Loss = 0.382688, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8158428072929382, Accuracy = 0.8421052694320679
    Iter #877568:  Learning rate = 0.003607:   Batch Loss = 0.378083, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8190844655036926, Accuracy = 0.8388664126396179
    Iter #878080:  Learning rate = 0.003607:   Batch Loss = 0.369653, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.838692307472229, Accuracy = 0.8323886394500732
    Iter #878592:  Learning rate = 0.003607:   Batch Loss = 0.357529, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8016550540924072, Accuracy = 0.8461538553237915
    Iter #879104:  Learning rate = 0.003607:   Batch Loss = 0.451547, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8128308057785034, Accuracy = 0.8425101041793823
    Iter #879616:  Learning rate = 0.003607:   Batch Loss = 0.496257, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.865129292011261, Accuracy = 0.8336032629013062
    Iter #880128:  Learning rate = 0.003607:   Batch Loss = 0.431769, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8613156080245972, Accuracy = 0.8323886394500732
    Iter #880640:  Learning rate = 0.003607:   Batch Loss = 0.346028, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7814921736717224, Accuracy = 0.8445343971252441
    Iter #881152:  Learning rate = 0.003607:   Batch Loss = 0.295776, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8579879999160767, Accuracy = 0.8214575052261353
    Iter #881664:  Learning rate = 0.003607:   Batch Loss = 0.388198, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8090041875839233, Accuracy = 0.835627555847168
    Iter #882176:  Learning rate = 0.003607:   Batch Loss = 0.401053, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7564932107925415, Accuracy = 0.8534412980079651
    Iter #882688:  Learning rate = 0.003607:   Batch Loss = 0.299570, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7967839241027832, Accuracy = 0.8429149985313416
    Iter #883200:  Learning rate = 0.003607:   Batch Loss = 0.382455, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7988040447235107, Accuracy = 0.8396761417388916
    Iter #883712:  Learning rate = 0.003607:   Batch Loss = 0.316188, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7869308590888977, Accuracy = 0.8518218398094177
    Iter #884224:  Learning rate = 0.003607:   Batch Loss = 0.459465, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8390340805053711, Accuracy = 0.840080976486206
    Iter #884736:  Learning rate = 0.003607:   Batch Loss = 0.351985, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.810858964920044, Accuracy = 0.8404858112335205
    Iter #885248:  Learning rate = 0.003607:   Batch Loss = 0.433246, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8053659200668335, Accuracy = 0.848987877368927
    Iter #885760:  Learning rate = 0.003607:   Batch Loss = 0.363779, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7562771439552307, Accuracy = 0.8570850491523743
    Iter #886272:  Learning rate = 0.003607:   Batch Loss = 0.340573, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7669950127601624, Accuracy = 0.8469635844230652
    Iter #886784:  Learning rate = 0.003607:   Batch Loss = 0.332026, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8082598447799683, Accuracy = 0.8392712473869324
    Iter #887296:  Learning rate = 0.003607:   Batch Loss = 0.386628, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7762496471405029, Accuracy = 0.8562753200531006
    Iter #887808:  Learning rate = 0.003607:   Batch Loss = 0.317789, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7148697376251221, Accuracy = 0.8680161833763123
    Iter #888320:  Learning rate = 0.003607:   Batch Loss = 0.330652, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7469236254692078, Accuracy = 0.8619433045387268
    Iter #888832:  Learning rate = 0.003607:   Batch Loss = 0.331837, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7458477020263672, Accuracy = 0.8570850491523743
    Iter #889344:  Learning rate = 0.003607:   Batch Loss = 0.328278, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7300488352775574, Accuracy = 0.8688259124755859
    Iter #889856:  Learning rate = 0.003607:   Batch Loss = 0.282340, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7937426567077637, Accuracy = 0.8526315689086914
    Iter #890368:  Learning rate = 0.003607:   Batch Loss = 0.291400, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7994867563247681, Accuracy = 0.8453441262245178
    Iter #890880:  Learning rate = 0.003607:   Batch Loss = 0.381853, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7478686571121216, Accuracy = 0.8724696636199951
    Iter #891392:  Learning rate = 0.003607:   Batch Loss = 0.393881, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7229881882667542, Accuracy = 0.8680161833763123
    Iter #891904:  Learning rate = 0.003607:   Batch Loss = 0.348927, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.744696319103241, Accuracy = 0.8607287406921387
    Iter #892416:  Learning rate = 0.003607:   Batch Loss = 0.275813, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7354917526245117, Accuracy = 0.8668016195297241
    Iter #892928:  Learning rate = 0.003607:   Batch Loss = 0.327109, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7246567010879517, Accuracy = 0.8688259124755859
    Iter #893440:  Learning rate = 0.003607:   Batch Loss = 0.274811, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6908469200134277, Accuracy = 0.8809716701507568
    Iter #893952:  Learning rate = 0.003607:   Batch Loss = 0.272670, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6980535387992859, Accuracy = 0.8825910687446594
    Iter #894464:  Learning rate = 0.003607:   Batch Loss = 0.284686, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6937124133110046, Accuracy = 0.8793522119522095
    Iter #894976:  Learning rate = 0.003607:   Batch Loss = 0.288305, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6831291913986206, Accuracy = 0.8838056921958923
    Iter #895488:  Learning rate = 0.003607:   Batch Loss = 0.256112, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6769515872001648, Accuracy = 0.8854250907897949
    Iter #896000:  Learning rate = 0.003607:   Batch Loss = 0.270002, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6659411191940308, Accuracy = 0.8850202560424805
    Iter #896512:  Learning rate = 0.003607:   Batch Loss = 0.263102, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6481359004974365, Accuracy = 0.8910931348800659
    Iter #897024:  Learning rate = 0.003607:   Batch Loss = 0.255736, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.646530270576477, Accuracy = 0.8919028043746948
    Iter #897536:  Learning rate = 0.003607:   Batch Loss = 0.249044, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.653349757194519, Accuracy = 0.8902834057807922
    Iter #898048:  Learning rate = 0.003607:   Batch Loss = 0.247460, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6420527100563049, Accuracy = 0.8927125334739685
    Iter #898560:  Learning rate = 0.003607:   Batch Loss = 0.241588, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6402899026870728, Accuracy = 0.8882591128349304
    Iter #899072:  Learning rate = 0.003607:   Batch Loss = 0.235677, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6297194361686707, Accuracy = 0.895546555519104
    Iter #899584:  Learning rate = 0.003607:   Batch Loss = 0.244472, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6327965259552002, Accuracy = 0.8939270973205566
    Iter #900096:  Learning rate = 0.003463:   Batch Loss = 0.240940, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6310769319534302, Accuracy = 0.8927125334739685
    Iter #900608:  Learning rate = 0.003463:   Batch Loss = 0.237044, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6309504508972168, Accuracy = 0.8935222625732422
    Iter #901120:  Learning rate = 0.003463:   Batch Loss = 0.234475, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6252723932266235, Accuracy = 0.8979756832122803
    Iter #901632:  Learning rate = 0.003463:   Batch Loss = 0.230971, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6180540323257446, Accuracy = 0.8971660137176514
    Iter #902144:  Learning rate = 0.003463:   Batch Loss = 0.231231, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6109206080436707, Accuracy = 0.8975708484649658
    Iter #902656:  Learning rate = 0.003463:   Batch Loss = 0.238614, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6081623435020447, Accuracy = 0.8975708484649658
    Iter #903168:  Learning rate = 0.003463:   Batch Loss = 0.229529, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6110941171646118, Accuracy = 0.895546555519104
    Iter #903680:  Learning rate = 0.003463:   Batch Loss = 0.230028, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6138513088226318, Accuracy = 0.8963562846183777
    Iter #904192:  Learning rate = 0.003463:   Batch Loss = 0.226235, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6108144521713257, Accuracy = 0.8975708484649658
    Iter #904704:  Learning rate = 0.003463:   Batch Loss = 0.224808, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6031273007392883, Accuracy = 0.8979756832122803
    Iter #905216:  Learning rate = 0.003463:   Batch Loss = 0.229292, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6036152839660645, Accuracy = 0.8967611193656921
    Iter #905728:  Learning rate = 0.003463:   Batch Loss = 0.227529, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6045796871185303, Accuracy = 0.898785412311554
    Iter #906240:  Learning rate = 0.003463:   Batch Loss = 0.225948, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6072863936424255, Accuracy = 0.8971660137176514
    Iter #906752:  Learning rate = 0.003463:   Batch Loss = 0.220806, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6095012426376343, Accuracy = 0.8967611193656921
    Iter #907264:  Learning rate = 0.003463:   Batch Loss = 0.223084, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6094368100166321, Accuracy = 0.8943319916725159
    Iter #907776:  Learning rate = 0.003463:   Batch Loss = 0.223010, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6070426106452942, Accuracy = 0.8951417207717896
    Iter #908288:  Learning rate = 0.003463:   Batch Loss = 0.225323, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6057804226875305, Accuracy = 0.895546555519104
    Iter #908800:  Learning rate = 0.003463:   Batch Loss = 0.224810, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6111400723457336, Accuracy = 0.8939270973205566
    Iter #909312:  Learning rate = 0.003463:   Batch Loss = 0.215014, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6125591993331909, Accuracy = 0.8927125334739685
    Iter #909824:  Learning rate = 0.003463:   Batch Loss = 0.215208, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6044191718101501, Accuracy = 0.8947368264198303
    Iter #910336:  Learning rate = 0.003463:   Batch Loss = 0.219799, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.599533200263977, Accuracy = 0.8971660137176514
    Iter #910848:  Learning rate = 0.003463:   Batch Loss = 0.215693, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5977648496627808, Accuracy = 0.8943319916725159
    Iter #911360:  Learning rate = 0.003463:   Batch Loss = 0.216154, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5990461111068726, Accuracy = 0.8943319916725159
    Iter #911872:  Learning rate = 0.003463:   Batch Loss = 0.212097, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6032261848449707, Accuracy = 0.8967611193656921
    Iter #912384:  Learning rate = 0.003463:   Batch Loss = 0.210412, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6022911071777344, Accuracy = 0.8979756832122803
    Iter #912896:  Learning rate = 0.003463:   Batch Loss = 0.217784, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5972046256065369, Accuracy = 0.8959513902664185
    Iter #913408:  Learning rate = 0.003463:   Batch Loss = 0.215789, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5951888561248779, Accuracy = 0.8983805775642395
    Iter #913920:  Learning rate = 0.003463:   Batch Loss = 0.215137, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5969358682632446, Accuracy = 0.8991903066635132
    Iter #914432:  Learning rate = 0.003463:   Batch Loss = 0.215867, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6026538610458374, Accuracy = 0.8947368264198303
    Iter #914944:  Learning rate = 0.003463:   Batch Loss = 0.209465, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6017173528671265, Accuracy = 0.8975708484649658
    Iter #915456:  Learning rate = 0.003463:   Batch Loss = 0.209709, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6003923416137695, Accuracy = 0.8927125334739685
    Iter #915968:  Learning rate = 0.003463:   Batch Loss = 0.213013, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.597874104976654, Accuracy = 0.8943319916725159
    Iter #916480:  Learning rate = 0.003463:   Batch Loss = 0.208562, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5975958704948425, Accuracy = 0.8931174278259277
    Iter #916992:  Learning rate = 0.003463:   Batch Loss = 0.210994, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5993234515190125, Accuracy = 0.8951417207717896
    Iter #917504:  Learning rate = 0.003463:   Batch Loss = 0.208122, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.60107421875, Accuracy = 0.895546555519104
    Iter #918016:  Learning rate = 0.003463:   Batch Loss = 0.209511, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5991670489311218, Accuracy = 0.8963562846183777
    Iter #918528:  Learning rate = 0.003463:   Batch Loss = 0.208304, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6024222373962402, Accuracy = 0.8947368264198303
    Iter #919040:  Learning rate = 0.003463:   Batch Loss = 0.207837, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5931016206741333, Accuracy = 0.8983805775642395
    Iter #919552:  Learning rate = 0.003463:   Batch Loss = 0.209952, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5901271104812622, Accuracy = 0.8979756832122803
    Iter #920064:  Learning rate = 0.003463:   Batch Loss = 0.207253, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.59040367603302, Accuracy = 0.8963562846183777
    Iter #920576:  Learning rate = 0.003463:   Batch Loss = 0.200225, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5904661417007446, Accuracy = 0.8959513902664185
    Iter #921088:  Learning rate = 0.003463:   Batch Loss = 0.206929, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.590105414390564, Accuracy = 0.8959513902664185
    Iter #921600:  Learning rate = 0.003463:   Batch Loss = 0.202589, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6000865697860718, Accuracy = 0.892307698726654
    Iter #922112:  Learning rate = 0.003463:   Batch Loss = 0.207046, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6008761525154114, Accuracy = 0.8943319916725159
    Iter #922624:  Learning rate = 0.003463:   Batch Loss = 0.203656, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.596815824508667, Accuracy = 0.8931174278259277
    Iter #923136:  Learning rate = 0.003463:   Batch Loss = 0.204417, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5990169048309326, Accuracy = 0.8927125334739685
    Iter #923648:  Learning rate = 0.003463:   Batch Loss = 0.200869, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6042436361312866, Accuracy = 0.8939270973205566
    Iter #924160:  Learning rate = 0.003463:   Batch Loss = 0.201667, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5995925664901733, Accuracy = 0.895546555519104
    Iter #924672:  Learning rate = 0.003463:   Batch Loss = 0.199132, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5928804278373718, Accuracy = 0.8931174278259277
    Iter #925184:  Learning rate = 0.003463:   Batch Loss = 0.202499, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.596842885017395, Accuracy = 0.8931174278259277
    Iter #925696:  Learning rate = 0.003463:   Batch Loss = 0.200917, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5950576663017273, Accuracy = 0.8927125334739685
    Iter #926208:  Learning rate = 0.003463:   Batch Loss = 0.195805, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5920028686523438, Accuracy = 0.8910931348800659
    Iter #926720:  Learning rate = 0.003463:   Batch Loss = 0.194426, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5870926380157471, Accuracy = 0.8935222625732422
    Iter #927232:  Learning rate = 0.003463:   Batch Loss = 0.198057, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5870876908302307, Accuracy = 0.895546555519104
    Iter #927744:  Learning rate = 0.003463:   Batch Loss = 0.199142, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5907297134399414, Accuracy = 0.8939270973205566
    Iter #928256:  Learning rate = 0.003463:   Batch Loss = 0.195954, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5900246500968933, Accuracy = 0.8943319916725159
    Iter #928768:  Learning rate = 0.003463:   Batch Loss = 0.208285, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5919198989868164, Accuracy = 0.8902834057807922
    Iter #929280:  Learning rate = 0.003463:   Batch Loss = 0.200656, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5961956977844238, Accuracy = 0.8943319916725159
    Iter #929792:  Learning rate = 0.003463:   Batch Loss = 0.202851, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5860117673873901, Accuracy = 0.8983805775642395
    Iter #930304:  Learning rate = 0.003463:   Batch Loss = 0.197013, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5988614559173584, Accuracy = 0.8935222625732422
    Iter #930816:  Learning rate = 0.003463:   Batch Loss = 0.198437, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5903268456459045, Accuracy = 0.8971660137176514
    Iter #931328:  Learning rate = 0.003463:   Batch Loss = 0.199416, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5962570905685425, Accuracy = 0.8947368264198303
    Iter #931840:  Learning rate = 0.003463:   Batch Loss = 0.195674, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5970045924186707, Accuracy = 0.8902834057807922
    Iter #932352:  Learning rate = 0.003463:   Batch Loss = 0.202522, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6507571935653687, Accuracy = 0.8781376481056213
    Iter #932864:  Learning rate = 0.003463:   Batch Loss = 0.250529, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7574242949485779, Accuracy = 0.846558690071106
    Iter #933376:  Learning rate = 0.003463:   Batch Loss = 0.266157, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7969530820846558, Accuracy = 0.8251011967658997
    Iter #933888:  Learning rate = 0.003463:   Batch Loss = 0.429443, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8680700659751892, Accuracy = 0.8085020184516907
    Iter #934400:  Learning rate = 0.003463:   Batch Loss = 0.445173, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8268235325813293, Accuracy = 0.8165991902351379
    Iter #934912:  Learning rate = 0.003463:   Batch Loss = 0.465316, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9828795790672302, Accuracy = 0.7761133313179016
    Iter #935424:  Learning rate = 0.003463:   Batch Loss = 0.637543, Accuracy = 0.875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8449400067329407, Accuracy = 0.8190283179283142
    Iter #935936:  Learning rate = 0.003463:   Batch Loss = 0.371723, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8409882187843323, Accuracy = 0.8089068531990051
    Iter #936448:  Learning rate = 0.003463:   Batch Loss = 0.447034, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8434828519821167, Accuracy = 0.8165991902351379
    Iter #936960:  Learning rate = 0.003463:   Batch Loss = 0.374242, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7992535829544067, Accuracy = 0.8230769038200378
    Iter #937472:  Learning rate = 0.003463:   Batch Loss = 0.380650, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7993295788764954, Accuracy = 0.8352226614952087
    Iter #937984:  Learning rate = 0.003463:   Batch Loss = 0.449942, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8028280735015869, Accuracy = 0.835627555847168
    Iter #938496:  Learning rate = 0.003463:   Batch Loss = 0.429994, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7646482586860657, Accuracy = 0.8493927121162415
    Iter #939008:  Learning rate = 0.003463:   Batch Loss = 0.421863, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7928377389907837, Accuracy = 0.8384615182876587
    Iter #939520:  Learning rate = 0.003463:   Batch Loss = 0.423613, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8422508835792542, Accuracy = 0.8238866329193115
    Iter #940032:  Learning rate = 0.003463:   Batch Loss = 0.522090, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8407605886459351, Accuracy = 0.8129554390907288
    Iter #940544:  Learning rate = 0.003463:   Batch Loss = 0.473832, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8394577503204346, Accuracy = 0.8283400535583496
    Iter #941056:  Learning rate = 0.003463:   Batch Loss = 0.342760, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7813186645507812, Accuracy = 0.8473684191703796
    Iter #941568:  Learning rate = 0.003463:   Batch Loss = 0.385297, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7602462768554688, Accuracy = 0.8461538553237915
    Iter #942080:  Learning rate = 0.003463:   Batch Loss = 0.364631, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7611119747161865, Accuracy = 0.8526315689086914
    Iter #942592:  Learning rate = 0.003463:   Batch Loss = 0.330775, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7756425738334656, Accuracy = 0.852226734161377
    Iter #943104:  Learning rate = 0.003463:   Batch Loss = 0.510367, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7555145621299744, Accuracy = 0.8558704257011414
    Iter #943616:  Learning rate = 0.003463:   Batch Loss = 0.264960, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7409967184066772, Accuracy = 0.8663967847824097
    Iter #944128:  Learning rate = 0.003463:   Batch Loss = 0.261260, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7003833055496216, Accuracy = 0.8781376481056213
    Iter #944640:  Learning rate = 0.003463:   Batch Loss = 0.306258, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7331551909446716, Accuracy = 0.8696356415748596
    Iter #945152:  Learning rate = 0.003463:   Batch Loss = 0.343040, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6863588094711304, Accuracy = 0.8748987913131714
    Iter #945664:  Learning rate = 0.003463:   Batch Loss = 0.268818, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7011559009552002, Accuracy = 0.8672064542770386
    Iter #946176:  Learning rate = 0.003463:   Batch Loss = 0.253035, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7228503227233887, Accuracy = 0.8728744983673096
    Iter #946688:  Learning rate = 0.003463:   Batch Loss = 0.270750, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6877874732017517, Accuracy = 0.8740890622138977
    Iter #947200:  Learning rate = 0.003463:   Batch Loss = 0.321200, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6740851402282715, Accuracy = 0.882186233997345
    Iter #947712:  Learning rate = 0.003463:   Batch Loss = 0.273299, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7131360769271851, Accuracy = 0.8740890622138977
    Iter #948224:  Learning rate = 0.003463:   Batch Loss = 0.269066, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7137120366096497, Accuracy = 0.8781376481056213
    Iter #948736:  Learning rate = 0.003463:   Batch Loss = 0.252222, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6597087383270264, Accuracy = 0.8817813992500305
    Iter #949248:  Learning rate = 0.003463:   Batch Loss = 0.245683, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6910122036933899, Accuracy = 0.8761133551597595
    Iter #949760:  Learning rate = 0.003463:   Batch Loss = 0.257071, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6799039840698242, Accuracy = 0.8809716701507568
    Iter #950272:  Learning rate = 0.003463:   Batch Loss = 0.251311, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6549153923988342, Accuracy = 0.8886639475822449
    Iter #950784:  Learning rate = 0.003463:   Batch Loss = 0.234782, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6463702917098999, Accuracy = 0.8882591128349304
    Iter #951296:  Learning rate = 0.003463:   Batch Loss = 0.232964, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6529942750930786, Accuracy = 0.887854278087616
    Iter #951808:  Learning rate = 0.003463:   Batch Loss = 0.253040, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6522492170333862, Accuracy = 0.8870445489883423
    Iter #952320:  Learning rate = 0.003463:   Batch Loss = 0.240003, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.647368848323822, Accuracy = 0.8906882405281067
    Iter #952832:  Learning rate = 0.003463:   Batch Loss = 0.228645, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6434114575386047, Accuracy = 0.8910931348800659
    Iter #953344:  Learning rate = 0.003463:   Batch Loss = 0.229108, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.630828857421875, Accuracy = 0.8935222625732422
    Iter #953856:  Learning rate = 0.003463:   Batch Loss = 0.224867, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6264582872390747, Accuracy = 0.8935222625732422
    Iter #954368:  Learning rate = 0.003463:   Batch Loss = 0.223468, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.629124104976654, Accuracy = 0.8971660137176514
    Iter #954880:  Learning rate = 0.003463:   Batch Loss = 0.228685, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6358505487442017, Accuracy = 0.8914979696273804
    Iter #955392:  Learning rate = 0.003463:   Batch Loss = 0.221743, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6315465569496155, Accuracy = 0.8906882405281067
    Iter #955904:  Learning rate = 0.003463:   Batch Loss = 0.217403, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6181190609931946, Accuracy = 0.8951417207717896
    Iter #956416:  Learning rate = 0.003463:   Batch Loss = 0.218175, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6134163737297058, Accuracy = 0.8971660137176514
    Iter #956928:  Learning rate = 0.003463:   Batch Loss = 0.216583, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6048063039779663, Accuracy = 0.898785412311554
    Iter #957440:  Learning rate = 0.003463:   Batch Loss = 0.220968, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6078415513038635, Accuracy = 0.8967611193656921
    Iter #957952:  Learning rate = 0.003463:   Batch Loss = 0.215308, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6061861515045166, Accuracy = 0.8979756832122803
    Iter #958464:  Learning rate = 0.003463:   Batch Loss = 0.214526, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.599903404712677, Accuracy = 0.8979756832122803
    Iter #958976:  Learning rate = 0.003463:   Batch Loss = 0.213536, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5969541072845459, Accuracy = 0.8983805775642395
    Iter #959488:  Learning rate = 0.003463:   Batch Loss = 0.211285, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6011868119239807, Accuracy = 0.8975708484649658
    Iter #960000:  Learning rate = 0.003463:   Batch Loss = 0.208854, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6045204401016235, Accuracy = 0.8943319916725159
    Iter #960512:  Learning rate = 0.003463:   Batch Loss = 0.213812, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6059800386428833, Accuracy = 0.8939270973205566
    Iter #961024:  Learning rate = 0.003463:   Batch Loss = 0.207453, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6060789823532104, Accuracy = 0.8931174278259277
    Iter #961536:  Learning rate = 0.003463:   Batch Loss = 0.205171, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5989680886268616, Accuracy = 0.8951417207717896
    Iter #962048:  Learning rate = 0.003463:   Batch Loss = 0.207234, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5974653959274292, Accuracy = 0.895546555519104
    Iter #962560:  Learning rate = 0.003463:   Batch Loss = 0.203591, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5982592105865479, Accuracy = 0.8979756832122803
    Iter #963072:  Learning rate = 0.003463:   Batch Loss = 0.204909, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6007128357887268, Accuracy = 0.8963562846183777
    Iter #963584:  Learning rate = 0.003463:   Batch Loss = 0.208358, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5994596481323242, Accuracy = 0.8943319916725159
    Iter #964096:  Learning rate = 0.003463:   Batch Loss = 0.206946, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5997757911682129, Accuracy = 0.8947368264198303
    Iter #964608:  Learning rate = 0.003463:   Batch Loss = 0.204246, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5970668792724609, Accuracy = 0.8963562846183777
    Iter #965120:  Learning rate = 0.003463:   Batch Loss = 0.203034, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5956028699874878, Accuracy = 0.8959513902664185
    Iter #965632:  Learning rate = 0.003463:   Batch Loss = 0.201103, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5948113799095154, Accuracy = 0.895546555519104
    Iter #966144:  Learning rate = 0.003463:   Batch Loss = 0.199456, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5968075394630432, Accuracy = 0.895546555519104
    Iter #966656:  Learning rate = 0.003463:   Batch Loss = 0.198542, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6000103950500488, Accuracy = 0.8943319916725159
    Iter #967168:  Learning rate = 0.003463:   Batch Loss = 0.199566, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5946527719497681, Accuracy = 0.895546555519104
    Iter #967680:  Learning rate = 0.003463:   Batch Loss = 0.196971, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5965495109558105, Accuracy = 0.8947368264198303
    Iter #968192:  Learning rate = 0.003463:   Batch Loss = 0.195722, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5917739272117615, Accuracy = 0.8927125334739685
    Iter #968704:  Learning rate = 0.003463:   Batch Loss = 0.196482, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.59162437915802, Accuracy = 0.8959513902664185
    Iter #969216:  Learning rate = 0.003463:   Batch Loss = 0.196542, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5905148386955261, Accuracy = 0.8947368264198303
    Iter #969728:  Learning rate = 0.003463:   Batch Loss = 0.200764, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5957432985305786, Accuracy = 0.8927125334739685
    Iter #970240:  Learning rate = 0.003463:   Batch Loss = 0.197792, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.596086323261261, Accuracy = 0.8947368264198303
    Iter #970752:  Learning rate = 0.003463:   Batch Loss = 0.193915, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5943223834037781, Accuracy = 0.898785412311554
    Iter #971264:  Learning rate = 0.003463:   Batch Loss = 0.195727, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.595737874507904, Accuracy = 0.8979756832122803
    Iter #971776:  Learning rate = 0.003463:   Batch Loss = 0.198522, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5978527069091797, Accuracy = 0.8959513902664185
    Iter #972288:  Learning rate = 0.003463:   Batch Loss = 0.194824, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5972352027893066, Accuracy = 0.895546555519104
    Iter #972800:  Learning rate = 0.003463:   Batch Loss = 0.192276, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5963878035545349, Accuracy = 0.8959513902664185
    Iter #973312:  Learning rate = 0.003463:   Batch Loss = 0.190766, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5961287021636963, Accuracy = 0.8951417207717896
    Iter #973824:  Learning rate = 0.003463:   Batch Loss = 0.190788, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5949212312698364, Accuracy = 0.8971660137176514
    Iter #974336:  Learning rate = 0.003463:   Batch Loss = 0.192650, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6015617847442627, Accuracy = 0.8939270973205566
    Iter #974848:  Learning rate = 0.003463:   Batch Loss = 0.189158, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5949487686157227, Accuracy = 0.8963562846183777
    Iter #975360:  Learning rate = 0.003463:   Batch Loss = 0.188499, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5919478535652161, Accuracy = 0.8971660137176514
    Iter #975872:  Learning rate = 0.003463:   Batch Loss = 0.190008, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.591759204864502, Accuracy = 0.8967611193656921
    Iter #976384:  Learning rate = 0.003463:   Batch Loss = 0.189911, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5967423915863037, Accuracy = 0.8935222625732422
    Iter #976896:  Learning rate = 0.003463:   Batch Loss = 0.188477, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5874879360198975, Accuracy = 0.895546555519104
    Iter #977408:  Learning rate = 0.003463:   Batch Loss = 0.187773, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5862850546836853, Accuracy = 0.8951417207717896
    Iter #977920:  Learning rate = 0.003463:   Batch Loss = 0.185191, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5936164855957031, Accuracy = 0.895546555519104
    Iter #978432:  Learning rate = 0.003463:   Batch Loss = 0.186798, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.587479293346405, Accuracy = 0.8951417207717896
    Iter #978944:  Learning rate = 0.003463:   Batch Loss = 0.188928, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5848970413208008, Accuracy = 0.8943319916725159
    Iter #979456:  Learning rate = 0.003463:   Batch Loss = 0.186635, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.589043915271759, Accuracy = 0.8906882405281067
    Iter #979968:  Learning rate = 0.003463:   Batch Loss = 0.185666, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5917017459869385, Accuracy = 0.8943319916725159
    Iter #980480:  Learning rate = 0.003463:   Batch Loss = 0.182780, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5958524942398071, Accuracy = 0.8919028043746948
    Iter #980992:  Learning rate = 0.003463:   Batch Loss = 0.185344, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5918951034545898, Accuracy = 0.8910931348800659
    Iter #981504:  Learning rate = 0.003463:   Batch Loss = 0.184971, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5934094786643982, Accuracy = 0.8914979696273804
    Iter #982016:  Learning rate = 0.003463:   Batch Loss = 0.185336, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5948358774185181, Accuracy = 0.8931174278259277
    Iter #982528:  Learning rate = 0.003463:   Batch Loss = 0.183748, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5965277552604675, Accuracy = 0.8914979696273804
    Iter #983040:  Learning rate = 0.003463:   Batch Loss = 0.180639, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6002188920974731, Accuracy = 0.8919028043746948
    Iter #983552:  Learning rate = 0.003463:   Batch Loss = 0.185323, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5950222611427307, Accuracy = 0.8951417207717896
    Iter #984064:  Learning rate = 0.003463:   Batch Loss = 0.182541, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5926143527030945, Accuracy = 0.8935222625732422
    Iter #984576:  Learning rate = 0.003463:   Batch Loss = 0.181394, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5913084149360657, Accuracy = 0.892307698726654
    Iter #985088:  Learning rate = 0.003463:   Batch Loss = 0.177676, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5848321914672852, Accuracy = 0.8951417207717896
    Iter #985600:  Learning rate = 0.003463:   Batch Loss = 0.182207, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5866254568099976, Accuracy = 0.8951417207717896
    Iter #986112:  Learning rate = 0.003463:   Batch Loss = 0.182511, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5880070924758911, Accuracy = 0.895546555519104
    Iter #986624:  Learning rate = 0.003463:   Batch Loss = 0.180729, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5936221480369568, Accuracy = 0.8919028043746948
    Iter #987136:  Learning rate = 0.003463:   Batch Loss = 0.178434, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5852538347244263, Accuracy = 0.8919028043746948
    Iter #987648:  Learning rate = 0.003463:   Batch Loss = 0.181259, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.584308385848999, Accuracy = 0.892307698726654
    Iter #988160:  Learning rate = 0.003463:   Batch Loss = 0.180851, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5911018252372742, Accuracy = 0.8943319916725159
    Iter #988672:  Learning rate = 0.003463:   Batch Loss = 0.176118, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6004226207733154, Accuracy = 0.887854278087616
    Iter #989184:  Learning rate = 0.003463:   Batch Loss = 0.176100, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5882104635238647, Accuracy = 0.8931174278259277
    Iter #989696:  Learning rate = 0.003463:   Batch Loss = 0.179799, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.59334796667099, Accuracy = 0.8902834057807922
    Iter #990208:  Learning rate = 0.003463:   Batch Loss = 0.179695, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5925127267837524, Accuracy = 0.8870445489883423
    Iter #990720:  Learning rate = 0.003463:   Batch Loss = 0.176023, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5977660417556763, Accuracy = 0.8870445489883423
    Iter #991232:  Learning rate = 0.003463:   Batch Loss = 0.182812, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5973288416862488, Accuracy = 0.892307698726654
    Iter #991744:  Learning rate = 0.003463:   Batch Loss = 0.177256, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5959073901176453, Accuracy = 0.8902834057807922
    Iter #992256:  Learning rate = 0.003463:   Batch Loss = 0.179655, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5830107927322388, Accuracy = 0.8943319916725159
    Iter #992768:  Learning rate = 0.003463:   Batch Loss = 0.177601, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5911502838134766, Accuracy = 0.8910931348800659
    Iter #993280:  Learning rate = 0.003463:   Batch Loss = 0.174227, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5909423232078552, Accuracy = 0.8874493837356567
    Iter #993792:  Learning rate = 0.003463:   Batch Loss = 0.177207, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5817157030105591, Accuracy = 0.8931174278259277
    Iter #994304:  Learning rate = 0.003463:   Batch Loss = 0.176029, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5964881777763367, Accuracy = 0.8882591128349304
    Iter #994816:  Learning rate = 0.003463:   Batch Loss = 0.186808, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6133444905281067, Accuracy = 0.8801619410514832
    Iter #995328:  Learning rate = 0.003463:   Batch Loss = 0.263094, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6323260068893433, Accuracy = 0.8736842274665833
    Iter #995840:  Learning rate = 0.003463:   Batch Loss = 0.246601, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6374387741088867, Accuracy = 0.8724696636199951
    Iter #996352:  Learning rate = 0.003463:   Batch Loss = 0.245139, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7138371467590332, Accuracy = 0.8518218398094177
    Iter #996864:  Learning rate = 0.003463:   Batch Loss = 0.452195, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8320061564445496, Accuracy = 0.8117408752441406
    Iter #997376:  Learning rate = 0.003463:   Batch Loss = 0.402365, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9551974534988403, Accuracy = 0.7842105031013489
    Iter #997888:  Learning rate = 0.003463:   Batch Loss = 0.589154, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9737067222595215, Accuracy = 0.7607287168502808
    Iter #998400:  Learning rate = 0.003463:   Batch Loss = 0.631009, Accuracy = 0.859375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8930120468139648, Accuracy = 0.7995951175689697
    Iter #998912:  Learning rate = 0.003463:   Batch Loss = 0.723708, Accuracy = 0.84375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9620271325111389, Accuracy = 0.7753036618232727
    Iter #999424:  Learning rate = 0.003463:   Batch Loss = 0.605184, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8804613351821899, Accuracy = 0.7939271330833435
    Iter #999936:  Learning rate = 0.003463:   Batch Loss = 0.672894, Accuracy = 0.875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9508213996887207, Accuracy = 0.7805668115615845
    Iter #1000448:  Learning rate = 0.003324:   Batch Loss = 0.411690, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9013001322746277, Accuracy = 0.794331967830658
    Iter #1000960:  Learning rate = 0.003324:   Batch Loss = 0.400489, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7754620909690857, Accuracy = 0.8307692408561707
    Iter #1001472:  Learning rate = 0.003324:   Batch Loss = 0.445032, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8046276569366455, Accuracy = 0.8336032629013062
    Iter #1001984:  Learning rate = 0.003324:   Batch Loss = 0.486612, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8541613817214966, Accuracy = 0.8165991902351379
    Iter #1002496:  Learning rate = 0.003324:   Batch Loss = 0.565966, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9068447947502136, Accuracy = 0.807692289352417
    Iter #1003008:  Learning rate = 0.003324:   Batch Loss = 0.337095, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9370620250701904, Accuracy = 0.7983805537223816
    Iter #1003520:  Learning rate = 0.003324:   Batch Loss = 0.503325, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8769720792770386, Accuracy = 0.8105263113975525
    Iter #1004032:  Learning rate = 0.003324:   Batch Loss = 0.482573, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7920993566513062, Accuracy = 0.835627555847168
    Iter #1004544:  Learning rate = 0.003324:   Batch Loss = 0.618478, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.868579626083374, Accuracy = 0.8105263113975525
    Iter #1005056:  Learning rate = 0.003324:   Batch Loss = 0.395486, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7607624530792236, Accuracy = 0.8376518487930298
    Iter #1005568:  Learning rate = 0.003324:   Batch Loss = 0.369764, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7677182555198669, Accuracy = 0.846558690071106
    Iter #1006080:  Learning rate = 0.003324:   Batch Loss = 0.323496, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7167484760284424, Accuracy = 0.8582996129989624
    Iter #1006592:  Learning rate = 0.003324:   Batch Loss = 0.312968, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7623796463012695, Accuracy = 0.8542510271072388
    Iter #1007104:  Learning rate = 0.003324:   Batch Loss = 0.316446, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7393000721931458, Accuracy = 0.8554655909538269
    Iter #1007616:  Learning rate = 0.003324:   Batch Loss = 0.361742, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7166909575462341, Accuracy = 0.8639675974845886
    Iter #1008128:  Learning rate = 0.003324:   Batch Loss = 0.347357, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6918075084686279, Accuracy = 0.8777328133583069
    Iter #1008640:  Learning rate = 0.003324:   Batch Loss = 0.356086, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.692590594291687, Accuracy = 0.8724696636199951
    Iter #1009152:  Learning rate = 0.003324:   Batch Loss = 0.259739, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6931732892990112, Accuracy = 0.8757085204124451
    Iter #1009664:  Learning rate = 0.003324:   Batch Loss = 0.274515, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6957147121429443, Accuracy = 0.8740890622138977
    Iter #1010176:  Learning rate = 0.003324:   Batch Loss = 0.289300, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7040841579437256, Accuracy = 0.8700404763221741
    Iter #1010688:  Learning rate = 0.003324:   Batch Loss = 0.307967, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7256006002426147, Accuracy = 0.862348198890686
    Iter #1011200:  Learning rate = 0.003324:   Batch Loss = 0.286713, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7118206024169922, Accuracy = 0.862348198890686
    Iter #1011712:  Learning rate = 0.003324:   Batch Loss = 0.343732, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7295646071434021, Accuracy = 0.8611335754394531
    Iter #1012224:  Learning rate = 0.003324:   Batch Loss = 0.356969, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7936803102493286, Accuracy = 0.8412955403327942
    Iter #1012736:  Learning rate = 0.003324:   Batch Loss = 0.308505, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7748883366584778, Accuracy = 0.8412955403327942
    Iter #1013248:  Learning rate = 0.003324:   Batch Loss = 0.363499, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7094737887382507, Accuracy = 0.8631578683853149
    Iter #1013760:  Learning rate = 0.003324:   Batch Loss = 0.323976, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7466530799865723, Accuracy = 0.8497975468635559
    Iter #1014272:  Learning rate = 0.003324:   Batch Loss = 0.343246, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7027848958969116, Accuracy = 0.8643724918365479
    Iter #1014784:  Learning rate = 0.003324:   Batch Loss = 0.261114, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7108405828475952, Accuracy = 0.8704453706741333
    Iter #1015296:  Learning rate = 0.003324:   Batch Loss = 0.266087, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7585063576698303, Accuracy = 0.8514170050621033
    Iter #1015808:  Learning rate = 0.003324:   Batch Loss = 0.264301, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6856746673583984, Accuracy = 0.8761133551597595
    Iter #1016320:  Learning rate = 0.003324:   Batch Loss = 0.285765, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6867180466651917, Accuracy = 0.8757085204124451
    Iter #1016832:  Learning rate = 0.003324:   Batch Loss = 0.245662, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7023206353187561, Accuracy = 0.8680161833763123
    Iter #1017344:  Learning rate = 0.003324:   Batch Loss = 0.255360, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6870876550674438, Accuracy = 0.873279333114624
    Iter #1017856:  Learning rate = 0.003324:   Batch Loss = 0.250725, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7557277679443359, Accuracy = 0.8607287406921387
    Iter #1018368:  Learning rate = 0.003324:   Batch Loss = 0.344805, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7231870293617249, Accuracy = 0.8635627627372742
    Iter #1018880:  Learning rate = 0.003324:   Batch Loss = 0.276894, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7086958885192871, Accuracy = 0.8672064542770386
    Iter #1019392:  Learning rate = 0.003324:   Batch Loss = 0.256921, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6936423182487488, Accuracy = 0.8716599345207214
    Iter #1019904:  Learning rate = 0.003324:   Batch Loss = 0.246810, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6632388234138489, Accuracy = 0.8825910687446594
    Iter #1020416:  Learning rate = 0.003324:   Batch Loss = 0.245734, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6516464352607727, Accuracy = 0.8834007978439331
    Iter #1020928:  Learning rate = 0.003324:   Batch Loss = 0.265269, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6493158936500549, Accuracy = 0.8910931348800659
    Iter #1021440:  Learning rate = 0.003324:   Batch Loss = 0.237333, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.657297670841217, Accuracy = 0.8805667757987976
    Iter #1021952:  Learning rate = 0.003324:   Batch Loss = 0.240194, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6504018902778625, Accuracy = 0.8842105269432068
    Iter #1022464:  Learning rate = 0.003324:   Batch Loss = 0.234830, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6453900337219238, Accuracy = 0.8874493837356567
    Iter #1022976:  Learning rate = 0.003324:   Batch Loss = 0.276351, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6519275307655334, Accuracy = 0.887854278087616
    Iter #1023488:  Learning rate = 0.003324:   Batch Loss = 0.230902, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.66053706407547, Accuracy = 0.8805667757987976
    Iter #1024000:  Learning rate = 0.003324:   Batch Loss = 0.233856, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6274353265762329, Accuracy = 0.8902834057807922
    Iter #1024512:  Learning rate = 0.003324:   Batch Loss = 0.236739, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6573032140731812, Accuracy = 0.8834007978439331
    Iter #1025024:  Learning rate = 0.003324:   Batch Loss = 0.247952, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6735522747039795, Accuracy = 0.873279333114624
    Iter #1025536:  Learning rate = 0.003324:   Batch Loss = 0.228855, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6608096957206726, Accuracy = 0.8777328133583069
    Iter #1026048:  Learning rate = 0.003324:   Batch Loss = 0.222977, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6317049264907837, Accuracy = 0.8898785710334778
    Iter #1026560:  Learning rate = 0.003324:   Batch Loss = 0.233353, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6536737084388733, Accuracy = 0.8785424828529358
    Iter #1027072:  Learning rate = 0.003324:   Batch Loss = 0.243701, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6655453443527222, Accuracy = 0.8769230842590332
    Iter #1027584:  Learning rate = 0.003324:   Batch Loss = 0.242403, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6609551906585693, Accuracy = 0.8785424828529358
    Iter #1028096:  Learning rate = 0.003324:   Batch Loss = 0.232602, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6770844459533691, Accuracy = 0.8740890622138977
    Iter #1028608:  Learning rate = 0.003324:   Batch Loss = 0.261610, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6845622062683105, Accuracy = 0.873279333114624
    Iter #1029120:  Learning rate = 0.003324:   Batch Loss = 0.233837, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6869913339614868, Accuracy = 0.8724696636199951
    Iter #1029632:  Learning rate = 0.003324:   Batch Loss = 0.233491, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6519376039505005, Accuracy = 0.8838056921958923
    Iter #1030144:  Learning rate = 0.003324:   Batch Loss = 0.250744, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6593828201293945, Accuracy = 0.8850202560424805
    Iter #1030656:  Learning rate = 0.003324:   Batch Loss = 0.256205, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6424462795257568, Accuracy = 0.8886639475822449
    Iter #1031168:  Learning rate = 0.003324:   Batch Loss = 0.236378, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7039400339126587, Accuracy = 0.8611335754394531
    Iter #1031680:  Learning rate = 0.003324:   Batch Loss = 0.254594, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6379690170288086, Accuracy = 0.8882591128349304
    Iter #1032192:  Learning rate = 0.003324:   Batch Loss = 0.241960, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6405669450759888, Accuracy = 0.882186233997345
    Iter #1032704:  Learning rate = 0.003324:   Batch Loss = 0.238219, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6693601012229919, Accuracy = 0.8757085204124451
    Iter #1033216:  Learning rate = 0.003324:   Batch Loss = 0.242233, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6808487772941589, Accuracy = 0.8700404763221741
    Iter #1033728:  Learning rate = 0.003324:   Batch Loss = 0.222885, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6724808812141418, Accuracy = 0.878947377204895
    Iter #1034240:  Learning rate = 0.003324:   Batch Loss = 0.218891, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6483924388885498, Accuracy = 0.8834007978439331
    Iter #1034752:  Learning rate = 0.003324:   Batch Loss = 0.248842, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6484827399253845, Accuracy = 0.8801619410514832
    Iter #1035264:  Learning rate = 0.003324:   Batch Loss = 0.218978, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6443290114402771, Accuracy = 0.8886639475822449
    Iter #1035776:  Learning rate = 0.003324:   Batch Loss = 0.239969, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6558611392974854, Accuracy = 0.8809716701507568
    Iter #1036288:  Learning rate = 0.003324:   Batch Loss = 0.224511, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6255756616592407, Accuracy = 0.8886639475822449
    Iter #1036800:  Learning rate = 0.003324:   Batch Loss = 0.220460, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6112902760505676, Accuracy = 0.8959513902664185
    Iter #1037312:  Learning rate = 0.003324:   Batch Loss = 0.213531, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.610539436340332, Accuracy = 0.8947368264198303
    Iter #1037824:  Learning rate = 0.003324:   Batch Loss = 0.213961, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6076326370239258, Accuracy = 0.895546555519104
    Iter #1038336:  Learning rate = 0.003324:   Batch Loss = 0.214750, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6030257940292358, Accuracy = 0.8979756832122803
    Iter #1038848:  Learning rate = 0.003324:   Batch Loss = 0.211091, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5957555770874023, Accuracy = 0.8975708484649658
    Iter #1039360:  Learning rate = 0.003324:   Batch Loss = 0.204848, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5927230715751648, Accuracy = 0.8951417207717896
    Iter #1039872:  Learning rate = 0.003324:   Batch Loss = 0.203451, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5917257070541382, Accuracy = 0.8947368264198303
    Iter #1040384:  Learning rate = 0.003324:   Batch Loss = 0.207560, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5873192548751831, Accuracy = 0.8947368264198303
    Iter #1040896:  Learning rate = 0.003324:   Batch Loss = 0.202893, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.583733081817627, Accuracy = 0.8975708484649658
    Iter #1041408:  Learning rate = 0.003324:   Batch Loss = 0.209948, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5793032646179199, Accuracy = 0.9004048705101013
    Iter #1041920:  Learning rate = 0.003324:   Batch Loss = 0.201866, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5824669599533081, Accuracy = 0.9020242691040039
    Iter #1042432:  Learning rate = 0.003324:   Batch Loss = 0.201456, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5824563503265381, Accuracy = 0.901214599609375
    Iter #1042944:  Learning rate = 0.003324:   Batch Loss = 0.205599, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5803536176681519, Accuracy = 0.8979756832122803
    Iter #1043456:  Learning rate = 0.003324:   Batch Loss = 0.199772, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5807090401649475, Accuracy = 0.8991903066635132
    Iter #1043968:  Learning rate = 0.003324:   Batch Loss = 0.200181, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.579685628414154, Accuracy = 0.8983805775642395
    Iter #1044480:  Learning rate = 0.003324:   Batch Loss = 0.197259, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5788241624832153, Accuracy = 0.8991903066635132
    Iter #1044992:  Learning rate = 0.003324:   Batch Loss = 0.195792, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5765384435653687, Accuracy = 0.8991903066635132
    Iter #1045504:  Learning rate = 0.003324:   Batch Loss = 0.195225, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.575086236000061, Accuracy = 0.898785412311554
    Iter #1046016:  Learning rate = 0.003324:   Batch Loss = 0.200559, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5749277472496033, Accuracy = 0.8983805775642395
    Iter #1046528:  Learning rate = 0.003324:   Batch Loss = 0.197148, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.574911892414093, Accuracy = 0.8983805775642395
    Iter #1047040:  Learning rate = 0.003324:   Batch Loss = 0.196254, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5742993354797363, Accuracy = 0.8991903066635132
    Iter #1047552:  Learning rate = 0.003324:   Batch Loss = 0.196118, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5750219821929932, Accuracy = 0.8991903066635132
    Iter #1048064:  Learning rate = 0.003324:   Batch Loss = 0.191608, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.578053891658783, Accuracy = 0.8975708484649658
    Iter #1048576:  Learning rate = 0.003324:   Batch Loss = 0.190504, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5789991021156311, Accuracy = 0.8963562846183777
    Iter #1049088:  Learning rate = 0.003324:   Batch Loss = 0.193438, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5777994394302368, Accuracy = 0.8959513902664185
    Iter #1049600:  Learning rate = 0.003324:   Batch Loss = 0.190490, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5773281455039978, Accuracy = 0.8979756832122803
    Iter #1050112:  Learning rate = 0.003324:   Batch Loss = 0.191385, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5730369091033936, Accuracy = 0.8979756832122803
    Iter #1050624:  Learning rate = 0.003324:   Batch Loss = 0.191061, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5759650468826294, Accuracy = 0.892307698726654
    Iter #1051136:  Learning rate = 0.003324:   Batch Loss = 0.191883, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5814906358718872, Accuracy = 0.8951417207717896
    Iter #1051648:  Learning rate = 0.003324:   Batch Loss = 0.188434, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5823493003845215, Accuracy = 0.8947368264198303
    Iter #1052160:  Learning rate = 0.003324:   Batch Loss = 0.186808, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5754410028457642, Accuracy = 0.895546555519104
    Iter #1052672:  Learning rate = 0.003324:   Batch Loss = 0.189299, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5741701126098633, Accuracy = 0.8927125334739685
    Iter #1053184:  Learning rate = 0.003324:   Batch Loss = 0.190768, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5752207636833191, Accuracy = 0.8931174278259277
    Iter #1053696:  Learning rate = 0.003324:   Batch Loss = 0.190960, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5776726007461548, Accuracy = 0.8959513902664185
    Iter #1054208:  Learning rate = 0.003324:   Batch Loss = 0.189896, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5831186771392822, Accuracy = 0.8931174278259277
    Iter #1054720:  Learning rate = 0.003324:   Batch Loss = 0.186930, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5797930359840393, Accuracy = 0.8975708484649658
    Iter #1055232:  Learning rate = 0.003324:   Batch Loss = 0.185055, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5762145519256592, Accuracy = 0.8951417207717896
    Iter #1055744:  Learning rate = 0.003324:   Batch Loss = 0.185930, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5781813859939575, Accuracy = 0.8963562846183777
    Iter #1056256:  Learning rate = 0.003324:   Batch Loss = 0.182149, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5810807347297668, Accuracy = 0.8943319916725159
    Iter #1056768:  Learning rate = 0.003324:   Batch Loss = 0.184054, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5799070596694946, Accuracy = 0.8959513902664185
    Iter #1057280:  Learning rate = 0.003324:   Batch Loss = 0.184768, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5794728398323059, Accuracy = 0.8939270973205566
    Iter #1057792:  Learning rate = 0.003324:   Batch Loss = 0.183245, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5773154497146606, Accuracy = 0.8971660137176514
    Iter #1058304:  Learning rate = 0.003324:   Batch Loss = 0.180216, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5715259313583374, Accuracy = 0.8943319916725159
    Iter #1058816:  Learning rate = 0.003324:   Batch Loss = 0.183204, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5785958766937256, Accuracy = 0.8927125334739685
    Iter #1059328:  Learning rate = 0.003324:   Batch Loss = 0.180319, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5956913828849792, Accuracy = 0.8906882405281067
    Iter #1059840:  Learning rate = 0.003324:   Batch Loss = 0.185189, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5819995403289795, Accuracy = 0.8947368264198303
    Iter #1060352:  Learning rate = 0.003324:   Batch Loss = 0.182161, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5784958600997925, Accuracy = 0.8914979696273804
    Iter #1060864:  Learning rate = 0.003324:   Batch Loss = 0.178138, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.583091139793396, Accuracy = 0.8919028043746948
    Iter #1061376:  Learning rate = 0.003324:   Batch Loss = 0.178588, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5855703353881836, Accuracy = 0.8890688419342041
    Iter #1061888:  Learning rate = 0.003324:   Batch Loss = 0.180673, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5857965350151062, Accuracy = 0.887854278087616
    Iter #1062400:  Learning rate = 0.003324:   Batch Loss = 0.182569, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5822044014930725, Accuracy = 0.8882591128349304
    Iter #1062912:  Learning rate = 0.003324:   Batch Loss = 0.180935, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5803597569465637, Accuracy = 0.8894736766815186
    Iter #1063424:  Learning rate = 0.003324:   Batch Loss = 0.180735, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5842430591583252, Accuracy = 0.8910931348800659
    Iter #1063936:  Learning rate = 0.003324:   Batch Loss = 0.176571, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5827697515487671, Accuracy = 0.8902834057807922
    Iter #1064448:  Learning rate = 0.003324:   Batch Loss = 0.179359, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5853698253631592, Accuracy = 0.8850202560424805
    Iter #1064960:  Learning rate = 0.003324:   Batch Loss = 0.176113, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5937587022781372, Accuracy = 0.8890688419342041
    Iter #1065472:  Learning rate = 0.003324:   Batch Loss = 0.182255, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.58340984582901, Accuracy = 0.8886639475822449
    Iter #1065984:  Learning rate = 0.003324:   Batch Loss = 0.175438, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5851050615310669, Accuracy = 0.8866396546363831
    Iter #1066496:  Learning rate = 0.003324:   Batch Loss = 0.177073, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5833456516265869, Accuracy = 0.8894736766815186
    Iter #1067008:  Learning rate = 0.003324:   Batch Loss = 0.184550, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5851192474365234, Accuracy = 0.8902834057807922
    Iter #1067520:  Learning rate = 0.003324:   Batch Loss = 0.176525, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5935956239700317, Accuracy = 0.8825910687446594
    Iter #1068032:  Learning rate = 0.003324:   Batch Loss = 0.192202, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6348237991333008, Accuracy = 0.8761133551597595
    Iter #1068544:  Learning rate = 0.003324:   Batch Loss = 0.206939, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7638749480247498, Accuracy = 0.8457489609718323
    Iter #1069056:  Learning rate = 0.003324:   Batch Loss = 0.367712, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7384607195854187, Accuracy = 0.8368421196937561
    Iter #1069568:  Learning rate = 0.003324:   Batch Loss = 0.432558, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7095300555229187, Accuracy = 0.8481781482696533
    Iter #1070080:  Learning rate = 0.003324:   Batch Loss = 0.405464, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7577131986618042, Accuracy = 0.8348178267478943
    Iter #1070592:  Learning rate = 0.003324:   Batch Loss = 0.369901, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7416991591453552, Accuracy = 0.8372469544410706
    Iter #1071104:  Learning rate = 0.003324:   Batch Loss = 0.417026, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.814801037311554, Accuracy = 0.8117408752441406
    Iter #1071616:  Learning rate = 0.003324:   Batch Loss = 0.984296, Accuracy = 0.734375
    PERFORMANCE ON TEST SET:             Batch Loss = 1.0640228986740112, Accuracy = 0.7408906817436218
    Iter #1072128:  Learning rate = 0.003324:   Batch Loss = 0.618670, Accuracy = 0.859375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9277070760726929, Accuracy = 0.7894737124443054
    Iter #1072640:  Learning rate = 0.003324:   Batch Loss = 0.467163, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.86403489112854, Accuracy = 0.791093111038208
    Iter #1073152:  Learning rate = 0.003324:   Batch Loss = 0.603000, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8282458782196045, Accuracy = 0.8149797320365906
    Iter #1073664:  Learning rate = 0.003324:   Batch Loss = 0.458395, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8123763203620911, Accuracy = 0.8303643465042114
    Iter #1074176:  Learning rate = 0.003324:   Batch Loss = 0.450532, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8378350138664246, Accuracy = 0.8145748972892761
    Iter #1074688:  Learning rate = 0.003324:   Batch Loss = 0.433287, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8403706550598145, Accuracy = 0.8186234831809998
    Iter #1075200:  Learning rate = 0.003324:   Batch Loss = 0.448896, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8490027189254761, Accuracy = 0.8222672343254089
    Iter #1075712:  Learning rate = 0.003324:   Batch Loss = 0.378824, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.779607355594635, Accuracy = 0.8352226614952087
    Iter #1076224:  Learning rate = 0.003324:   Batch Loss = 0.459415, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8070817589759827, Accuracy = 0.8259109258651733
    Iter #1076736:  Learning rate = 0.003324:   Batch Loss = 0.309414, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7308281660079956, Accuracy = 0.8550607562065125
    Iter #1077248:  Learning rate = 0.003324:   Batch Loss = 0.328724, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7073428630828857, Accuracy = 0.8740890622138977
    Iter #1077760:  Learning rate = 0.003324:   Batch Loss = 0.387351, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7738416790962219, Accuracy = 0.8352226614952087
    Iter #1078272:  Learning rate = 0.003324:   Batch Loss = 0.396743, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8204553127288818, Accuracy = 0.8331983685493469
    Iter #1078784:  Learning rate = 0.003324:   Batch Loss = 0.543133, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8213078379631042, Accuracy = 0.8230769038200378
    Iter #1079296:  Learning rate = 0.003324:   Batch Loss = 0.474715, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7284792065620422, Accuracy = 0.8574898838996887
    Iter #1079808:  Learning rate = 0.003324:   Batch Loss = 0.360971, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7773262858390808, Accuracy = 0.8417003750801086
    Iter #1080320:  Learning rate = 0.003324:   Batch Loss = 0.374297, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7167336940765381, Accuracy = 0.859919011592865
    Iter #1080832:  Learning rate = 0.003324:   Batch Loss = 0.303016, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7562814950942993, Accuracy = 0.8526315689086914
    Iter #1081344:  Learning rate = 0.003324:   Batch Loss = 0.268287, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7560257911682129, Accuracy = 0.8506072759628296
    Iter #1081856:  Learning rate = 0.003324:   Batch Loss = 0.304640, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7252814173698425, Accuracy = 0.8651821613311768
    Iter #1082368:  Learning rate = 0.003324:   Batch Loss = 0.293882, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7040146589279175, Accuracy = 0.8659918904304504
    Iter #1082880:  Learning rate = 0.003324:   Batch Loss = 0.274607, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6744155883789062, Accuracy = 0.8777328133583069
    Iter #1083392:  Learning rate = 0.003324:   Batch Loss = 0.357282, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7125351428985596, Accuracy = 0.8659918904304504
    Iter #1083904:  Learning rate = 0.003324:   Batch Loss = 0.266141, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.663941502571106, Accuracy = 0.8874493837356567
    Iter #1084416:  Learning rate = 0.003324:   Batch Loss = 0.245732, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7163314819335938, Accuracy = 0.8744939565658569
    Iter #1084928:  Learning rate = 0.003324:   Batch Loss = 0.255403, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6913584470748901, Accuracy = 0.8700404763221741
    Iter #1085440:  Learning rate = 0.003324:   Batch Loss = 0.280710, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6762005090713501, Accuracy = 0.8757085204124451
    Iter #1085952:  Learning rate = 0.003324:   Batch Loss = 0.255810, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6658445000648499, Accuracy = 0.8797571063041687
    Iter #1086464:  Learning rate = 0.003324:   Batch Loss = 0.259493, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6565895676612854, Accuracy = 0.882186233997345
    Iter #1086976:  Learning rate = 0.003324:   Batch Loss = 0.236184, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6595369577407837, Accuracy = 0.8801619410514832
    Iter #1087488:  Learning rate = 0.003324:   Batch Loss = 0.238842, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6360204815864563, Accuracy = 0.8898785710334778
    Iter #1088000:  Learning rate = 0.003324:   Batch Loss = 0.243382, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6342955231666565, Accuracy = 0.8825910687446594
    Iter #1088512:  Learning rate = 0.003324:   Batch Loss = 0.229226, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6126985549926758, Accuracy = 0.8902834057807922
    Iter #1089024:  Learning rate = 0.003324:   Batch Loss = 0.223425, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5980305075645447, Accuracy = 0.8995951414108276
    Iter #1089536:  Learning rate = 0.003324:   Batch Loss = 0.226441, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5850005149841309, Accuracy = 0.9016194343566895
    Iter #1090048:  Learning rate = 0.003324:   Batch Loss = 0.224663, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5872714519500732, Accuracy = 0.8967611193656921
    Iter #1090560:  Learning rate = 0.003324:   Batch Loss = 0.221818, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6127574443817139, Accuracy = 0.8951417207717896
    Iter #1091072:  Learning rate = 0.003324:   Batch Loss = 0.218090, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6030387878417969, Accuracy = 0.895546555519104
    Iter #1091584:  Learning rate = 0.003324:   Batch Loss = 0.221751, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5844681859016418, Accuracy = 0.8963562846183777
    Iter #1092096:  Learning rate = 0.003324:   Batch Loss = 0.215556, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5888038873672485, Accuracy = 0.8967611193656921
    Iter #1092608:  Learning rate = 0.003324:   Batch Loss = 0.213181, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5806311368942261, Accuracy = 0.8971660137176514
    Iter #1093120:  Learning rate = 0.003324:   Batch Loss = 0.215334, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5768797397613525, Accuracy = 0.8959513902664185
    Iter #1093632:  Learning rate = 0.003324:   Batch Loss = 0.213506, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.58396977186203, Accuracy = 0.8999999761581421
    Iter #1094144:  Learning rate = 0.003324:   Batch Loss = 0.210669, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5855910778045654, Accuracy = 0.8975708484649658
    Iter #1094656:  Learning rate = 0.003324:   Batch Loss = 0.210542, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5772133469581604, Accuracy = 0.8975708484649658
    Iter #1095168:  Learning rate = 0.003324:   Batch Loss = 0.209159, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5758677124977112, Accuracy = 0.8967611193656921
    Iter #1095680:  Learning rate = 0.003324:   Batch Loss = 0.207029, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5769716501235962, Accuracy = 0.895546555519104
    Iter #1096192:  Learning rate = 0.003324:   Batch Loss = 0.203225, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5725913047790527, Accuracy = 0.8959513902664185
    Iter #1096704:  Learning rate = 0.003324:   Batch Loss = 0.207739, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5671176910400391, Accuracy = 0.8963562846183777
    Iter #1097216:  Learning rate = 0.003324:   Batch Loss = 0.203416, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5672794580459595, Accuracy = 0.8999999761581421
    Iter #1097728:  Learning rate = 0.003324:   Batch Loss = 0.200107, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5704184770584106, Accuracy = 0.8959513902664185
    Iter #1098240:  Learning rate = 0.003324:   Batch Loss = 0.201060, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5687713027000427, Accuracy = 0.8995951414108276
    Iter #1098752:  Learning rate = 0.003324:   Batch Loss = 0.199515, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5638338327407837, Accuracy = 0.8975708484649658
    Iter #1099264:  Learning rate = 0.003324:   Batch Loss = 0.205733, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5605032444000244, Accuracy = 0.8939270973205566
    Iter #1099776:  Learning rate = 0.003324:   Batch Loss = 0.197897, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5575101971626282, Accuracy = 0.8983805775642395
    Iter #1100288:  Learning rate = 0.003191:   Batch Loss = 0.202618, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5573062300682068, Accuracy = 0.8979756832122803
    Iter #1100800:  Learning rate = 0.003191:   Batch Loss = 0.196523, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5581076741218567, Accuracy = 0.8991903066635132
    Iter #1101312:  Learning rate = 0.003191:   Batch Loss = 0.199292, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5620168447494507, Accuracy = 0.8983805775642395
    Iter #1101824:  Learning rate = 0.003191:   Batch Loss = 0.200217, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5660577416419983, Accuracy = 0.8995951414108276
    Iter #1102336:  Learning rate = 0.003191:   Batch Loss = 0.195363, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.563007652759552, Accuracy = 0.8975708484649658
    Iter #1102848:  Learning rate = 0.003191:   Batch Loss = 0.195847, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5635181665420532, Accuracy = 0.8967611193656921
    Iter #1103360:  Learning rate = 0.003191:   Batch Loss = 0.194966, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5667145252227783, Accuracy = 0.8979756832122803
    Iter #1103872:  Learning rate = 0.003191:   Batch Loss = 0.193613, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5683455467224121, Accuracy = 0.8967611193656921
    Iter #1104384:  Learning rate = 0.003191:   Batch Loss = 0.192002, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5669962167739868, Accuracy = 0.8979756832122803
    Iter #1104896:  Learning rate = 0.003191:   Batch Loss = 0.195837, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5610085129737854, Accuracy = 0.8999999761581421
    Iter #1105408:  Learning rate = 0.003191:   Batch Loss = 0.193136, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5589148998260498, Accuracy = 0.9008097052574158
    Iter #1105920:  Learning rate = 0.003191:   Batch Loss = 0.190192, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5574964284896851, Accuracy = 0.8991903066635132
    Iter #1106432:  Learning rate = 0.003191:   Batch Loss = 0.190512, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5566261410713196, Accuracy = 0.8975708484649658
    Iter #1106944:  Learning rate = 0.003191:   Batch Loss = 0.189482, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5579663515090942, Accuracy = 0.8983805775642395
    Iter #1107456:  Learning rate = 0.003191:   Batch Loss = 0.188410, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5552600622177124, Accuracy = 0.8983805775642395
    Iter #1107968:  Learning rate = 0.003191:   Batch Loss = 0.192502, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5540984272956848, Accuracy = 0.8999999761581421
    Iter #1108480:  Learning rate = 0.003191:   Batch Loss = 0.191316, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5555272102355957, Accuracy = 0.8995951414108276
    Iter #1108992:  Learning rate = 0.003191:   Batch Loss = 0.189036, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5561386346817017, Accuracy = 0.8991903066635132
    Iter #1109504:  Learning rate = 0.003191:   Batch Loss = 0.186483, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5590195655822754, Accuracy = 0.8971660137176514
    Iter #1110016:  Learning rate = 0.003191:   Batch Loss = 0.187217, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5598076581954956, Accuracy = 0.9008097052574158
    Iter #1110528:  Learning rate = 0.003191:   Batch Loss = 0.186781, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5551488399505615, Accuracy = 0.9008097052574158
    Iter #1111040:  Learning rate = 0.003191:   Batch Loss = 0.187799, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5547313690185547, Accuracy = 0.9016194343566895
    Iter #1111552:  Learning rate = 0.003191:   Batch Loss = 0.183397, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5588443279266357, Accuracy = 0.8983805775642395
    Iter #1112064:  Learning rate = 0.003191:   Batch Loss = 0.185767, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5565976500511169, Accuracy = 0.8995951414108276
    Iter #1112576:  Learning rate = 0.003191:   Batch Loss = 0.185902, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5645278096199036, Accuracy = 0.8975708484649658
    Iter #1113088:  Learning rate = 0.003191:   Batch Loss = 0.184112, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5667325854301453, Accuracy = 0.8967611193656921
    Iter #1113600:  Learning rate = 0.003191:   Batch Loss = 0.182271, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5592811107635498, Accuracy = 0.8967611193656921
    Iter #1114112:  Learning rate = 0.003191:   Batch Loss = 0.184150, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5578169226646423, Accuracy = 0.8967611193656921
    Iter #1114624:  Learning rate = 0.003191:   Batch Loss = 0.181509, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5611228346824646, Accuracy = 0.8951417207717896
    Iter #1115136:  Learning rate = 0.003191:   Batch Loss = 0.185393, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.550615668296814, Accuracy = 0.8991903066635132
    Iter #1115648:  Learning rate = 0.003191:   Batch Loss = 0.178513, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5507128238677979, Accuracy = 0.8991903066635132
    Iter #1116160:  Learning rate = 0.003191:   Batch Loss = 0.180334, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5525314807891846, Accuracy = 0.8999999761581421
    Iter #1116672:  Learning rate = 0.003191:   Batch Loss = 0.186872, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5505901575088501, Accuracy = 0.8999999761581421
    Iter #1117184:  Learning rate = 0.003191:   Batch Loss = 0.180803, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5449625849723816, Accuracy = 0.898785412311554
    Iter #1117696:  Learning rate = 0.003191:   Batch Loss = 0.176059, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5543656945228577, Accuracy = 0.8967611193656921
    Iter #1118208:  Learning rate = 0.003191:   Batch Loss = 0.177523, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5591007471084595, Accuracy = 0.8983805775642395
    Iter #1118720:  Learning rate = 0.003191:   Batch Loss = 0.183240, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5531511306762695, Accuracy = 0.8991903066635132
    Iter #1119232:  Learning rate = 0.003191:   Batch Loss = 0.177723, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5546647906303406, Accuracy = 0.9016194343566895
    Iter #1119744:  Learning rate = 0.003191:   Batch Loss = 0.174606, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5550739765167236, Accuracy = 0.8959513902664185
    Iter #1120256:  Learning rate = 0.003191:   Batch Loss = 0.177752, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5535529255867004, Accuracy = 0.8971660137176514
    Iter #1120768:  Learning rate = 0.003191:   Batch Loss = 0.173979, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5527592897415161, Accuracy = 0.8999999761581421
    Iter #1121280:  Learning rate = 0.003191:   Batch Loss = 0.172125, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5502585768699646, Accuracy = 0.8983805775642395
    Iter #1121792:  Learning rate = 0.003191:   Batch Loss = 0.173547, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5460106134414673, Accuracy = 0.8979756832122803
    Iter #1122304:  Learning rate = 0.003191:   Batch Loss = 0.170754, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5457521080970764, Accuracy = 0.8963562846183777
    Iter #1122816:  Learning rate = 0.003191:   Batch Loss = 0.175505, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5517650842666626, Accuracy = 0.8939270973205566
    Iter #1123328:  Learning rate = 0.003191:   Batch Loss = 0.171974, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5498400330543518, Accuracy = 0.8951417207717896
    Iter #1123840:  Learning rate = 0.003191:   Batch Loss = 0.173196, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5476537346839905, Accuracy = 0.8975708484649658
    Iter #1124352:  Learning rate = 0.003191:   Batch Loss = 0.169908, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5491265654563904, Accuracy = 0.8971660137176514
    Iter #1124864:  Learning rate = 0.003191:   Batch Loss = 0.169993, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.559229850769043, Accuracy = 0.8951417207717896
    Iter #1125376:  Learning rate = 0.003191:   Batch Loss = 0.172591, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.552312970161438, Accuracy = 0.8959513902664185
    Iter #1125888:  Learning rate = 0.003191:   Batch Loss = 0.172113, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5521720051765442, Accuracy = 0.8963562846183777
    Iter #1126400:  Learning rate = 0.003191:   Batch Loss = 0.173352, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5498344898223877, Accuracy = 0.8963562846183777
    Iter #1126912:  Learning rate = 0.003191:   Batch Loss = 0.172615, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5627114176750183, Accuracy = 0.8906882405281067
    Iter #1127424:  Learning rate = 0.003191:   Batch Loss = 0.174085, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5566259026527405, Accuracy = 0.8959513902664185
    Iter #1127936:  Learning rate = 0.003191:   Batch Loss = 0.175262, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5528472661972046, Accuracy = 0.8947368264198303
    Iter #1128448:  Learning rate = 0.003191:   Batch Loss = 0.169888, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5725796222686768, Accuracy = 0.8894736766815186
    Iter #1128960:  Learning rate = 0.003191:   Batch Loss = 0.169051, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5617715120315552, Accuracy = 0.8951417207717896
    Iter #1129472:  Learning rate = 0.003191:   Batch Loss = 0.169127, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5563730597496033, Accuracy = 0.8975708484649658
    Iter #1129984:  Learning rate = 0.003191:   Batch Loss = 0.171169, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5577590465545654, Accuracy = 0.892307698726654
    Iter #1130496:  Learning rate = 0.003191:   Batch Loss = 0.170023, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5588881969451904, Accuracy = 0.8943319916725159
    Iter #1131008:  Learning rate = 0.003191:   Batch Loss = 0.168650, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5607032775878906, Accuracy = 0.8939270973205566
    Iter #1131520:  Learning rate = 0.003191:   Batch Loss = 0.173960, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.555014431476593, Accuracy = 0.8947368264198303
    Iter #1132032:  Learning rate = 0.003191:   Batch Loss = 0.165648, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5530417561531067, Accuracy = 0.8979756832122803
    Iter #1132544:  Learning rate = 0.003191:   Batch Loss = 0.172726, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5656700730323792, Accuracy = 0.8947368264198303
    Iter #1133056:  Learning rate = 0.003191:   Batch Loss = 0.165938, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5636488199234009, Accuracy = 0.8914979696273804
    Iter #1133568:  Learning rate = 0.003191:   Batch Loss = 0.170148, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5594524145126343, Accuracy = 0.8939270973205566
    Iter #1134080:  Learning rate = 0.003191:   Batch Loss = 0.168792, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.558095395565033, Accuracy = 0.8967611193656921
    Iter #1134592:  Learning rate = 0.003191:   Batch Loss = 0.169694, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5613842606544495, Accuracy = 0.8959513902664185
    Iter #1135104:  Learning rate = 0.003191:   Batch Loss = 0.169741, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5571252703666687, Accuracy = 0.8919028043746948
    Iter #1135616:  Learning rate = 0.003191:   Batch Loss = 0.163645, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5572435855865479, Accuracy = 0.8931174278259277
    Iter #1136128:  Learning rate = 0.003191:   Batch Loss = 0.165228, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5648931264877319, Accuracy = 0.8919028043746948
    Iter #1136640:  Learning rate = 0.003191:   Batch Loss = 0.163046, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.558998703956604, Accuracy = 0.895546555519104
    Iter #1137152:  Learning rate = 0.003191:   Batch Loss = 0.175156, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5484445095062256, Accuracy = 0.895546555519104
    Iter #1137664:  Learning rate = 0.003191:   Batch Loss = 0.164961, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5568404793739319, Accuracy = 0.8931174278259277
    Iter #1138176:  Learning rate = 0.003191:   Batch Loss = 0.167490, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5641162991523743, Accuracy = 0.892307698726654
    Iter #1138688:  Learning rate = 0.003191:   Batch Loss = 0.164636, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5590764284133911, Accuracy = 0.892307698726654
    Iter #1139200:  Learning rate = 0.003191:   Batch Loss = 0.164789, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5513560771942139, Accuracy = 0.8967611193656921
    Iter #1139712:  Learning rate = 0.003191:   Batch Loss = 0.166408, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5591534972190857, Accuracy = 0.8939270973205566
    Iter #1140224:  Learning rate = 0.003191:   Batch Loss = 0.161786, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5615556240081787, Accuracy = 0.8894736766815186
    Iter #1140736:  Learning rate = 0.003191:   Batch Loss = 0.162427, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5521926283836365, Accuracy = 0.8963562846183777
    Iter #1141248:  Learning rate = 0.003191:   Batch Loss = 0.165896, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5583313703536987, Accuracy = 0.8935222625732422
    Iter #1141760:  Learning rate = 0.003191:   Batch Loss = 0.161556, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5580293536186218, Accuracy = 0.892307698726654
    Iter #1142272:  Learning rate = 0.003191:   Batch Loss = 0.162077, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5617212057113647, Accuracy = 0.8886639475822449
    Iter #1142784:  Learning rate = 0.003191:   Batch Loss = 0.171061, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5665689706802368, Accuracy = 0.8898785710334778
    Iter #1143296:  Learning rate = 0.003191:   Batch Loss = 0.159133, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5645806193351746, Accuracy = 0.8935222625732422
    Iter #1143808:  Learning rate = 0.003191:   Batch Loss = 0.161472, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5706721544265747, Accuracy = 0.8902834057807922
    Iter #1144320:  Learning rate = 0.003191:   Batch Loss = 0.164093, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5746601223945618, Accuracy = 0.8898785710334778
    Iter #1144832:  Learning rate = 0.003191:   Batch Loss = 0.157186, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5643782615661621, Accuracy = 0.8947368264198303
    Iter #1145344:  Learning rate = 0.003191:   Batch Loss = 0.158034, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5544803142547607, Accuracy = 0.8898785710334778
    Iter #1145856:  Learning rate = 0.003191:   Batch Loss = 0.164240, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5659448504447937, Accuracy = 0.8886639475822449
    Iter #1146368:  Learning rate = 0.003191:   Batch Loss = 0.162395, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5605817437171936, Accuracy = 0.8914979696273804
    Iter #1146880:  Learning rate = 0.003191:   Batch Loss = 0.161319, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5579121112823486, Accuracy = 0.8931174278259277
    Iter #1147392:  Learning rate = 0.003191:   Batch Loss = 0.160850, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5719111561775208, Accuracy = 0.8874493837356567
    Iter #1147904:  Learning rate = 0.003191:   Batch Loss = 0.157180, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5717147588729858, Accuracy = 0.8870445489883423
    Iter #1148416:  Learning rate = 0.003191:   Batch Loss = 0.156156, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5597033500671387, Accuracy = 0.8919028043746948
    Iter #1148928:  Learning rate = 0.003191:   Batch Loss = 0.160127, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5524190664291382, Accuracy = 0.8914979696273804
    Iter #1149440:  Learning rate = 0.003191:   Batch Loss = 0.158025, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5634433031082153, Accuracy = 0.8906882405281067
    Iter #1149952:  Learning rate = 0.003191:   Batch Loss = 0.155873, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5615750551223755, Accuracy = 0.8935222625732422
    Iter #1150464:  Learning rate = 0.003191:   Batch Loss = 0.158491, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5625659227371216, Accuracy = 0.8902834057807922
    Iter #1150976:  Learning rate = 0.003191:   Batch Loss = 0.155243, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5792865753173828, Accuracy = 0.8829959630966187
    Iter #1151488:  Learning rate = 0.003191:   Batch Loss = 0.159075, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5639880895614624, Accuracy = 0.8874493837356567
    Iter #1152000:  Learning rate = 0.003191:   Batch Loss = 0.161084, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5745348334312439, Accuracy = 0.8874493837356567
    Iter #1152512:  Learning rate = 0.003191:   Batch Loss = 0.218802, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5672351121902466, Accuracy = 0.8825910687446594
    Iter #1153024:  Learning rate = 0.003191:   Batch Loss = 0.175830, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6178169250488281, Accuracy = 0.8773279190063477
    Iter #1153536:  Learning rate = 0.003191:   Batch Loss = 0.173190, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6289175152778625, Accuracy = 0.8724696636199951
    Iter #1154048:  Learning rate = 0.003191:   Batch Loss = 0.289472, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9005162715911865, Accuracy = 0.7927125692367554
    Iter #1154560:  Learning rate = 0.003191:   Batch Loss = 0.496917, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9726343154907227, Accuracy = 0.7526316046714783
    Iter #1155072:  Learning rate = 0.003191:   Batch Loss = 0.523484, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9295875430107117, Accuracy = 0.7615384459495544
    Iter #1155584:  Learning rate = 0.003191:   Batch Loss = 0.621567, Accuracy = 0.875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9709699749946594, Accuracy = 0.7631579041481018
    Iter #1156096:  Learning rate = 0.003191:   Batch Loss = 0.508722, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9020000696182251, Accuracy = 0.7813765406608582
    Iter #1156608:  Learning rate = 0.003191:   Batch Loss = 0.591674, Accuracy = 0.875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8509020805358887, Accuracy = 0.7914980053901672
    Iter #1157120:  Learning rate = 0.003191:   Batch Loss = 0.391320, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8969022631645203, Accuracy = 0.7894737124443054
    Iter #1157632:  Learning rate = 0.003191:   Batch Loss = 0.631814, Accuracy = 0.84375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8312718272209167, Accuracy = 0.8060728907585144
    Iter #1158144:  Learning rate = 0.003191:   Batch Loss = 0.494530, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8824123740196228, Accuracy = 0.8012145757675171
    Iter #1158656:  Learning rate = 0.003191:   Batch Loss = 0.642574, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8225223422050476, Accuracy = 0.8190283179283142
    Iter #1159168:  Learning rate = 0.003191:   Batch Loss = 0.366994, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7434971332550049, Accuracy = 0.840080976486206
    Iter #1159680:  Learning rate = 0.003191:   Batch Loss = 0.388630, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7994621396064758, Accuracy = 0.8275303840637207
    Iter #1160192:  Learning rate = 0.003191:   Batch Loss = 0.352528, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8388599157333374, Accuracy = 0.8157894611358643
    Iter #1160704:  Learning rate = 0.003191:   Batch Loss = 0.674140, Accuracy = 0.859375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9977692365646362, Accuracy = 0.7659919261932373
    Iter #1161216:  Learning rate = 0.003191:   Batch Loss = 0.522686, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8358670473098755, Accuracy = 0.8125506043434143
    Iter #1161728:  Learning rate = 0.003191:   Batch Loss = 0.482067, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8188256621360779, Accuracy = 0.8242915272712708
    Iter #1162240:  Learning rate = 0.003191:   Batch Loss = 0.483153, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8911041021347046, Accuracy = 0.8008097410202026
    Iter #1162752:  Learning rate = 0.003191:   Batch Loss = 0.426028, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8758725523948669, Accuracy = 0.8085020184516907
    Iter #1163264:  Learning rate = 0.003191:   Batch Loss = 0.357886, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7587686777114868, Accuracy = 0.8404858112335205
    Iter #1163776:  Learning rate = 0.003191:   Batch Loss = 0.326421, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8066990375518799, Accuracy = 0.835627555847168
    Iter #1164288:  Learning rate = 0.003191:   Batch Loss = 0.337841, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6993774175643921, Accuracy = 0.8627530336380005
    Iter #1164800:  Learning rate = 0.003191:   Batch Loss = 0.272004, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7730783224105835, Accuracy = 0.8457489609718323
    Iter #1165312:  Learning rate = 0.003191:   Batch Loss = 0.350567, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7603887915611267, Accuracy = 0.8445343971252441
    Iter #1165824:  Learning rate = 0.003191:   Batch Loss = 0.302423, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7705600261688232, Accuracy = 0.8497975468635559
    Iter #1166336:  Learning rate = 0.003191:   Batch Loss = 0.310638, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7605092525482178, Accuracy = 0.8485829830169678
    Iter #1166848:  Learning rate = 0.003191:   Batch Loss = 0.271842, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7115128040313721, Accuracy = 0.8587044477462769
    Iter #1167360:  Learning rate = 0.003191:   Batch Loss = 0.326826, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.70171058177948, Accuracy = 0.8639675974845886
    Iter #1167872:  Learning rate = 0.003191:   Batch Loss = 0.270667, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.711235523223877, Accuracy = 0.8603239059448242
    Iter #1168384:  Learning rate = 0.003191:   Batch Loss = 0.253466, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6652122139930725, Accuracy = 0.8829959630966187
    Iter #1168896:  Learning rate = 0.003191:   Batch Loss = 0.272171, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.695464015007019, Accuracy = 0.8740890622138977
    Iter #1169408:  Learning rate = 0.003191:   Batch Loss = 0.252461, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6598495244979858, Accuracy = 0.8761133551597595
    Iter #1169920:  Learning rate = 0.003191:   Batch Loss = 0.237253, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6578351259231567, Accuracy = 0.8858299851417542
    Iter #1170432:  Learning rate = 0.003191:   Batch Loss = 0.269588, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6604134440422058, Accuracy = 0.8773279190063477
    Iter #1170944:  Learning rate = 0.003191:   Batch Loss = 0.270237, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6999320983886719, Accuracy = 0.8680161833763123
    Iter #1171456:  Learning rate = 0.003191:   Batch Loss = 0.238955, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6675378680229187, Accuracy = 0.8797571063041687
    Iter #1171968:  Learning rate = 0.003191:   Batch Loss = 0.258931, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6853609085083008, Accuracy = 0.8676113486289978
    Iter #1172480:  Learning rate = 0.003191:   Batch Loss = 0.263093, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6640306115150452, Accuracy = 0.8692307472229004
    Iter #1172992:  Learning rate = 0.003191:   Batch Loss = 0.271121, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7010199427604675, Accuracy = 0.8668016195297241
    Iter #1173504:  Learning rate = 0.003191:   Batch Loss = 0.241824, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7227860689163208, Accuracy = 0.8591092824935913
    Iter #1174016:  Learning rate = 0.003191:   Batch Loss = 0.224644, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6618458032608032, Accuracy = 0.8757085204124451
    Iter #1174528:  Learning rate = 0.003191:   Batch Loss = 0.240889, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6430985927581787, Accuracy = 0.8846153616905212
    Iter #1175040:  Learning rate = 0.003191:   Batch Loss = 0.234950, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6473603844642639, Accuracy = 0.8781376481056213
    Iter #1175552:  Learning rate = 0.003191:   Batch Loss = 0.229075, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6904180645942688, Accuracy = 0.873279333114624
    Iter #1176064:  Learning rate = 0.003191:   Batch Loss = 0.226410, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6687315106391907, Accuracy = 0.8672064542770386
    Iter #1176576:  Learning rate = 0.003191:   Batch Loss = 0.226202, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6542229652404785, Accuracy = 0.8773279190063477
    Iter #1177088:  Learning rate = 0.003191:   Batch Loss = 0.231034, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.670367419719696, Accuracy = 0.8708502054214478
    Iter #1177600:  Learning rate = 0.003191:   Batch Loss = 0.292361, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6316224336624146, Accuracy = 0.8886639475822449
    Iter #1178112:  Learning rate = 0.003191:   Batch Loss = 0.224957, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6352361440658569, Accuracy = 0.8854250907897949
    Iter #1178624:  Learning rate = 0.003191:   Batch Loss = 0.224306, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6582684516906738, Accuracy = 0.8801619410514832
    Iter #1179136:  Learning rate = 0.003191:   Batch Loss = 0.234776, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6371018290519714, Accuracy = 0.8882591128349304
    Iter #1179648:  Learning rate = 0.003191:   Batch Loss = 0.279851, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7019065022468567, Accuracy = 0.8708502054214478
    Iter #1180160:  Learning rate = 0.003191:   Batch Loss = 0.249089, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.667029619216919, Accuracy = 0.8817813992500305
    Iter #1180672:  Learning rate = 0.003191:   Batch Loss = 0.237385, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6351868510246277, Accuracy = 0.8898785710334778
    Iter #1181184:  Learning rate = 0.003191:   Batch Loss = 0.229323, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6386660933494568, Accuracy = 0.8805667757987976
    Iter #1181696:  Learning rate = 0.003191:   Batch Loss = 0.280565, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6217812299728394, Accuracy = 0.8935222625732422
    Iter #1182208:  Learning rate = 0.003191:   Batch Loss = 0.258374, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6380057334899902, Accuracy = 0.8846153616905212
    Iter #1182720:  Learning rate = 0.003191:   Batch Loss = 0.216163, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6427483558654785, Accuracy = 0.8870445489883423
    Iter #1183232:  Learning rate = 0.003191:   Batch Loss = 0.236353, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6644458174705505, Accuracy = 0.8801619410514832
    Iter #1183744:  Learning rate = 0.003191:   Batch Loss = 0.220050, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6613786816596985, Accuracy = 0.8753036260604858
    Iter #1184256:  Learning rate = 0.003191:   Batch Loss = 0.220905, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.650205135345459, Accuracy = 0.8773279190063477
    Iter #1184768:  Learning rate = 0.003191:   Batch Loss = 0.293327, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6403278112411499, Accuracy = 0.8858299851417542
    Iter #1185280:  Learning rate = 0.003191:   Batch Loss = 0.236377, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6393775343894958, Accuracy = 0.8813765048980713
    Iter #1185792:  Learning rate = 0.003191:   Batch Loss = 0.220303, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6195181608200073, Accuracy = 0.8874493837356567
    Iter #1186304:  Learning rate = 0.003191:   Batch Loss = 0.240248, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6506809592247009, Accuracy = 0.8744939565658569
    Iter #1186816:  Learning rate = 0.003191:   Batch Loss = 0.242863, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6587790846824646, Accuracy = 0.8842105269432068
    Iter #1187328:  Learning rate = 0.003191:   Batch Loss = 0.238687, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6708494424819946, Accuracy = 0.873279333114624
    Iter #1187840:  Learning rate = 0.003191:   Batch Loss = 0.255831, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6722152233123779, Accuracy = 0.865587055683136
    Iter #1188352:  Learning rate = 0.003191:   Batch Loss = 0.274999, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6661136150360107, Accuracy = 0.8712550401687622
    Iter #1188864:  Learning rate = 0.003191:   Batch Loss = 0.259912, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6655533313751221, Accuracy = 0.8716599345207214
    Iter #1189376:  Learning rate = 0.003191:   Batch Loss = 0.262627, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6604187488555908, Accuracy = 0.8700404763221741
    Iter #1189888:  Learning rate = 0.003191:   Batch Loss = 0.225025, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6290880441665649, Accuracy = 0.8850202560424805
    Iter #1190400:  Learning rate = 0.003191:   Batch Loss = 0.225985, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6424218416213989, Accuracy = 0.8805667757987976
    Iter #1190912:  Learning rate = 0.003191:   Batch Loss = 0.236880, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6459125876426697, Accuracy = 0.8838056921958923
    Iter #1191424:  Learning rate = 0.003191:   Batch Loss = 0.253936, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6407696008682251, Accuracy = 0.8898785710334778
    Iter #1191936:  Learning rate = 0.003191:   Batch Loss = 0.235861, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6393441557884216, Accuracy = 0.8825910687446594
    Iter #1192448:  Learning rate = 0.003191:   Batch Loss = 0.230221, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6209931373596191, Accuracy = 0.887854278087616
    Iter #1192960:  Learning rate = 0.003191:   Batch Loss = 0.221308, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6037237644195557, Accuracy = 0.8959513902664185
    Iter #1193472:  Learning rate = 0.003191:   Batch Loss = 0.236579, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6518945693969727, Accuracy = 0.8813765048980713
    Iter #1193984:  Learning rate = 0.003191:   Batch Loss = 0.262206, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6608803272247314, Accuracy = 0.8704453706741333
    Iter #1194496:  Learning rate = 0.003191:   Batch Loss = 0.282919, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6296170353889465, Accuracy = 0.8817813992500305
    Iter #1195008:  Learning rate = 0.003191:   Batch Loss = 0.225695, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6262836456298828, Accuracy = 0.8838056921958923
    Iter #1195520:  Learning rate = 0.003191:   Batch Loss = 0.225573, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6251193284988403, Accuracy = 0.8882591128349304
    Iter #1196032:  Learning rate = 0.003191:   Batch Loss = 0.213805, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6001293659210205, Accuracy = 0.8991903066635132
    Iter #1196544:  Learning rate = 0.003191:   Batch Loss = 0.221608, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6135657429695129, Accuracy = 0.8894736766815186
    Iter #1197056:  Learning rate = 0.003191:   Batch Loss = 0.209706, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6150867938995361, Accuracy = 0.8890688419342041
    Iter #1197568:  Learning rate = 0.003191:   Batch Loss = 0.209135, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.603732705116272, Accuracy = 0.8939270973205566
    Iter #1198080:  Learning rate = 0.003191:   Batch Loss = 0.214502, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5952948331832886, Accuracy = 0.8963562846183777
    Iter #1198592:  Learning rate = 0.003191:   Batch Loss = 0.211073, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5821570754051208, Accuracy = 0.8967611193656921
    Iter #1199104:  Learning rate = 0.003191:   Batch Loss = 0.202689, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.580906867980957, Accuracy = 0.8971660137176514
    Iter #1199616:  Learning rate = 0.003191:   Batch Loss = 0.203817, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5893126130104065, Accuracy = 0.8947368264198303
    Iter #1200128:  Learning rate = 0.003064:   Batch Loss = 0.198782, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5790179967880249, Accuracy = 0.8995951414108276
    Iter #1200640:  Learning rate = 0.003064:   Batch Loss = 0.199911, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.56563401222229, Accuracy = 0.901214599609375
    Iter #1201152:  Learning rate = 0.003064:   Batch Loss = 0.198745, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5623509883880615, Accuracy = 0.9004048705101013
    Iter #1201664:  Learning rate = 0.003064:   Batch Loss = 0.197162, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5650156736373901, Accuracy = 0.8999999761581421
    Iter #1202176:  Learning rate = 0.003064:   Batch Loss = 0.197731, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5654212236404419, Accuracy = 0.8983805775642395
    Iter #1202688:  Learning rate = 0.003064:   Batch Loss = 0.194472, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5612794160842896, Accuracy = 0.8995951414108276
    Iter #1203200:  Learning rate = 0.003064:   Batch Loss = 0.189282, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5596959590911865, Accuracy = 0.9016194343566895
    Iter #1203712:  Learning rate = 0.003064:   Batch Loss = 0.189933, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.56192547082901, Accuracy = 0.9016194343566895
    Iter #1204224:  Learning rate = 0.003064:   Batch Loss = 0.190416, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5615617632865906, Accuracy = 0.8999999761581421
    Iter #1204736:  Learning rate = 0.003064:   Batch Loss = 0.191969, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5636007189750671, Accuracy = 0.8983805775642395
    Iter #1205248:  Learning rate = 0.003064:   Batch Loss = 0.189496, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5585662126541138, Accuracy = 0.898785412311554
    Iter #1205760:  Learning rate = 0.003064:   Batch Loss = 0.189886, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5554724931716919, Accuracy = 0.901214599609375
    Iter #1206272:  Learning rate = 0.003064:   Batch Loss = 0.188756, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5523713231086731, Accuracy = 0.9004048705101013
    Iter #1206784:  Learning rate = 0.003064:   Batch Loss = 0.189098, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5503547191619873, Accuracy = 0.901214599609375
    Iter #1207296:  Learning rate = 0.003064:   Batch Loss = 0.188738, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5488599538803101, Accuracy = 0.898785412311554
    Iter #1207808:  Learning rate = 0.003064:   Batch Loss = 0.182289, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5499332547187805, Accuracy = 0.8983805775642395
    Iter #1208320:  Learning rate = 0.003064:   Batch Loss = 0.186000, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5493780970573425, Accuracy = 0.8967611193656921
    Iter #1208832:  Learning rate = 0.003064:   Batch Loss = 0.183843, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5500842928886414, Accuracy = 0.8983805775642395
    Iter #1209344:  Learning rate = 0.003064:   Batch Loss = 0.187923, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5494681000709534, Accuracy = 0.8999999761581421
    Iter #1209856:  Learning rate = 0.003064:   Batch Loss = 0.181138, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5519474744796753, Accuracy = 0.8975708484649658
    Iter #1210368:  Learning rate = 0.003064:   Batch Loss = 0.180440, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5505117774009705, Accuracy = 0.8971660137176514
    Iter #1210880:  Learning rate = 0.003064:   Batch Loss = 0.182133, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5474343299865723, Accuracy = 0.9004048705101013
    Iter #1211392:  Learning rate = 0.003064:   Batch Loss = 0.180946, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.549083411693573, Accuracy = 0.8999999761581421
    Iter #1211904:  Learning rate = 0.003064:   Batch Loss = 0.179942, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5456544160842896, Accuracy = 0.8967611193656921
    Iter #1212416:  Learning rate = 0.003064:   Batch Loss = 0.180413, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5441635251045227, Accuracy = 0.8947368264198303
    Iter #1212928:  Learning rate = 0.003064:   Batch Loss = 0.180072, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5557282567024231, Accuracy = 0.8943319916725159
    Iter #1213440:  Learning rate = 0.003064:   Batch Loss = 0.174997, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5644853115081787, Accuracy = 0.8939270973205566
    Iter #1213952:  Learning rate = 0.003064:   Batch Loss = 0.178777, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5513682961463928, Accuracy = 0.8971660137176514
    Iter #1214464:  Learning rate = 0.003064:   Batch Loss = 0.177889, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5436914563179016, Accuracy = 0.8971660137176514
    Iter #1214976:  Learning rate = 0.003064:   Batch Loss = 0.177045, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.549368679523468, Accuracy = 0.8967611193656921
    Iter #1215488:  Learning rate = 0.003064:   Batch Loss = 0.177955, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5545790791511536, Accuracy = 0.8943319916725159
    Iter #1216000:  Learning rate = 0.003064:   Batch Loss = 0.177539, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.553339958190918, Accuracy = 0.8959513902664185
    Iter #1216512:  Learning rate = 0.003064:   Batch Loss = 0.175378, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5485581159591675, Accuracy = 0.8951417207717896
    Iter #1217024:  Learning rate = 0.003064:   Batch Loss = 0.175376, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5481847524642944, Accuracy = 0.8939270973205566
    Iter #1217536:  Learning rate = 0.003064:   Batch Loss = 0.174453, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5486301779747009, Accuracy = 0.895546555519104
    Iter #1218048:  Learning rate = 0.003064:   Batch Loss = 0.172827, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5425049066543579, Accuracy = 0.8991903066635132
    Iter #1218560:  Learning rate = 0.003064:   Batch Loss = 0.172592, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5409531593322754, Accuracy = 0.8979756832122803
    Iter #1219072:  Learning rate = 0.003064:   Batch Loss = 0.173540, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5470665097236633, Accuracy = 0.8951417207717896
    Iter #1219584:  Learning rate = 0.003064:   Batch Loss = 0.173395, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5462861061096191, Accuracy = 0.8975708484649658
    Iter #1220096:  Learning rate = 0.003064:   Batch Loss = 0.175557, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5422289967536926, Accuracy = 0.8995951414108276
    Iter #1220608:  Learning rate = 0.003064:   Batch Loss = 0.172377, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5471117496490479, Accuracy = 0.8975708484649658
    Iter #1221120:  Learning rate = 0.003064:   Batch Loss = 0.171081, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5470729470252991, Accuracy = 0.8959513902664185
    Iter #1221632:  Learning rate = 0.003064:   Batch Loss = 0.170476, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5496686697006226, Accuracy = 0.8943319916725159
    Iter #1222144:  Learning rate = 0.003064:   Batch Loss = 0.172067, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5489262938499451, Accuracy = 0.8947368264198303
    Iter #1222656:  Learning rate = 0.003064:   Batch Loss = 0.169536, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5441926717758179, Accuracy = 0.8951417207717896
    Iter #1223168:  Learning rate = 0.003064:   Batch Loss = 0.168859, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5445250868797302, Accuracy = 0.898785412311554
    Iter #1223680:  Learning rate = 0.003064:   Batch Loss = 0.166389, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5472254753112793, Accuracy = 0.8979756832122803
    Iter #1224192:  Learning rate = 0.003064:   Batch Loss = 0.170002, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5444503426551819, Accuracy = 0.8951417207717896
    Iter #1224704:  Learning rate = 0.003064:   Batch Loss = 0.168973, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5473867654800415, Accuracy = 0.8979756832122803
    Iter #1225216:  Learning rate = 0.003064:   Batch Loss = 0.167007, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5368146896362305, Accuracy = 0.898785412311554
    Iter #1225728:  Learning rate = 0.003064:   Batch Loss = 0.165235, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5382264256477356, Accuracy = 0.8979756832122803
    Iter #1226240:  Learning rate = 0.003064:   Batch Loss = 0.171491, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5490790605545044, Accuracy = 0.8939270973205566
    Iter #1226752:  Learning rate = 0.003064:   Batch Loss = 0.163987, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5459264516830444, Accuracy = 0.8959513902664185
    Iter #1227264:  Learning rate = 0.003064:   Batch Loss = 0.168762, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.542161226272583, Accuracy = 0.8971660137176514
    Iter #1227776:  Learning rate = 0.003064:   Batch Loss = 0.165204, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5448982119560242, Accuracy = 0.8963562846183777
    Iter #1228288:  Learning rate = 0.003064:   Batch Loss = 0.163547, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5425986647605896, Accuracy = 0.895546555519104
    Iter #1228800:  Learning rate = 0.003064:   Batch Loss = 0.168276, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5435224771499634, Accuracy = 0.8959513902664185
    Iter #1229312:  Learning rate = 0.003064:   Batch Loss = 0.167360, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5474072098731995, Accuracy = 0.895546555519104
    Iter #1229824:  Learning rate = 0.003064:   Batch Loss = 0.163810, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5469841957092285, Accuracy = 0.8947368264198303
    Iter #1230336:  Learning rate = 0.003064:   Batch Loss = 0.164002, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.537589430809021, Accuracy = 0.8975708484649658
    Iter #1230848:  Learning rate = 0.003064:   Batch Loss = 0.165632, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5393270254135132, Accuracy = 0.8979756832122803
    Iter #1231360:  Learning rate = 0.003064:   Batch Loss = 0.161840, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5475994348526001, Accuracy = 0.8951417207717896
    Iter #1231872:  Learning rate = 0.003064:   Batch Loss = 0.166519, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5438718795776367, Accuracy = 0.8951417207717896
    Iter #1232384:  Learning rate = 0.003064:   Batch Loss = 0.162573, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5388401746749878, Accuracy = 0.8959513902664185
    Iter #1232896:  Learning rate = 0.003064:   Batch Loss = 0.160836, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5453910827636719, Accuracy = 0.8951417207717896
    Iter #1233408:  Learning rate = 0.003064:   Batch Loss = 0.160525, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5499633550643921, Accuracy = 0.8906882405281067
    Iter #1233920:  Learning rate = 0.003064:   Batch Loss = 0.165475, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5497641563415527, Accuracy = 0.8935222625732422
    Iter #1234432:  Learning rate = 0.003064:   Batch Loss = 0.161728, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5437929630279541, Accuracy = 0.8959513902664185
    Iter #1234944:  Learning rate = 0.003064:   Batch Loss = 0.158988, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.555353045463562, Accuracy = 0.8882591128349304
    Iter #1235456:  Learning rate = 0.003064:   Batch Loss = 0.163825, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.560603678226471, Accuracy = 0.887854278087616
    Iter #1235968:  Learning rate = 0.003064:   Batch Loss = 0.164061, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5574046969413757, Accuracy = 0.8898785710334778
    Iter #1236480:  Learning rate = 0.003064:   Batch Loss = 0.160536, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5687735676765442, Accuracy = 0.8890688419342041
    Iter #1236992:  Learning rate = 0.003064:   Batch Loss = 0.162412, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5626175403594971, Accuracy = 0.8906882405281067
    Iter #1237504:  Learning rate = 0.003064:   Batch Loss = 0.160486, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.549479067325592, Accuracy = 0.8951417207717896
    Iter #1238016:  Learning rate = 0.003064:   Batch Loss = 0.163141, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5514919757843018, Accuracy = 0.895546555519104
    Iter #1238528:  Learning rate = 0.003064:   Batch Loss = 0.156794, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5513057708740234, Accuracy = 0.8919028043746948
    Iter #1239040:  Learning rate = 0.003064:   Batch Loss = 0.157923, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5531550645828247, Accuracy = 0.8935222625732422
    Iter #1239552:  Learning rate = 0.003064:   Batch Loss = 0.158177, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5508108735084534, Accuracy = 0.8919028043746948
    Iter #1240064:  Learning rate = 0.003064:   Batch Loss = 0.161601, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5458030700683594, Accuracy = 0.8939270973205566
    Iter #1240576:  Learning rate = 0.003064:   Batch Loss = 0.160454, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5665213465690613, Accuracy = 0.887854278087616
    Iter #1241088:  Learning rate = 0.003064:   Batch Loss = 0.160871, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5521528720855713, Accuracy = 0.8898785710334778
    Iter #1241600:  Learning rate = 0.003064:   Batch Loss = 0.158881, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5484020113945007, Accuracy = 0.8910931348800659
    Iter #1242112:  Learning rate = 0.003064:   Batch Loss = 0.157300, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5551244020462036, Accuracy = 0.8890688419342041
    Iter #1242624:  Learning rate = 0.003064:   Batch Loss = 0.156201, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5492802858352661, Accuracy = 0.8906882405281067
    Iter #1243136:  Learning rate = 0.003064:   Batch Loss = 0.156380, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5549466013908386, Accuracy = 0.8914979696273804
    Iter #1243648:  Learning rate = 0.003064:   Batch Loss = 0.160048, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5596798658370972, Accuracy = 0.8919028043746948
    Iter #1244160:  Learning rate = 0.003064:   Batch Loss = 0.154312, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5492340326309204, Accuracy = 0.892307698726654
    Iter #1244672:  Learning rate = 0.003064:   Batch Loss = 0.153470, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5506203770637512, Accuracy = 0.8927125334739685
    Iter #1245184:  Learning rate = 0.003064:   Batch Loss = 0.155922, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5543451309204102, Accuracy = 0.8894736766815186
    Iter #1245696:  Learning rate = 0.003064:   Batch Loss = 0.161026, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5501079559326172, Accuracy = 0.8894736766815186
    Iter #1246208:  Learning rate = 0.003064:   Batch Loss = 0.162983, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5523436665534973, Accuracy = 0.8894736766815186
    Iter #1246720:  Learning rate = 0.003064:   Batch Loss = 0.153534, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5547388792037964, Accuracy = 0.8898785710334778
    Iter #1247232:  Learning rate = 0.003064:   Batch Loss = 0.153838, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5507249236106873, Accuracy = 0.8902834057807922
    Iter #1247744:  Learning rate = 0.003064:   Batch Loss = 0.153166, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.569606602191925, Accuracy = 0.8894736766815186
    Iter #1248256:  Learning rate = 0.003064:   Batch Loss = 0.155720, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5523433685302734, Accuracy = 0.8910931348800659
    Iter #1248768:  Learning rate = 0.003064:   Batch Loss = 0.152138, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5502102971076965, Accuracy = 0.8894736766815186
    Iter #1249280:  Learning rate = 0.003064:   Batch Loss = 0.152483, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5565232634544373, Accuracy = 0.8874493837356567
    Iter #1249792:  Learning rate = 0.003064:   Batch Loss = 0.154198, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5471383333206177, Accuracy = 0.8890688419342041
    Iter #1250304:  Learning rate = 0.003064:   Batch Loss = 0.154882, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5534341931343079, Accuracy = 0.8882591128349304
    Iter #1250816:  Learning rate = 0.003064:   Batch Loss = 0.152571, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5649917125701904, Accuracy = 0.887854278087616
    Iter #1251328:  Learning rate = 0.003064:   Batch Loss = 0.155621, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5592947006225586, Accuracy = 0.8886639475822449
    Iter #1251840:  Learning rate = 0.003064:   Batch Loss = 0.153827, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5558361411094666, Accuracy = 0.8854250907897949
    Iter #1252352:  Learning rate = 0.003064:   Batch Loss = 0.151453, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5572499632835388, Accuracy = 0.8882591128349304
    Iter #1252864:  Learning rate = 0.003064:   Batch Loss = 0.153407, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5615370273590088, Accuracy = 0.8850202560424805
    Iter #1253376:  Learning rate = 0.003064:   Batch Loss = 0.153001, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5506077408790588, Accuracy = 0.8931174278259277
    Iter #1253888:  Learning rate = 0.003064:   Batch Loss = 0.155948, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5795460343360901, Accuracy = 0.8825910687446594
    Iter #1254400:  Learning rate = 0.003064:   Batch Loss = 0.150948, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.558190107345581, Accuracy = 0.8882591128349304
    Iter #1254912:  Learning rate = 0.003064:   Batch Loss = 0.150098, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5549341440200806, Accuracy = 0.8858299851417542
    Iter #1255424:  Learning rate = 0.003064:   Batch Loss = 0.153696, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5540279746055603, Accuracy = 0.8886639475822449
    Iter #1255936:  Learning rate = 0.003064:   Batch Loss = 0.149473, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5533617734909058, Accuracy = 0.8910931348800659
    Iter #1256448:  Learning rate = 0.003064:   Batch Loss = 0.152709, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5702335238456726, Accuracy = 0.8829959630966187
    Iter #1256960:  Learning rate = 0.003064:   Batch Loss = 0.151555, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5534618496894836, Accuracy = 0.8862348198890686
    Iter #1257472:  Learning rate = 0.003064:   Batch Loss = 0.159221, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5668526291847229, Accuracy = 0.8874493837356567
    Iter #1257984:  Learning rate = 0.003064:   Batch Loss = 0.162271, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5785274505615234, Accuracy = 0.878947377204895
    Iter #1258496:  Learning rate = 0.003064:   Batch Loss = 0.235526, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6856997609138489, Accuracy = 0.8485829830169678
    Iter #1259008:  Learning rate = 0.003064:   Batch Loss = 0.542068, Accuracy = 0.875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7418869733810425, Accuracy = 0.8336032629013062
    Iter #1259520:  Learning rate = 0.003064:   Batch Loss = 0.493207, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7699567079544067, Accuracy = 0.8226720690727234
    Iter #1260032:  Learning rate = 0.003064:   Batch Loss = 0.481571, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8601831793785095, Accuracy = 0.7914980053901672
    Iter #1260544:  Learning rate = 0.003064:   Batch Loss = 0.327044, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9083395004272461, Accuracy = 0.7858299612998962
    Iter #1261056:  Learning rate = 0.003064:   Batch Loss = 0.438756, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.866013765335083, Accuracy = 0.7947368621826172
    Iter #1261568:  Learning rate = 0.003064:   Batch Loss = 0.532248, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8757243752479553, Accuracy = 0.7979757189750671
    Iter #1262080:  Learning rate = 0.003064:   Batch Loss = 0.440143, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8203737735748291, Accuracy = 0.8093117475509644
    Iter #1262592:  Learning rate = 0.003064:   Batch Loss = 0.533291, Accuracy = 0.875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8978643417358398, Accuracy = 0.7898785471916199
    Iter #1263104:  Learning rate = 0.003064:   Batch Loss = 0.513040, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8737407326698303, Accuracy = 0.7821862101554871
    Iter #1263616:  Learning rate = 0.003064:   Batch Loss = 0.536284, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8582135438919067, Accuracy = 0.8089068531990051
    Iter #1264128:  Learning rate = 0.003064:   Batch Loss = 0.769088, Accuracy = 0.8125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.891075074672699, Accuracy = 0.7927125692367554
    Iter #1264640:  Learning rate = 0.003064:   Batch Loss = 0.585598, Accuracy = 0.859375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7868196368217468, Accuracy = 0.826720654964447
    Iter #1265152:  Learning rate = 0.003064:   Batch Loss = 0.684005, Accuracy = 0.859375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7868586778640747, Accuracy = 0.8331983685493469
    Iter #1265664:  Learning rate = 0.003064:   Batch Loss = 0.348523, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8284797668457031, Accuracy = 0.8194332122802734
    Iter #1266176:  Learning rate = 0.003064:   Batch Loss = 0.374301, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7831652760505676, Accuracy = 0.8425101041793823
    Iter #1266688:  Learning rate = 0.003064:   Batch Loss = 0.337984, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8224841356277466, Accuracy = 0.8190283179283142
    Iter #1267200:  Learning rate = 0.003064:   Batch Loss = 0.332188, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.794188380241394, Accuracy = 0.8275303840637207
    Iter #1267712:  Learning rate = 0.003064:   Batch Loss = 0.633887, Accuracy = 0.875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7857235670089722, Accuracy = 0.8429149985313416
    Iter #1268224:  Learning rate = 0.003064:   Batch Loss = 0.432491, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7525680661201477, Accuracy = 0.848987877368927
    Iter #1268736:  Learning rate = 0.003064:   Batch Loss = 0.310674, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.788825511932373, Accuracy = 0.8344129323959351
    Iter #1269248:  Learning rate = 0.003064:   Batch Loss = 0.369546, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7484659552574158, Accuracy = 0.8530364632606506
    Iter #1269760:  Learning rate = 0.003064:   Batch Loss = 0.325793, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7099582552909851, Accuracy = 0.8663967847824097
    Iter #1270272:  Learning rate = 0.003064:   Batch Loss = 0.269875, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7083364129066467, Accuracy = 0.8587044477462769
    Iter #1270784:  Learning rate = 0.003064:   Batch Loss = 0.276858, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6808525323867798, Accuracy = 0.8680161833763123
    Iter #1271296:  Learning rate = 0.003064:   Batch Loss = 0.368646, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7080256938934326, Accuracy = 0.8672064542770386
    Iter #1271808:  Learning rate = 0.003064:   Batch Loss = 0.275891, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6801939606666565, Accuracy = 0.8684210777282715
    Iter #1272320:  Learning rate = 0.003064:   Batch Loss = 0.288827, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6837201118469238, Accuracy = 0.8692307472229004
    Iter #1272832:  Learning rate = 0.003064:   Batch Loss = 0.315478, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6574739217758179, Accuracy = 0.8744939565658569
    Iter #1273344:  Learning rate = 0.003064:   Batch Loss = 0.234516, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.652672290802002, Accuracy = 0.8858299851417542
    Iter #1273856:  Learning rate = 0.003064:   Batch Loss = 0.329842, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6591755151748657, Accuracy = 0.8785424828529358
    Iter #1274368:  Learning rate = 0.003064:   Batch Loss = 0.250516, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6826057434082031, Accuracy = 0.8639675974845886
    Iter #1274880:  Learning rate = 0.003064:   Batch Loss = 0.266945, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6639934778213501, Accuracy = 0.8797571063041687
    Iter #1275392:  Learning rate = 0.003064:   Batch Loss = 0.240729, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6493561863899231, Accuracy = 0.8769230842590332
    Iter #1275904:  Learning rate = 0.003064:   Batch Loss = 0.237614, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6326446533203125, Accuracy = 0.882186233997345
    Iter #1276416:  Learning rate = 0.003064:   Batch Loss = 0.227094, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6081219911575317, Accuracy = 0.892307698726654
    Iter #1276928:  Learning rate = 0.003064:   Batch Loss = 0.221610, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6025865077972412, Accuracy = 0.8906882405281067
    Iter #1277440:  Learning rate = 0.003064:   Batch Loss = 0.238928, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6040546894073486, Accuracy = 0.8947368264198303
    Iter #1277952:  Learning rate = 0.003064:   Batch Loss = 0.222223, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6739727854728699, Accuracy = 0.8631578683853149
    Iter #1278464:  Learning rate = 0.003064:   Batch Loss = 0.225514, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6283749341964722, Accuracy = 0.8834007978439331
    Iter #1278976:  Learning rate = 0.003064:   Batch Loss = 0.236605, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7002537250518799, Accuracy = 0.8607287406921387
    Iter #1279488:  Learning rate = 0.003064:   Batch Loss = 0.226840, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6277592182159424, Accuracy = 0.8862348198890686
    Iter #1280000:  Learning rate = 0.003064:   Batch Loss = 0.242822, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6261036992073059, Accuracy = 0.8886639475822449
    Iter #1280512:  Learning rate = 0.003064:   Batch Loss = 0.285683, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6512259840965271, Accuracy = 0.882186233997345
    Iter #1281024:  Learning rate = 0.003064:   Batch Loss = 0.218087, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.59702068567276, Accuracy = 0.895546555519104
    Iter #1281536:  Learning rate = 0.003064:   Batch Loss = 0.240399, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6025227308273315, Accuracy = 0.8943319916725159
    Iter #1282048:  Learning rate = 0.003064:   Batch Loss = 0.234751, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6268070340156555, Accuracy = 0.8858299851417542
    Iter #1282560:  Learning rate = 0.003064:   Batch Loss = 0.209152, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6209943890571594, Accuracy = 0.8858299851417542
    Iter #1283072:  Learning rate = 0.003064:   Batch Loss = 0.217653, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.596716046333313, Accuracy = 0.8866396546363831
    Iter #1283584:  Learning rate = 0.003064:   Batch Loss = 0.209166, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5837278962135315, Accuracy = 0.898785412311554
    Iter #1284096:  Learning rate = 0.003064:   Batch Loss = 0.218473, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5977468490600586, Accuracy = 0.8910931348800659
    Iter #1284608:  Learning rate = 0.003064:   Batch Loss = 0.202831, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5841197371482849, Accuracy = 0.8882591128349304
    Iter #1285120:  Learning rate = 0.003064:   Batch Loss = 0.206846, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.578088641166687, Accuracy = 0.8906882405281067
    Iter #1285632:  Learning rate = 0.003064:   Batch Loss = 0.204183, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5737952589988708, Accuracy = 0.8939270973205566
    Iter #1286144:  Learning rate = 0.003064:   Batch Loss = 0.199449, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5777219533920288, Accuracy = 0.8959513902664185
    Iter #1286656:  Learning rate = 0.003064:   Batch Loss = 0.203491, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5670870542526245, Accuracy = 0.898785412311554
    Iter #1287168:  Learning rate = 0.003064:   Batch Loss = 0.207525, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5630933046340942, Accuracy = 0.898785412311554
    Iter #1287680:  Learning rate = 0.003064:   Batch Loss = 0.197356, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.556354284286499, Accuracy = 0.8979756832122803
    Iter #1288192:  Learning rate = 0.003064:   Batch Loss = 0.194836, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5533440709114075, Accuracy = 0.8999999761581421
    Iter #1288704:  Learning rate = 0.003064:   Batch Loss = 0.194734, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.549507737159729, Accuracy = 0.9004048705101013
    Iter #1289216:  Learning rate = 0.003064:   Batch Loss = 0.191608, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5532165169715881, Accuracy = 0.8967611193656921
    Iter #1289728:  Learning rate = 0.003064:   Batch Loss = 0.193650, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5554831624031067, Accuracy = 0.8963562846183777
    Iter #1290240:  Learning rate = 0.003064:   Batch Loss = 0.192624, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5530120730400085, Accuracy = 0.8975708484649658
    Iter #1290752:  Learning rate = 0.003064:   Batch Loss = 0.190726, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.550263524055481, Accuracy = 0.8979756832122803
    Iter #1291264:  Learning rate = 0.003064:   Batch Loss = 0.189132, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5482299327850342, Accuracy = 0.8967611193656921
    Iter #1291776:  Learning rate = 0.003064:   Batch Loss = 0.187080, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5460553169250488, Accuracy = 0.898785412311554
    Iter #1292288:  Learning rate = 0.003064:   Batch Loss = 0.184266, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5487160086631775, Accuracy = 0.8967611193656921
    Iter #1292800:  Learning rate = 0.003064:   Batch Loss = 0.186355, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5477616786956787, Accuracy = 0.8951417207717896
    Iter #1293312:  Learning rate = 0.003064:   Batch Loss = 0.183483, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5453267693519592, Accuracy = 0.8959513902664185
    Iter #1293824:  Learning rate = 0.003064:   Batch Loss = 0.184340, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5410813689231873, Accuracy = 0.8979756832122803
    Iter #1294336:  Learning rate = 0.003064:   Batch Loss = 0.181276, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5380107760429382, Accuracy = 0.9008097052574158
    Iter #1294848:  Learning rate = 0.003064:   Batch Loss = 0.183616, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5398171544075012, Accuracy = 0.8995951414108276
    Iter #1295360:  Learning rate = 0.003064:   Batch Loss = 0.181907, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5432462096214294, Accuracy = 0.8971660137176514
    Iter #1295872:  Learning rate = 0.003064:   Batch Loss = 0.182875, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5433406233787537, Accuracy = 0.8995951414108276
    Iter #1296384:  Learning rate = 0.003064:   Batch Loss = 0.177958, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5413844585418701, Accuracy = 0.898785412311554
    Iter #1296896:  Learning rate = 0.003064:   Batch Loss = 0.181824, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5404622554779053, Accuracy = 0.8975708484649658
    Iter #1297408:  Learning rate = 0.003064:   Batch Loss = 0.180828, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.539160966873169, Accuracy = 0.8975708484649658
    Iter #1297920:  Learning rate = 0.003064:   Batch Loss = 0.182889, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5403476357460022, Accuracy = 0.8963562846183777
    Iter #1298432:  Learning rate = 0.003064:   Batch Loss = 0.177377, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5402724742889404, Accuracy = 0.8983805775642395
    Iter #1298944:  Learning rate = 0.003064:   Batch Loss = 0.182103, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5357694029808044, Accuracy = 0.9004048705101013
    Iter #1299456:  Learning rate = 0.003064:   Batch Loss = 0.180082, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5392033457756042, Accuracy = 0.8995951414108276
    Iter #1299968:  Learning rate = 0.003064:   Batch Loss = 0.177802, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5407415628433228, Accuracy = 0.8963562846183777
    Iter #1300480:  Learning rate = 0.002941:   Batch Loss = 0.176429, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5411683320999146, Accuracy = 0.8971660137176514
    Iter #1300992:  Learning rate = 0.002941:   Batch Loss = 0.180545, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5376043915748596, Accuracy = 0.8979756832122803
    Iter #1301504:  Learning rate = 0.002941:   Batch Loss = 0.173668, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5318952798843384, Accuracy = 0.9004048705101013
    Iter #1302016:  Learning rate = 0.002941:   Batch Loss = 0.174143, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5303685665130615, Accuracy = 0.9004048705101013
    Iter #1302528:  Learning rate = 0.002941:   Batch Loss = 0.172709, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5308359861373901, Accuracy = 0.8979756832122803
    Iter #1303040:  Learning rate = 0.002941:   Batch Loss = 0.178482, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5334075689315796, Accuracy = 0.8975708484649658
    Iter #1303552:  Learning rate = 0.002941:   Batch Loss = 0.172954, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5360154509544373, Accuracy = 0.8983805775642395
    Iter #1304064:  Learning rate = 0.002941:   Batch Loss = 0.170840, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5363808274269104, Accuracy = 0.8999999761581421
    Iter #1304576:  Learning rate = 0.002941:   Batch Loss = 0.174809, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5366318821907043, Accuracy = 0.8991903066635132
    Iter #1305088:  Learning rate = 0.002941:   Batch Loss = 0.171074, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5321678519248962, Accuracy = 0.8991903066635132
    Iter #1305600:  Learning rate = 0.002941:   Batch Loss = 0.169990, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.535213053226471, Accuracy = 0.8963562846183777
    Iter #1306112:  Learning rate = 0.002941:   Batch Loss = 0.172581, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5377575159072876, Accuracy = 0.8959513902664185
    Iter #1306624:  Learning rate = 0.002941:   Batch Loss = 0.168665, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5355062484741211, Accuracy = 0.8971660137176514
    Iter #1307136:  Learning rate = 0.002941:   Batch Loss = 0.167437, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5361995100975037, Accuracy = 0.8939270973205566
    Iter #1307648:  Learning rate = 0.002941:   Batch Loss = 0.166951, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5329234004020691, Accuracy = 0.8951417207717896
    Iter #1308160:  Learning rate = 0.002941:   Batch Loss = 0.168322, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.528594434261322, Accuracy = 0.8967611193656921
    Iter #1308672:  Learning rate = 0.002941:   Batch Loss = 0.167812, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5316635370254517, Accuracy = 0.8935222625732422
    Iter #1309184:  Learning rate = 0.002941:   Batch Loss = 0.167566, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5361043810844421, Accuracy = 0.8935222625732422
    Iter #1309696:  Learning rate = 0.002941:   Batch Loss = 0.166621, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5335984826087952, Accuracy = 0.8943319916725159
    Iter #1310208:  Learning rate = 0.002941:   Batch Loss = 0.164274, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.533257007598877, Accuracy = 0.8927125334739685
    Iter #1310720:  Learning rate = 0.002941:   Batch Loss = 0.164989, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5408295392990112, Accuracy = 0.8935222625732422
    Iter #1311232:  Learning rate = 0.002941:   Batch Loss = 0.168984, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5329309105873108, Accuracy = 0.8951417207717896
    Iter #1311744:  Learning rate = 0.002941:   Batch Loss = 0.162941, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5318654179573059, Accuracy = 0.8935222625732422
    Iter #1312256:  Learning rate = 0.002941:   Batch Loss = 0.162533, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5326506495475769, Accuracy = 0.8906882405281067
    Iter #1312768:  Learning rate = 0.002941:   Batch Loss = 0.162750, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5306780338287354, Accuracy = 0.8919028043746948
    Iter #1313280:  Learning rate = 0.002941:   Batch Loss = 0.162957, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5345171093940735, Accuracy = 0.8939270973205566
    Iter #1313792:  Learning rate = 0.002941:   Batch Loss = 0.166779, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5364776253700256, Accuracy = 0.8927125334739685
    Iter #1314304:  Learning rate = 0.002941:   Batch Loss = 0.163162, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.539242684841156, Accuracy = 0.8935222625732422
    Iter #1314816:  Learning rate = 0.002941:   Batch Loss = 0.162921, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5351852178573608, Accuracy = 0.8939270973205566
    Iter #1315328:  Learning rate = 0.002941:   Batch Loss = 0.163200, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5403212308883667, Accuracy = 0.892307698726654
    Iter #1315840:  Learning rate = 0.002941:   Batch Loss = 0.162094, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.545508086681366, Accuracy = 0.8894736766815186
    Iter #1316352:  Learning rate = 0.002941:   Batch Loss = 0.162357, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5439163446426392, Accuracy = 0.8947368264198303
    Iter #1316864:  Learning rate = 0.002941:   Batch Loss = 0.160788, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5390307307243347, Accuracy = 0.8971660137176514
    Iter #1317376:  Learning rate = 0.002941:   Batch Loss = 0.159465, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.540263831615448, Accuracy = 0.8910931348800659
    Iter #1317888:  Learning rate = 0.002941:   Batch Loss = 0.161918, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5357629060745239, Accuracy = 0.8927125334739685
    Iter #1318400:  Learning rate = 0.002941:   Batch Loss = 0.157752, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.540471076965332, Accuracy = 0.8906882405281067
    Iter #1318912:  Learning rate = 0.002941:   Batch Loss = 0.157617, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5423399209976196, Accuracy = 0.8910931348800659
    Iter #1319424:  Learning rate = 0.002941:   Batch Loss = 0.154688, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5420149564743042, Accuracy = 0.8898785710334778
    Iter #1319936:  Learning rate = 0.002941:   Batch Loss = 0.157956, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5446088910102844, Accuracy = 0.8898785710334778
    Iter #1320448:  Learning rate = 0.002941:   Batch Loss = 0.157918, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5412144660949707, Accuracy = 0.8902834057807922
    Iter #1320960:  Learning rate = 0.002941:   Batch Loss = 0.155990, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5427424907684326, Accuracy = 0.887854278087616
    Iter #1321472:  Learning rate = 0.002941:   Batch Loss = 0.157259, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5442348122596741, Accuracy = 0.8902834057807922
    Iter #1321984:  Learning rate = 0.002941:   Batch Loss = 0.156078, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5476506948471069, Accuracy = 0.8931174278259277
    Iter #1322496:  Learning rate = 0.002941:   Batch Loss = 0.158823, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5571758151054382, Accuracy = 0.8902834057807922
    Iter #1323008:  Learning rate = 0.002941:   Batch Loss = 0.152504, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5432242155075073, Accuracy = 0.887854278087616
    Iter #1323520:  Learning rate = 0.002941:   Batch Loss = 0.152999, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5427874326705933, Accuracy = 0.8898785710334778
    Iter #1324032:  Learning rate = 0.002941:   Batch Loss = 0.156350, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5440139174461365, Accuracy = 0.8927125334739685
    Iter #1324544:  Learning rate = 0.002941:   Batch Loss = 0.154961, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.536716639995575, Accuracy = 0.8894736766815186
    Iter #1325056:  Learning rate = 0.002941:   Batch Loss = 0.154628, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.533753514289856, Accuracy = 0.8927125334739685
    Iter #1325568:  Learning rate = 0.002941:   Batch Loss = 0.152587, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5374849438667297, Accuracy = 0.8906882405281067
    Iter #1326080:  Learning rate = 0.002941:   Batch Loss = 0.153256, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5369126200675964, Accuracy = 0.8858299851417542
    Iter #1326592:  Learning rate = 0.002941:   Batch Loss = 0.153267, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.542130172252655, Accuracy = 0.8874493837356567
    Iter #1327104:  Learning rate = 0.002941:   Batch Loss = 0.156846, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5452805757522583, Accuracy = 0.887854278087616
    Iter #1327616:  Learning rate = 0.002941:   Batch Loss = 0.152558, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5422545671463013, Accuracy = 0.8931174278259277
    Iter #1328128:  Learning rate = 0.002941:   Batch Loss = 0.150276, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.549216628074646, Accuracy = 0.8866396546363831
    Iter #1328640:  Learning rate = 0.002941:   Batch Loss = 0.150325, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5443968176841736, Accuracy = 0.8870445489883423
    Iter #1329152:  Learning rate = 0.002941:   Batch Loss = 0.156419, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5481300950050354, Accuracy = 0.8846153616905212
    Iter #1329664:  Learning rate = 0.002941:   Batch Loss = 0.153613, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5447594523429871, Accuracy = 0.8882591128349304
    Iter #1330176:  Learning rate = 0.002941:   Batch Loss = 0.150661, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5406219363212585, Accuracy = 0.8882591128349304
    Iter #1330688:  Learning rate = 0.002941:   Batch Loss = 0.152505, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5457273721694946, Accuracy = 0.8898785710334778
    Iter #1331200:  Learning rate = 0.002941:   Batch Loss = 0.152388, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5395532846450806, Accuracy = 0.8874493837356567
    Iter #1331712:  Learning rate = 0.002941:   Batch Loss = 0.149633, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5478756427764893, Accuracy = 0.8886639475822449
    Iter #1332224:  Learning rate = 0.002941:   Batch Loss = 0.148683, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5380832552909851, Accuracy = 0.8874493837356567
    Iter #1332736:  Learning rate = 0.002941:   Batch Loss = 0.153623, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5418276786804199, Accuracy = 0.887854278087616
    Iter #1333248:  Learning rate = 0.002941:   Batch Loss = 0.151452, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5463229417800903, Accuracy = 0.8882591128349304
    Iter #1333760:  Learning rate = 0.002941:   Batch Loss = 0.149116, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.549830436706543, Accuracy = 0.8842105269432068
    Iter #1334272:  Learning rate = 0.002941:   Batch Loss = 0.150430, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5560371279716492, Accuracy = 0.8870445489883423
    Iter #1334784:  Learning rate = 0.002941:   Batch Loss = 0.147768, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.536088228225708, Accuracy = 0.8882591128349304
    Iter #1335296:  Learning rate = 0.002941:   Batch Loss = 0.150588, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5409805774688721, Accuracy = 0.8886639475822449
    Iter #1335808:  Learning rate = 0.002941:   Batch Loss = 0.147518, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5512147545814514, Accuracy = 0.8862348198890686
    Iter #1336320:  Learning rate = 0.002941:   Batch Loss = 0.150639, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5488008260726929, Accuracy = 0.8838056921958923
    Iter #1336832:  Learning rate = 0.002941:   Batch Loss = 0.149581, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5501060485839844, Accuracy = 0.8866396546363831
    Iter #1337344:  Learning rate = 0.002941:   Batch Loss = 0.149280, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.542616605758667, Accuracy = 0.8910931348800659
    Iter #1337856:  Learning rate = 0.002941:   Batch Loss = 0.145639, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5536940693855286, Accuracy = 0.8874493837356567
    Iter #1338368:  Learning rate = 0.002941:   Batch Loss = 0.151207, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5449941158294678, Accuracy = 0.8870445489883423
    Iter #1338880:  Learning rate = 0.002941:   Batch Loss = 0.147873, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.541680634021759, Accuracy = 0.8846153616905212
    Iter #1339392:  Learning rate = 0.002941:   Batch Loss = 0.150248, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5491812229156494, Accuracy = 0.8882591128349304
    Iter #1339904:  Learning rate = 0.002941:   Batch Loss = 0.147264, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.557489812374115, Accuracy = 0.8850202560424805
    Iter #1340416:  Learning rate = 0.002941:   Batch Loss = 0.150604, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5437849760055542, Accuracy = 0.887854278087616
    Iter #1340928:  Learning rate = 0.002941:   Batch Loss = 0.143344, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.547965407371521, Accuracy = 0.8870445489883423
    Iter #1341440:  Learning rate = 0.002941:   Batch Loss = 0.149199, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5495101809501648, Accuracy = 0.8838056921958923
    Iter #1341952:  Learning rate = 0.002941:   Batch Loss = 0.142593, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5320956707000732, Accuracy = 0.8914979696273804
    Iter #1342464:  Learning rate = 0.002941:   Batch Loss = 0.149622, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5652579069137573, Accuracy = 0.8834007978439331
    Iter #1342976:  Learning rate = 0.002941:   Batch Loss = 0.150310, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5567000508308411, Accuracy = 0.8825910687446594
    Iter #1343488:  Learning rate = 0.002941:   Batch Loss = 0.153719, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5930656790733337, Accuracy = 0.8724696636199951
    Iter #1344000:  Learning rate = 0.002941:   Batch Loss = 0.147153, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5565863847732544, Accuracy = 0.8882591128349304
    Iter #1344512:  Learning rate = 0.002941:   Batch Loss = 0.147889, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5824962854385376, Accuracy = 0.8797571063041687
    Iter #1345024:  Learning rate = 0.002941:   Batch Loss = 0.151934, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5779443979263306, Accuracy = 0.8769230842590332
    Iter #1345536:  Learning rate = 0.002941:   Batch Loss = 0.145200, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5540214776992798, Accuracy = 0.8886639475822449
    Iter #1346048:  Learning rate = 0.002941:   Batch Loss = 0.179644, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5492278337478638, Accuracy = 0.8874493837356567
    Iter #1346560:  Learning rate = 0.002941:   Batch Loss = 0.259543, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.625137209892273, Accuracy = 0.8720647692680359
    Iter #1347072:  Learning rate = 0.002941:   Batch Loss = 0.364428, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8862367272377014, Accuracy = 0.7935222387313843
    Iter #1347584:  Learning rate = 0.002941:   Batch Loss = 0.578166, Accuracy = 0.8125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8975072503089905, Accuracy = 0.7797570824623108
    Iter #1348096:  Learning rate = 0.002941:   Batch Loss = 0.377194, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9108396172523499, Accuracy = 0.7785425186157227
    Iter #1348608:  Learning rate = 0.002941:   Batch Loss = 0.874265, Accuracy = 0.796875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8288618922233582, Accuracy = 0.8036437034606934
    Iter #1349120:  Learning rate = 0.002941:   Batch Loss = 0.427486, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8768532872200012, Accuracy = 0.7963562607765198
    Iter #1349632:  Learning rate = 0.002941:   Batch Loss = 0.494456, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9713895916938782, Accuracy = 0.755465567111969
    Iter #1350144:  Learning rate = 0.002941:   Batch Loss = 0.426008, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8381742835044861, Accuracy = 0.8121457695960999
    Iter #1350656:  Learning rate = 0.002941:   Batch Loss = 0.369231, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.797626256942749, Accuracy = 0.8275303840637207
    Iter #1351168:  Learning rate = 0.002941:   Batch Loss = 0.389037, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9791510105133057, Accuracy = 0.7530364394187927
    Iter #1351680:  Learning rate = 0.002941:   Batch Loss = 0.526394, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8641384840011597, Accuracy = 0.8032388687133789
    Iter #1352192:  Learning rate = 0.002941:   Batch Loss = 0.538051, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7919653654098511, Accuracy = 0.8255060911178589
    Iter #1352704:  Learning rate = 0.002941:   Batch Loss = 0.430058, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7765025496482849, Accuracy = 0.8303643465042114
    Iter #1353216:  Learning rate = 0.002941:   Batch Loss = 0.405179, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7616445422172546, Accuracy = 0.8372469544410706
    Iter #1353728:  Learning rate = 0.002941:   Batch Loss = 0.366587, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7672458291053772, Accuracy = 0.8449392914772034
    Iter #1354240:  Learning rate = 0.002941:   Batch Loss = 0.372152, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7831544876098633, Accuracy = 0.8246963620185852
    Iter #1354752:  Learning rate = 0.002941:   Batch Loss = 0.459368, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7309027910232544, Accuracy = 0.8538461327552795
    Iter #1355264:  Learning rate = 0.002941:   Batch Loss = 0.431350, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8092707395553589, Accuracy = 0.8271254897117615
    Iter #1355776:  Learning rate = 0.002941:   Batch Loss = 0.295526, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8251934051513672, Accuracy = 0.8234817981719971
    Iter #1356288:  Learning rate = 0.002941:   Batch Loss = 0.439653, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7631313800811768, Accuracy = 0.8461538553237915
    Iter #1356800:  Learning rate = 0.002941:   Batch Loss = 0.331720, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7396486401557922, Accuracy = 0.8441295623779297
    Iter #1357312:  Learning rate = 0.002941:   Batch Loss = 0.315636, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6938250064849854, Accuracy = 0.8639675974845886
    Iter #1357824:  Learning rate = 0.002941:   Batch Loss = 0.284246, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6956411004066467, Accuracy = 0.8635627627372742
    Iter #1358336:  Learning rate = 0.002941:   Batch Loss = 0.329495, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6532742381095886, Accuracy = 0.8777328133583069
    Iter #1358848:  Learning rate = 0.002941:   Batch Loss = 0.242767, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6430042386054993, Accuracy = 0.8809716701507568
    Iter #1359360:  Learning rate = 0.002941:   Batch Loss = 0.235384, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6556605100631714, Accuracy = 0.8769230842590332
    Iter #1359872:  Learning rate = 0.002941:   Batch Loss = 0.229965, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6527001261711121, Accuracy = 0.8777328133583069
    Iter #1360384:  Learning rate = 0.002941:   Batch Loss = 0.255487, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6702502965927124, Accuracy = 0.8797571063041687
    Iter #1360896:  Learning rate = 0.002941:   Batch Loss = 0.255187, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6269322633743286, Accuracy = 0.8931174278259277
    Iter #1361408:  Learning rate = 0.002941:   Batch Loss = 0.260538, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6144400835037231, Accuracy = 0.8919028043746948
    Iter #1361920:  Learning rate = 0.002941:   Batch Loss = 0.233270, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6573494672775269, Accuracy = 0.8744939565658569
    Iter #1362432:  Learning rate = 0.002941:   Batch Loss = 0.247313, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6521916389465332, Accuracy = 0.8805667757987976
    Iter #1362944:  Learning rate = 0.002941:   Batch Loss = 0.225750, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6176027059555054, Accuracy = 0.8931174278259277
    Iter #1363456:  Learning rate = 0.002941:   Batch Loss = 0.217306, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6377464532852173, Accuracy = 0.887854278087616
    Iter #1363968:  Learning rate = 0.002941:   Batch Loss = 0.213158, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6388347744941711, Accuracy = 0.8797571063041687
    Iter #1364480:  Learning rate = 0.002941:   Batch Loss = 0.211422, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6015020608901978, Accuracy = 0.8898785710334778
    Iter #1364992:  Learning rate = 0.002941:   Batch Loss = 0.211338, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5937098860740662, Accuracy = 0.8935222625732422
    Iter #1365504:  Learning rate = 0.002941:   Batch Loss = 0.204153, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.615719735622406, Accuracy = 0.8894736766815186
    Iter #1366016:  Learning rate = 0.002941:   Batch Loss = 0.213050, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5989723801612854, Accuracy = 0.8971660137176514
    Iter #1366528:  Learning rate = 0.002941:   Batch Loss = 0.217678, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5936872959136963, Accuracy = 0.8947368264198303
    Iter #1367040:  Learning rate = 0.002941:   Batch Loss = 0.217670, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6109901666641235, Accuracy = 0.8886639475822449
    Iter #1367552:  Learning rate = 0.002941:   Batch Loss = 0.200409, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.598387598991394, Accuracy = 0.8935222625732422
    Iter #1368064:  Learning rate = 0.002941:   Batch Loss = 0.208641, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5803705453872681, Accuracy = 0.895546555519104
    Iter #1368576:  Learning rate = 0.002941:   Batch Loss = 0.199283, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5716546773910522, Accuracy = 0.9036437273025513
    Iter #1369088:  Learning rate = 0.002941:   Batch Loss = 0.247053, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.566712498664856, Accuracy = 0.9028339982032776
    Iter #1369600:  Learning rate = 0.002941:   Batch Loss = 0.194762, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5604153275489807, Accuracy = 0.904453456401825
    Iter #1370112:  Learning rate = 0.002941:   Batch Loss = 0.196892, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5564713478088379, Accuracy = 0.9024291634559631
    Iter #1370624:  Learning rate = 0.002941:   Batch Loss = 0.191536, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5603952407836914, Accuracy = 0.904453456401825
    Iter #1371136:  Learning rate = 0.002941:   Batch Loss = 0.189970, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5562616586685181, Accuracy = 0.9048582911491394
    Iter #1371648:  Learning rate = 0.002941:   Batch Loss = 0.187041, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.549070417881012, Accuracy = 0.9056680202484131
    Iter #1372160:  Learning rate = 0.002941:   Batch Loss = 0.187787, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5481671094894409, Accuracy = 0.9052631855010986
    Iter #1372672:  Learning rate = 0.002941:   Batch Loss = 0.191904, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5440032482147217, Accuracy = 0.9048582911491394
    Iter #1373184:  Learning rate = 0.002941:   Batch Loss = 0.182974, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5387227535247803, Accuracy = 0.9060728549957275
    Iter #1373696:  Learning rate = 0.002941:   Batch Loss = 0.183553, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5373745560646057, Accuracy = 0.9064777493476868
    Iter #1374208:  Learning rate = 0.002941:   Batch Loss = 0.184079, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5372223258018494, Accuracy = 0.9064777493476868
    Iter #1374720:  Learning rate = 0.002941:   Batch Loss = 0.181219, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5337694883346558, Accuracy = 0.9048582911491394
    Iter #1375232:  Learning rate = 0.002941:   Batch Loss = 0.183869, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5345035195350647, Accuracy = 0.9068825840950012
    Iter #1375744:  Learning rate = 0.002941:   Batch Loss = 0.179403, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5381744503974915, Accuracy = 0.9060728549957275
    Iter #1376256:  Learning rate = 0.002941:   Batch Loss = 0.183099, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5392485857009888, Accuracy = 0.9028339982032776
    Iter #1376768:  Learning rate = 0.002941:   Batch Loss = 0.177400, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5306085348129272, Accuracy = 0.9032388925552368
    Iter #1377280:  Learning rate = 0.002941:   Batch Loss = 0.180014, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5301622748374939, Accuracy = 0.9048582911491394
    Iter #1377792:  Learning rate = 0.002941:   Batch Loss = 0.177666, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5320287942886353, Accuracy = 0.9020242691040039
    Iter #1378304:  Learning rate = 0.002941:   Batch Loss = 0.177387, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5361120700836182, Accuracy = 0.9040485620498657
    Iter #1378816:  Learning rate = 0.002941:   Batch Loss = 0.175851, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5342187881469727, Accuracy = 0.9040485620498657
    Iter #1379328:  Learning rate = 0.002941:   Batch Loss = 0.175502, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5341364145278931, Accuracy = 0.9024291634559631
    Iter #1379840:  Learning rate = 0.002941:   Batch Loss = 0.173712, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5329599976539612, Accuracy = 0.9040485620498657
    Iter #1380352:  Learning rate = 0.002941:   Batch Loss = 0.176361, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5340539216995239, Accuracy = 0.904453456401825
    Iter #1380864:  Learning rate = 0.002941:   Batch Loss = 0.174764, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5322449803352356, Accuracy = 0.9028339982032776
    Iter #1381376:  Learning rate = 0.002941:   Batch Loss = 0.171745, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5361348390579224, Accuracy = 0.901214599609375
    Iter #1381888:  Learning rate = 0.002941:   Batch Loss = 0.173550, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5333656072616577, Accuracy = 0.9008097052574158
    Iter #1382400:  Learning rate = 0.002941:   Batch Loss = 0.170643, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5314322113990784, Accuracy = 0.9040485620498657
    Iter #1382912:  Learning rate = 0.002941:   Batch Loss = 0.170388, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5364227890968323, Accuracy = 0.9016194343566895
    Iter #1383424:  Learning rate = 0.002941:   Batch Loss = 0.167851, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5329803228378296, Accuracy = 0.9024291634559631
    Iter #1383936:  Learning rate = 0.002941:   Batch Loss = 0.172105, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5332599878311157, Accuracy = 0.901214599609375
    Iter #1384448:  Learning rate = 0.002941:   Batch Loss = 0.168989, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5356098413467407, Accuracy = 0.9024291634559631
    Iter #1384960:  Learning rate = 0.002941:   Batch Loss = 0.168276, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5388929843902588, Accuracy = 0.9008097052574158
    Iter #1385472:  Learning rate = 0.002941:   Batch Loss = 0.167204, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5356379747390747, Accuracy = 0.8991903066635132
    Iter #1385984:  Learning rate = 0.002941:   Batch Loss = 0.167433, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5332910418510437, Accuracy = 0.901214599609375
    Iter #1386496:  Learning rate = 0.002941:   Batch Loss = 0.163836, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5305197238922119, Accuracy = 0.9048582911491394
    Iter #1387008:  Learning rate = 0.002941:   Batch Loss = 0.167331, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.532545804977417, Accuracy = 0.9032388925552368
    Iter #1387520:  Learning rate = 0.002941:   Batch Loss = 0.170331, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5446138381958008, Accuracy = 0.8971660137176514
    Iter #1388032:  Learning rate = 0.002941:   Batch Loss = 0.165538, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5373380184173584, Accuracy = 0.9020242691040039
    Iter #1388544:  Learning rate = 0.002941:   Batch Loss = 0.161729, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5308234691619873, Accuracy = 0.9020242691040039
    Iter #1389056:  Learning rate = 0.002941:   Batch Loss = 0.164194, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5322262644767761, Accuracy = 0.8995951414108276
    Iter #1389568:  Learning rate = 0.002941:   Batch Loss = 0.161912, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5357734560966492, Accuracy = 0.8999999761581421
    Iter #1390080:  Learning rate = 0.002941:   Batch Loss = 0.162215, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5315059423446655, Accuracy = 0.8991903066635132
    Iter #1390592:  Learning rate = 0.002941:   Batch Loss = 0.166656, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5350834131240845, Accuracy = 0.898785412311554
    Iter #1391104:  Learning rate = 0.002941:   Batch Loss = 0.161526, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5465711951255798, Accuracy = 0.8979756832122803
    Iter #1391616:  Learning rate = 0.002941:   Batch Loss = 0.166096, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5391185879707336, Accuracy = 0.8991903066635132
    Iter #1392128:  Learning rate = 0.002941:   Batch Loss = 0.159440, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5319141149520874, Accuracy = 0.9020242691040039
    Iter #1392640:  Learning rate = 0.002941:   Batch Loss = 0.159101, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5305073261260986, Accuracy = 0.9004048705101013
    Iter #1393152:  Learning rate = 0.002941:   Batch Loss = 0.160445, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5340288877487183, Accuracy = 0.8975708484649658
    Iter #1393664:  Learning rate = 0.002941:   Batch Loss = 0.158757, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5367907881736755, Accuracy = 0.8963562846183777
    Iter #1394176:  Learning rate = 0.002941:   Batch Loss = 0.161437, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5299133658409119, Accuracy = 0.8975708484649658
    Iter #1394688:  Learning rate = 0.002941:   Batch Loss = 0.158286, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5308380126953125, Accuracy = 0.8991903066635132
    Iter #1395200:  Learning rate = 0.002941:   Batch Loss = 0.160529, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.538152813911438, Accuracy = 0.8967611193656921
    Iter #1395712:  Learning rate = 0.002941:   Batch Loss = 0.161052, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5374386310577393, Accuracy = 0.8971660137176514
    Iter #1396224:  Learning rate = 0.002941:   Batch Loss = 0.158877, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5430837869644165, Accuracy = 0.8935222625732422
    Iter #1396736:  Learning rate = 0.002941:   Batch Loss = 0.158113, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5335865020751953, Accuracy = 0.8947368264198303
    Iter #1397248:  Learning rate = 0.002941:   Batch Loss = 0.156445, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5286332368850708, Accuracy = 0.8999999761581421
    Iter #1397760:  Learning rate = 0.002941:   Batch Loss = 0.155563, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5282790660858154, Accuracy = 0.8995951414108276
    Iter #1398272:  Learning rate = 0.002941:   Batch Loss = 0.158629, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5363680124282837, Accuracy = 0.895546555519104
    Iter #1398784:  Learning rate = 0.002941:   Batch Loss = 0.155910, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5243638753890991, Accuracy = 0.8991903066635132
    Iter #1399296:  Learning rate = 0.002941:   Batch Loss = 0.157717, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5204179286956787, Accuracy = 0.901214599609375
    Iter #1399808:  Learning rate = 0.002941:   Batch Loss = 0.158326, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5340965390205383, Accuracy = 0.8983805775642395
    Iter #1400320:  Learning rate = 0.002823:   Batch Loss = 0.153827, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5316249132156372, Accuracy = 0.8963562846183777
    Iter #1400832:  Learning rate = 0.002823:   Batch Loss = 0.153195, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5360747575759888, Accuracy = 0.8898785710334778
    Iter #1401344:  Learning rate = 0.002823:   Batch Loss = 0.158006, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5569863319396973, Accuracy = 0.8902834057807922
    Iter #1401856:  Learning rate = 0.002823:   Batch Loss = 0.155911, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5463547110557556, Accuracy = 0.8939270973205566
    Iter #1402368:  Learning rate = 0.002823:   Batch Loss = 0.155061, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5345942974090576, Accuracy = 0.8939270973205566
    Iter #1402880:  Learning rate = 0.002823:   Batch Loss = 0.154887, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5450526475906372, Accuracy = 0.8935222625732422
    Iter #1403392:  Learning rate = 0.002823:   Batch Loss = 0.156705, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5408400893211365, Accuracy = 0.892307698726654
    Iter #1403904:  Learning rate = 0.002823:   Batch Loss = 0.152074, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5319089889526367, Accuracy = 0.8979756832122803
    Iter #1404416:  Learning rate = 0.002823:   Batch Loss = 0.157029, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5310215950012207, Accuracy = 0.898785412311554
    Iter #1404928:  Learning rate = 0.002823:   Batch Loss = 0.151748, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5330579876899719, Accuracy = 0.8959513902664185
    Iter #1405440:  Learning rate = 0.002823:   Batch Loss = 0.154664, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5309926271438599, Accuracy = 0.8967611193656921
    Iter #1405952:  Learning rate = 0.002823:   Batch Loss = 0.150777, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5316480398178101, Accuracy = 0.895546555519104
    Iter #1406464:  Learning rate = 0.002823:   Batch Loss = 0.155459, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5370802879333496, Accuracy = 0.8914979696273804
    Iter #1406976:  Learning rate = 0.002823:   Batch Loss = 0.154916, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5473286509513855, Accuracy = 0.8919028043746948
    Iter #1407488:  Learning rate = 0.002823:   Batch Loss = 0.153824, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.528831958770752, Accuracy = 0.8927125334739685
    Iter #1408000:  Learning rate = 0.002823:   Batch Loss = 0.149377, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5222873687744141, Accuracy = 0.8991903066635132
    Iter #1408512:  Learning rate = 0.002823:   Batch Loss = 0.149996, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5284239053726196, Accuracy = 0.8967611193656921
    Iter #1409024:  Learning rate = 0.002823:   Batch Loss = 0.152277, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5304766297340393, Accuracy = 0.8935222625732422
    Iter #1409536:  Learning rate = 0.002823:   Batch Loss = 0.150889, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5338401794433594, Accuracy = 0.8939270973205566
    Iter #1410048:  Learning rate = 0.002823:   Batch Loss = 0.149841, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5281320810317993, Accuracy = 0.8959513902664185
    Iter #1410560:  Learning rate = 0.002823:   Batch Loss = 0.149611, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5274972915649414, Accuracy = 0.8943319916725159
    Iter #1411072:  Learning rate = 0.002823:   Batch Loss = 0.149073, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.536237359046936, Accuracy = 0.892307698726654
    Iter #1411584:  Learning rate = 0.002823:   Batch Loss = 0.147714, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5305886268615723, Accuracy = 0.8939270973205566
    Iter #1412096:  Learning rate = 0.002823:   Batch Loss = 0.146788, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5308715105056763, Accuracy = 0.8927125334739685
    Iter #1412608:  Learning rate = 0.002823:   Batch Loss = 0.151486, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5304507613182068, Accuracy = 0.8971660137176514
    Iter #1413120:  Learning rate = 0.002823:   Batch Loss = 0.146571, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5325155854225159, Accuracy = 0.8935222625732422
    Iter #1413632:  Learning rate = 0.002823:   Batch Loss = 0.150018, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5334975123405457, Accuracy = 0.8919028043746948
    Iter #1414144:  Learning rate = 0.002823:   Batch Loss = 0.148218, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5278311967849731, Accuracy = 0.8935222625732422
    Iter #1414656:  Learning rate = 0.002823:   Batch Loss = 0.144874, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5321383476257324, Accuracy = 0.8931174278259277
    Iter #1415168:  Learning rate = 0.002823:   Batch Loss = 0.146484, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5364205837249756, Accuracy = 0.8910931348800659
    Iter #1415680:  Learning rate = 0.002823:   Batch Loss = 0.148214, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5329557657241821, Accuracy = 0.8906882405281067
    Iter #1416192:  Learning rate = 0.002823:   Batch Loss = 0.144066, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.52717125415802, Accuracy = 0.8983805775642395
    Iter #1416704:  Learning rate = 0.002823:   Batch Loss = 0.146478, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5321044921875, Accuracy = 0.8939270973205566
    Iter #1417216:  Learning rate = 0.002823:   Batch Loss = 0.143090, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5375428199768066, Accuracy = 0.8927125334739685
    Iter #1417728:  Learning rate = 0.002823:   Batch Loss = 0.141932, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5393406748771667, Accuracy = 0.8886639475822449
    Iter #1418240:  Learning rate = 0.002823:   Batch Loss = 0.146674, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5320414304733276, Accuracy = 0.892307698726654
    Iter #1418752:  Learning rate = 0.002823:   Batch Loss = 0.146350, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.537469744682312, Accuracy = 0.8939270973205566
    Iter #1419264:  Learning rate = 0.002823:   Batch Loss = 0.145921, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5480835437774658, Accuracy = 0.8890688419342041
    Iter #1419776:  Learning rate = 0.002823:   Batch Loss = 0.144095, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5511515736579895, Accuracy = 0.8894736766815186
    Iter #1420288:  Learning rate = 0.002823:   Batch Loss = 0.147743, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5425311923027039, Accuracy = 0.8906882405281067
    Iter #1420800:  Learning rate = 0.002823:   Batch Loss = 0.144175, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5421447157859802, Accuracy = 0.8882591128349304
    Iter #1421312:  Learning rate = 0.002823:   Batch Loss = 0.147598, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5452834367752075, Accuracy = 0.8902834057807922
    Iter #1421824:  Learning rate = 0.002823:   Batch Loss = 0.143592, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.544256865978241, Accuracy = 0.8914979696273804
    Iter #1422336:  Learning rate = 0.002823:   Batch Loss = 0.146227, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5506921410560608, Accuracy = 0.887854278087616
    Iter #1422848:  Learning rate = 0.002823:   Batch Loss = 0.142078, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5472114086151123, Accuracy = 0.8898785710334778
    Iter #1423360:  Learning rate = 0.002823:   Batch Loss = 0.139904, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5447322726249695, Accuracy = 0.8874493837356567
    Iter #1423872:  Learning rate = 0.002823:   Batch Loss = 0.153595, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5690767765045166, Accuracy = 0.882186233997345
    Iter #1424384:  Learning rate = 0.002823:   Batch Loss = 0.156670, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6461794376373291, Accuracy = 0.8619433045387268
    Iter #1424896:  Learning rate = 0.002823:   Batch Loss = 0.298786, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.712922215461731, Accuracy = 0.8408907055854797
    Iter #1425408:  Learning rate = 0.002823:   Batch Loss = 0.559114, Accuracy = 0.859375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7744022011756897, Accuracy = 0.8105263113975525
    Iter #1425920:  Learning rate = 0.002823:   Batch Loss = 0.308267, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7686072587966919, Accuracy = 0.8165991902351379
    Iter #1426432:  Learning rate = 0.002823:   Batch Loss = 0.433742, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7417769432067871, Accuracy = 0.8230769038200378
    Iter #1426944:  Learning rate = 0.002823:   Batch Loss = 0.631092, Accuracy = 0.859375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8163908123970032, Accuracy = 0.7975708246231079
    Iter #1427456:  Learning rate = 0.002823:   Batch Loss = 0.475816, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.878770112991333, Accuracy = 0.7919028401374817
    Iter #1427968:  Learning rate = 0.002823:   Batch Loss = 0.385643, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7281389236450195, Accuracy = 0.8372469544410706
    Iter #1428480:  Learning rate = 0.002823:   Batch Loss = 0.354268, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8459529280662537, Accuracy = 0.7995951175689697
    Iter #1428992:  Learning rate = 0.002823:   Batch Loss = 0.347979, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7604028582572937, Accuracy = 0.8259109258651733
    Iter #1429504:  Learning rate = 0.002823:   Batch Loss = 0.332598, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7559711933135986, Accuracy = 0.829959511756897
    Iter #1430016:  Learning rate = 0.002823:   Batch Loss = 0.366243, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7460558414459229, Accuracy = 0.8404858112335205
    Iter #1430528:  Learning rate = 0.002823:   Batch Loss = 0.344371, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.711473286151886, Accuracy = 0.8453441262245178
    Iter #1431040:  Learning rate = 0.002823:   Batch Loss = 0.363462, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7356319427490234, Accuracy = 0.8396761417388916
    Iter #1431552:  Learning rate = 0.002823:   Batch Loss = 0.336821, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.718136191368103, Accuracy = 0.8558704257011414
    Iter #1432064:  Learning rate = 0.002823:   Batch Loss = 0.245672, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7537521123886108, Accuracy = 0.8408907055854797
    Iter #1432576:  Learning rate = 0.002823:   Batch Loss = 0.268787, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7302356958389282, Accuracy = 0.8469635844230652
    Iter #1433088:  Learning rate = 0.002823:   Batch Loss = 0.361910, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7602075338363647, Accuracy = 0.8417003750801086
    Iter #1433600:  Learning rate = 0.002823:   Batch Loss = 0.374302, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6680827140808105, Accuracy = 0.8716599345207214
    Iter #1434112:  Learning rate = 0.002823:   Batch Loss = 0.300614, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6884473562240601, Accuracy = 0.856680154800415
    Iter #1434624:  Learning rate = 0.002823:   Batch Loss = 0.289548, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6955320835113525, Accuracy = 0.8582996129989624
    Iter #1435136:  Learning rate = 0.002823:   Batch Loss = 0.444974, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6918630599975586, Accuracy = 0.8643724918365479
    Iter #1435648:  Learning rate = 0.002823:   Batch Loss = 0.465330, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7225056886672974, Accuracy = 0.8453441262245178
    Iter #1436160:  Learning rate = 0.002823:   Batch Loss = 0.317798, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6437378525733948, Accuracy = 0.878947377204895
    Iter #1436672:  Learning rate = 0.002823:   Batch Loss = 0.228301, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6755592226982117, Accuracy = 0.8627530336380005
    Iter #1437184:  Learning rate = 0.002823:   Batch Loss = 0.302709, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6814475059509277, Accuracy = 0.8647773265838623
    Iter #1437696:  Learning rate = 0.002823:   Batch Loss = 0.258423, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6616567373275757, Accuracy = 0.8704453706741333
    Iter #1438208:  Learning rate = 0.002823:   Batch Loss = 0.272369, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7188733816146851, Accuracy = 0.8558704257011414
    Iter #1438720:  Learning rate = 0.002823:   Batch Loss = 0.252936, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6504015922546387, Accuracy = 0.8805667757987976
    Iter #1439232:  Learning rate = 0.002823:   Batch Loss = 0.223734, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6842031478881836, Accuracy = 0.8647773265838623
    Iter #1439744:  Learning rate = 0.002823:   Batch Loss = 0.236505, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6240605115890503, Accuracy = 0.8813765048980713
    Iter #1440256:  Learning rate = 0.002823:   Batch Loss = 0.232123, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6722085475921631, Accuracy = 0.8748987913131714
    Iter #1440768:  Learning rate = 0.002823:   Batch Loss = 0.261747, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.634712815284729, Accuracy = 0.8805667757987976
    Iter #1441280:  Learning rate = 0.002823:   Batch Loss = 0.228195, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6443107724189758, Accuracy = 0.8773279190063477
    Iter #1441792:  Learning rate = 0.002823:   Batch Loss = 0.200246, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6393442153930664, Accuracy = 0.8838056921958923
    Iter #1442304:  Learning rate = 0.002823:   Batch Loss = 0.249198, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6407867074012756, Accuracy = 0.8744939565658569
    Iter #1442816:  Learning rate = 0.002823:   Batch Loss = 0.204946, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6171656847000122, Accuracy = 0.8846153616905212
    Iter #1443328:  Learning rate = 0.002823:   Batch Loss = 0.202937, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6453661918640137, Accuracy = 0.8785424828529358
    Iter #1443840:  Learning rate = 0.002823:   Batch Loss = 0.198456, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6016381978988647, Accuracy = 0.8858299851417542
    Iter #1444352:  Learning rate = 0.002823:   Batch Loss = 0.239984, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5980891585350037, Accuracy = 0.8858299851417542
    Iter #1444864:  Learning rate = 0.002823:   Batch Loss = 0.201459, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5917885303497314, Accuracy = 0.8906882405281067
    Iter #1445376:  Learning rate = 0.002823:   Batch Loss = 0.201198, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5834656953811646, Accuracy = 0.895546555519104
    Iter #1445888:  Learning rate = 0.002823:   Batch Loss = 0.207396, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6167371869087219, Accuracy = 0.8846153616905212
    Iter #1446400:  Learning rate = 0.002823:   Batch Loss = 0.192517, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6233079433441162, Accuracy = 0.8829959630966187
    Iter #1446912:  Learning rate = 0.002823:   Batch Loss = 0.212934, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.602092444896698, Accuracy = 0.8919028043746948
    Iter #1447424:  Learning rate = 0.002823:   Batch Loss = 0.196543, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6021444201469421, Accuracy = 0.8882591128349304
    Iter #1447936:  Learning rate = 0.002823:   Batch Loss = 0.197439, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6037806272506714, Accuracy = 0.8858299851417542
    Iter #1448448:  Learning rate = 0.002823:   Batch Loss = 0.190031, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6004846692085266, Accuracy = 0.8858299851417542
    Iter #1448960:  Learning rate = 0.002823:   Batch Loss = 0.194470, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5943846106529236, Accuracy = 0.8874493837356567
    Iter #1449472:  Learning rate = 0.002823:   Batch Loss = 0.210013, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6319458484649658, Accuracy = 0.8838056921958923
    Iter #1449984:  Learning rate = 0.002823:   Batch Loss = 0.239400, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6114315986633301, Accuracy = 0.8866396546363831
    Iter #1450496:  Learning rate = 0.002823:   Batch Loss = 0.279288, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5915131568908691, Accuracy = 0.8931174278259277
    Iter #1451008:  Learning rate = 0.002823:   Batch Loss = 0.203703, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6246296167373657, Accuracy = 0.8769230842590332
    Iter #1451520:  Learning rate = 0.002823:   Batch Loss = 0.220862, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6083377599716187, Accuracy = 0.8854250907897949
    Iter #1452032:  Learning rate = 0.002823:   Batch Loss = 0.208149, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6475564241409302, Accuracy = 0.8716599345207214
    Iter #1452544:  Learning rate = 0.002823:   Batch Loss = 0.209908, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5927730798721313, Accuracy = 0.8894736766815186
    Iter #1453056:  Learning rate = 0.002823:   Batch Loss = 0.218837, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.685779869556427, Accuracy = 0.8615384697914124
    Iter #1453568:  Learning rate = 0.002823:   Batch Loss = 0.196455, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5967903733253479, Accuracy = 0.8902834057807922
    Iter #1454080:  Learning rate = 0.002823:   Batch Loss = 0.205258, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6207879781723022, Accuracy = 0.8809716701507568
    Iter #1454592:  Learning rate = 0.002823:   Batch Loss = 0.216229, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6063969135284424, Accuracy = 0.8829959630966187
    Iter #1455104:  Learning rate = 0.002823:   Batch Loss = 0.198532, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6081586480140686, Accuracy = 0.887854278087616
    Iter #1455616:  Learning rate = 0.002823:   Batch Loss = 0.292678, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6362907886505127, Accuracy = 0.8785424828529358
    Iter #1456128:  Learning rate = 0.002823:   Batch Loss = 0.236188, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6331490278244019, Accuracy = 0.8785424828529358
    Iter #1456640:  Learning rate = 0.002823:   Batch Loss = 0.225912, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6692698001861572, Accuracy = 0.8680161833763123
    Iter #1457152:  Learning rate = 0.002823:   Batch Loss = 0.261118, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6659568548202515, Accuracy = 0.8712550401687622
    Iter #1457664:  Learning rate = 0.002823:   Batch Loss = 0.217166, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6292412281036377, Accuracy = 0.8793522119522095
    Iter #1458176:  Learning rate = 0.002823:   Batch Loss = 0.235887, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6361896395683289, Accuracy = 0.8805667757987976
    Iter #1458688:  Learning rate = 0.002823:   Batch Loss = 0.196748, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6179236769676208, Accuracy = 0.8874493837356567
    Iter #1459200:  Learning rate = 0.002823:   Batch Loss = 0.211641, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6543411612510681, Accuracy = 0.8716599345207214
    Iter #1459712:  Learning rate = 0.002823:   Batch Loss = 0.215307, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6336030960083008, Accuracy = 0.8825910687446594
    Iter #1460224:  Learning rate = 0.002823:   Batch Loss = 0.223568, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6202394962310791, Accuracy = 0.8906882405281067
    Iter #1460736:  Learning rate = 0.002823:   Batch Loss = 0.195393, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5935832262039185, Accuracy = 0.8935222625732422
    Iter #1461248:  Learning rate = 0.002823:   Batch Loss = 0.195490, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5982730388641357, Accuracy = 0.8866396546363831
    Iter #1461760:  Learning rate = 0.002823:   Batch Loss = 0.192531, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5958421230316162, Accuracy = 0.8898785710334778
    Iter #1462272:  Learning rate = 0.002823:   Batch Loss = 0.188116, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5812937021255493, Accuracy = 0.8983805775642395
    Iter #1462784:  Learning rate = 0.002823:   Batch Loss = 0.220168, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5960195660591125, Accuracy = 0.8870445489883423
    Iter #1463296:  Learning rate = 0.002823:   Batch Loss = 0.188310, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5927383303642273, Accuracy = 0.8935222625732422
    Iter #1463808:  Learning rate = 0.002823:   Batch Loss = 0.189894, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5736845135688782, Accuracy = 0.8963562846183777
    Iter #1464320:  Learning rate = 0.002823:   Batch Loss = 0.186740, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5711572170257568, Accuracy = 0.8943319916725159
    Iter #1464832:  Learning rate = 0.002823:   Batch Loss = 0.180813, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5713346004486084, Accuracy = 0.9004048705101013
    Iter #1465344:  Learning rate = 0.002823:   Batch Loss = 0.186627, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5811314582824707, Accuracy = 0.8963562846183777
    Iter #1465856:  Learning rate = 0.002823:   Batch Loss = 0.180932, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5678302645683289, Accuracy = 0.8963562846183777
    Iter #1466368:  Learning rate = 0.002823:   Batch Loss = 0.181089, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5574113130569458, Accuracy = 0.8971660137176514
    Iter #1466880:  Learning rate = 0.002823:   Batch Loss = 0.182190, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5619086623191833, Accuracy = 0.8951417207717896
    Iter #1467392:  Learning rate = 0.002823:   Batch Loss = 0.176664, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5584437251091003, Accuracy = 0.8963562846183777
    Iter #1467904:  Learning rate = 0.002823:   Batch Loss = 0.174492, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5520462989807129, Accuracy = 0.9004048705101013
    Iter #1468416:  Learning rate = 0.002823:   Batch Loss = 0.174237, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5522322654724121, Accuracy = 0.8975708484649658
    Iter #1468928:  Learning rate = 0.002823:   Batch Loss = 0.173791, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5464738607406616, Accuracy = 0.901214599609375
    Iter #1469440:  Learning rate = 0.002823:   Batch Loss = 0.174517, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.543287456035614, Accuracy = 0.8995951414108276
    Iter #1469952:  Learning rate = 0.002823:   Batch Loss = 0.173960, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5412037968635559, Accuracy = 0.901214599609375
    Iter #1470464:  Learning rate = 0.002823:   Batch Loss = 0.171689, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5385741591453552, Accuracy = 0.9020242691040039
    Iter #1470976:  Learning rate = 0.002823:   Batch Loss = 0.172509, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5321394205093384, Accuracy = 0.9052631855010986
    Iter #1471488:  Learning rate = 0.002823:   Batch Loss = 0.173413, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5338690876960754, Accuracy = 0.904453456401825
    Iter #1472000:  Learning rate = 0.002823:   Batch Loss = 0.169613, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5385462641716003, Accuracy = 0.8995951414108276
    Iter #1472512:  Learning rate = 0.002823:   Batch Loss = 0.171501, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5380840301513672, Accuracy = 0.9008097052574158
    Iter #1473024:  Learning rate = 0.002823:   Batch Loss = 0.171556, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5356508493423462, Accuracy = 0.9008097052574158
    Iter #1473536:  Learning rate = 0.002823:   Batch Loss = 0.167095, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5332801938056946, Accuracy = 0.9020242691040039
    Iter #1474048:  Learning rate = 0.002823:   Batch Loss = 0.167057, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5321460962295532, Accuracy = 0.901214599609375
    Iter #1474560:  Learning rate = 0.002823:   Batch Loss = 0.167837, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5301987528800964, Accuracy = 0.9016194343566895
    Iter #1475072:  Learning rate = 0.002823:   Batch Loss = 0.165874, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5299999713897705, Accuracy = 0.9004048705101013
    Iter #1475584:  Learning rate = 0.002823:   Batch Loss = 0.166270, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5343712568283081, Accuracy = 0.8995951414108276
    Iter #1476096:  Learning rate = 0.002823:   Batch Loss = 0.170232, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5381495356559753, Accuracy = 0.898785412311554
    Iter #1476608:  Learning rate = 0.002823:   Batch Loss = 0.164747, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5351382493972778, Accuracy = 0.9004048705101013
    Iter #1477120:  Learning rate = 0.002823:   Batch Loss = 0.162145, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5317109823226929, Accuracy = 0.9024291634559631
    Iter #1477632:  Learning rate = 0.002823:   Batch Loss = 0.162578, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5277710556983948, Accuracy = 0.9028339982032776
    Iter #1478144:  Learning rate = 0.002823:   Batch Loss = 0.161987, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5269457101821899, Accuracy = 0.904453456401825
    Iter #1478656:  Learning rate = 0.002823:   Batch Loss = 0.164687, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5277574062347412, Accuracy = 0.9020242691040039
    Iter #1479168:  Learning rate = 0.002823:   Batch Loss = 0.161448, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5298511981964111, Accuracy = 0.901214599609375
    Iter #1479680:  Learning rate = 0.002823:   Batch Loss = 0.159955, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5291206240653992, Accuracy = 0.9016194343566895
    Iter #1480192:  Learning rate = 0.002823:   Batch Loss = 0.161345, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.525159478187561, Accuracy = 0.901214599609375
    Iter #1480704:  Learning rate = 0.002823:   Batch Loss = 0.162627, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5261914730072021, Accuracy = 0.9016194343566895
    Iter #1481216:  Learning rate = 0.002823:   Batch Loss = 0.159861, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.527470588684082, Accuracy = 0.9020242691040039
    Iter #1481728:  Learning rate = 0.002823:   Batch Loss = 0.160364, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5325273871421814, Accuracy = 0.9016194343566895
    Iter #1482240:  Learning rate = 0.002823:   Batch Loss = 0.158373, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5330943465232849, Accuracy = 0.9004048705101013
    Iter #1482752:  Learning rate = 0.002823:   Batch Loss = 0.160340, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5300403833389282, Accuracy = 0.9028339982032776
    Iter #1483264:  Learning rate = 0.002823:   Batch Loss = 0.157595, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5286122560501099, Accuracy = 0.9020242691040039
    Iter #1483776:  Learning rate = 0.002823:   Batch Loss = 0.157410, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5249142646789551, Accuracy = 0.9028339982032776
    Iter #1484288:  Learning rate = 0.002823:   Batch Loss = 0.158565, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5258466005325317, Accuracy = 0.9036437273025513
    Iter #1484800:  Learning rate = 0.002823:   Batch Loss = 0.156420, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5246022939682007, Accuracy = 0.9032388925552368
    Iter #1485312:  Learning rate = 0.002823:   Batch Loss = 0.157762, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5268042683601379, Accuracy = 0.9020242691040039
    Iter #1485824:  Learning rate = 0.002823:   Batch Loss = 0.156035, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5262497663497925, Accuracy = 0.901214599609375
    Iter #1486336:  Learning rate = 0.002823:   Batch Loss = 0.158926, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5293998718261719, Accuracy = 0.8999999761581421
    Iter #1486848:  Learning rate = 0.002823:   Batch Loss = 0.154632, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5273265838623047, Accuracy = 0.9016194343566895
    Iter #1487360:  Learning rate = 0.002823:   Batch Loss = 0.156660, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5287541151046753, Accuracy = 0.8959513902664185
    Iter #1487872:  Learning rate = 0.002823:   Batch Loss = 0.155912, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5254392623901367, Accuracy = 0.8991903066635132
    Iter #1488384:  Learning rate = 0.002823:   Batch Loss = 0.154761, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5255810022354126, Accuracy = 0.9004048705101013
    Iter #1488896:  Learning rate = 0.002823:   Batch Loss = 0.152054, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5296934843063354, Accuracy = 0.8991903066635132
    Iter #1489408:  Learning rate = 0.002823:   Batch Loss = 0.155860, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5307921171188354, Accuracy = 0.8991903066635132
    Iter #1489920:  Learning rate = 0.002823:   Batch Loss = 0.156438, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5311877727508545, Accuracy = 0.9008097052574158
    Iter #1490432:  Learning rate = 0.002823:   Batch Loss = 0.160946, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5309802293777466, Accuracy = 0.9008097052574158
    Iter #1490944:  Learning rate = 0.002823:   Batch Loss = 0.154394, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.535850465297699, Accuracy = 0.8995951414108276
    Iter #1491456:  Learning rate = 0.002823:   Batch Loss = 0.153084, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.539665937423706, Accuracy = 0.9004048705101013
    Iter #1491968:  Learning rate = 0.002823:   Batch Loss = 0.153112, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5368240475654602, Accuracy = 0.9004048705101013
    Iter #1492480:  Learning rate = 0.002823:   Batch Loss = 0.153447, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5351388454437256, Accuracy = 0.8999999761581421
    Iter #1492992:  Learning rate = 0.002823:   Batch Loss = 0.149626, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5287883281707764, Accuracy = 0.898785412311554
    Iter #1493504:  Learning rate = 0.002823:   Batch Loss = 0.154235, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5282115936279297, Accuracy = 0.8963562846183777
    Iter #1494016:  Learning rate = 0.002823:   Batch Loss = 0.149526, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5297520160675049, Accuracy = 0.8983805775642395
    Iter #1494528:  Learning rate = 0.002823:   Batch Loss = 0.151557, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5356293320655823, Accuracy = 0.8999999761581421
    Iter #1495040:  Learning rate = 0.002823:   Batch Loss = 0.150724, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.536094069480896, Accuracy = 0.8963562846183777
    Iter #1495552:  Learning rate = 0.002823:   Batch Loss = 0.148680, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.532827615737915, Accuracy = 0.895546555519104
    Iter #1496064:  Learning rate = 0.002823:   Batch Loss = 0.150269, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5310938358306885, Accuracy = 0.8975708484649658
    Iter #1496576:  Learning rate = 0.002823:   Batch Loss = 0.146604, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5346301794052124, Accuracy = 0.8947368264198303
    Iter #1497088:  Learning rate = 0.002823:   Batch Loss = 0.148951, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5401178002357483, Accuracy = 0.8951417207717896
    Iter #1497600:  Learning rate = 0.002823:   Batch Loss = 0.151851, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5352562069892883, Accuracy = 0.895546555519104
    Iter #1498112:  Learning rate = 0.002823:   Batch Loss = 0.147479, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5283239483833313, Accuracy = 0.8959513902664185
    Iter #1498624:  Learning rate = 0.002823:   Batch Loss = 0.147823, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5290542244911194, Accuracy = 0.8967611193656921
    Iter #1499136:  Learning rate = 0.002823:   Batch Loss = 0.147450, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5249220132827759, Accuracy = 0.8979756832122803
    Iter #1499648:  Learning rate = 0.002823:   Batch Loss = 0.146848, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.531653642654419, Accuracy = 0.8951417207717896
    Iter #1500160:  Learning rate = 0.002710:   Batch Loss = 0.145054, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5335261821746826, Accuracy = 0.8971660137176514
    Iter #1500672:  Learning rate = 0.002710:   Batch Loss = 0.147499, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5344711542129517, Accuracy = 0.8951417207717896
    Iter #1501184:  Learning rate = 0.002710:   Batch Loss = 0.145895, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5388614535331726, Accuracy = 0.8939270973205566
    Iter #1501696:  Learning rate = 0.002710:   Batch Loss = 0.147512, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5310261249542236, Accuracy = 0.8967611193656921
    Iter #1502208:  Learning rate = 0.002710:   Batch Loss = 0.146411, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5288068652153015, Accuracy = 0.8959513902664185
    Iter #1502720:  Learning rate = 0.002710:   Batch Loss = 0.148726, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5313390493392944, Accuracy = 0.8947368264198303
    Iter #1503232:  Learning rate = 0.002710:   Batch Loss = 0.145602, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5286583304405212, Accuracy = 0.8975708484649658
    Iter #1503744:  Learning rate = 0.002710:   Batch Loss = 0.144935, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.533558189868927, Accuracy = 0.8959513902664185
    Iter #1504256:  Learning rate = 0.002710:   Batch Loss = 0.147885, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5347236394882202, Accuracy = 0.8939270973205566
    Iter #1504768:  Learning rate = 0.002710:   Batch Loss = 0.142534, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.534818708896637, Accuracy = 0.8967611193656921
    Iter #1505280:  Learning rate = 0.002710:   Batch Loss = 0.145581, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5355667471885681, Accuracy = 0.8947368264198303
    Iter #1505792:  Learning rate = 0.002710:   Batch Loss = 0.141316, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5320848822593689, Accuracy = 0.8967611193656921
    Iter #1506304:  Learning rate = 0.002710:   Batch Loss = 0.146739, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.536494255065918, Accuracy = 0.8999999761581421
    Iter #1506816:  Learning rate = 0.002710:   Batch Loss = 0.144790, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5444199442863464, Accuracy = 0.8854250907897949
    Iter #1507328:  Learning rate = 0.002710:   Batch Loss = 0.160366, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5224035978317261, Accuracy = 0.9016194343566895
    Iter #1507840:  Learning rate = 0.002710:   Batch Loss = 0.276166, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5392075181007385, Accuracy = 0.8991903066635132
    Iter #1508352:  Learning rate = 0.002710:   Batch Loss = 0.174778, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5912177562713623, Accuracy = 0.8684210777282715
    Iter #1508864:  Learning rate = 0.002710:   Batch Loss = 0.183975, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6052823662757874, Accuracy = 0.8663967847824097
    Iter #1509376:  Learning rate = 0.002710:   Batch Loss = 0.272326, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7003044486045837, Accuracy = 0.8546558618545532
    Iter #1509888:  Learning rate = 0.002710:   Batch Loss = 0.314944, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7696939706802368, Accuracy = 0.8336032629013062
    Iter #1510400:  Learning rate = 0.002710:   Batch Loss = 0.350313, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6930027008056641, Accuracy = 0.8453441262245178
    Iter #1510912:  Learning rate = 0.002710:   Batch Loss = 0.373506, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7755762338638306, Accuracy = 0.8190283179283142
    Iter #1511424:  Learning rate = 0.002710:   Batch Loss = 0.523411, Accuracy = 0.84375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7444357872009277, Accuracy = 0.8283400535583496
    Iter #1511936:  Learning rate = 0.002710:   Batch Loss = 0.410336, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7407511472702026, Accuracy = 0.8311740756034851
    Iter #1512448:  Learning rate = 0.002710:   Batch Loss = 0.469310, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8329834938049316, Accuracy = 0.7975708246231079
    Iter #1512960:  Learning rate = 0.002710:   Batch Loss = 0.329481, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7818492650985718, Accuracy = 0.8174089193344116
    Iter #1513472:  Learning rate = 0.002710:   Batch Loss = 0.498376, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8037692904472351, Accuracy = 0.821052610874176
    Iter #1513984:  Learning rate = 0.002710:   Batch Loss = 0.388121, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8756270408630371, Accuracy = 0.7894737124443054
    Iter #1514496:  Learning rate = 0.002710:   Batch Loss = 0.370566, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7714332342147827, Accuracy = 0.8246963620185852
    Iter #1515008:  Learning rate = 0.002710:   Batch Loss = 0.426676, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7526382803916931, Accuracy = 0.8352226614952087
    Iter #1515520:  Learning rate = 0.002710:   Batch Loss = 0.369217, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7300532460212708, Accuracy = 0.8380566835403442
    Iter #1516032:  Learning rate = 0.002710:   Batch Loss = 0.229696, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7255585789680481, Accuracy = 0.8412955403327942
    Iter #1516544:  Learning rate = 0.002710:   Batch Loss = 0.236296, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6763137578964233, Accuracy = 0.8672064542770386
    Iter #1517056:  Learning rate = 0.002710:   Batch Loss = 0.214932, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.649570107460022, Accuracy = 0.8684210777282715
    Iter #1517568:  Learning rate = 0.002710:   Batch Loss = 0.358995, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6613503098487854, Accuracy = 0.8676113486289978
    Iter #1518080:  Learning rate = 0.002710:   Batch Loss = 0.240113, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.665074348449707, Accuracy = 0.865587055683136
    Iter #1518592:  Learning rate = 0.002710:   Batch Loss = 0.241545, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6451910734176636, Accuracy = 0.8696356415748596
    Iter #1519104:  Learning rate = 0.002710:   Batch Loss = 0.262435, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6599570512771606, Accuracy = 0.8647773265838623
    Iter #1519616:  Learning rate = 0.002710:   Batch Loss = 0.333975, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6402913331985474, Accuracy = 0.8744939565658569
    Iter #1520128:  Learning rate = 0.002710:   Batch Loss = 0.245114, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7088180780410767, Accuracy = 0.8534412980079651
    Iter #1520640:  Learning rate = 0.002710:   Batch Loss = 0.263546, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6588492393493652, Accuracy = 0.8748987913131714
    Iter #1521152:  Learning rate = 0.002710:   Batch Loss = 0.203999, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5995429158210754, Accuracy = 0.887854278087616
    Iter #1521664:  Learning rate = 0.002710:   Batch Loss = 0.219597, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6774568557739258, Accuracy = 0.8720647692680359
    Iter #1522176:  Learning rate = 0.002710:   Batch Loss = 0.324349, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.678481936454773, Accuracy = 0.8672064542770386
    Iter #1522688:  Learning rate = 0.002710:   Batch Loss = 0.346321, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6454439759254456, Accuracy = 0.8753036260604858
    Iter #1523200:  Learning rate = 0.002710:   Batch Loss = 0.220734, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.670390784740448, Accuracy = 0.8748987913131714
    Iter #1523712:  Learning rate = 0.002710:   Batch Loss = 0.234919, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6951212286949158, Accuracy = 0.8663967847824097
    Iter #1524224:  Learning rate = 0.002710:   Batch Loss = 0.242613, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7068308591842651, Accuracy = 0.8619433045387268
    Iter #1524736:  Learning rate = 0.002710:   Batch Loss = 0.226322, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6128890514373779, Accuracy = 0.8842105269432068
    Iter #1525248:  Learning rate = 0.002710:   Batch Loss = 0.212235, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5903528928756714, Accuracy = 0.8931174278259277
    Iter #1525760:  Learning rate = 0.002710:   Batch Loss = 0.198523, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5957733988761902, Accuracy = 0.8866396546363831
    Iter #1526272:  Learning rate = 0.002710:   Batch Loss = 0.191257, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5798315405845642, Accuracy = 0.892307698726654
    Iter #1526784:  Learning rate = 0.002710:   Batch Loss = 0.194500, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5540522336959839, Accuracy = 0.8951417207717896
    Iter #1527296:  Learning rate = 0.002710:   Batch Loss = 0.205619, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5556042194366455, Accuracy = 0.8995951414108276
    Iter #1527808:  Learning rate = 0.002710:   Batch Loss = 0.189660, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5395380258560181, Accuracy = 0.9004048705101013
    Iter #1528320:  Learning rate = 0.002710:   Batch Loss = 0.186207, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5359069108963013, Accuracy = 0.9040485620498657
    Iter #1528832:  Learning rate = 0.002710:   Batch Loss = 0.185111, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5341111421585083, Accuracy = 0.9020242691040039
    Iter #1529344:  Learning rate = 0.002710:   Batch Loss = 0.182651, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.536981463432312, Accuracy = 0.898785412311554
    Iter #1529856:  Learning rate = 0.002710:   Batch Loss = 0.182421, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5370600819587708, Accuracy = 0.9016194343566895
    Iter #1530368:  Learning rate = 0.002710:   Batch Loss = 0.179813, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5303965210914612, Accuracy = 0.904453456401825
    Iter #1530880:  Learning rate = 0.002710:   Batch Loss = 0.177067, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5280188322067261, Accuracy = 0.9040485620498657
    Iter #1531392:  Learning rate = 0.002710:   Batch Loss = 0.174790, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5285724401473999, Accuracy = 0.9024291634559631
    Iter #1531904:  Learning rate = 0.002710:   Batch Loss = 0.173749, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5268593430519104, Accuracy = 0.9020242691040039
    Iter #1532416:  Learning rate = 0.002710:   Batch Loss = 0.174870, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5197880864143372, Accuracy = 0.9020242691040039
    Iter #1532928:  Learning rate = 0.002710:   Batch Loss = 0.175567, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5177140831947327, Accuracy = 0.904453456401825
    Iter #1533440:  Learning rate = 0.002710:   Batch Loss = 0.173834, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5156664252281189, Accuracy = 0.9040485620498657
    Iter #1533952:  Learning rate = 0.002710:   Batch Loss = 0.169040, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5149466395378113, Accuracy = 0.9048582911491394
    Iter #1534464:  Learning rate = 0.002710:   Batch Loss = 0.170069, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5155047178268433, Accuracy = 0.9040485620498657
    Iter #1534976:  Learning rate = 0.002710:   Batch Loss = 0.167714, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5143930912017822, Accuracy = 0.9036437273025513
    Iter #1535488:  Learning rate = 0.002710:   Batch Loss = 0.169377, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5138369798660278, Accuracy = 0.9036437273025513
    Iter #1536000:  Learning rate = 0.002710:   Batch Loss = 0.166765, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5152287483215332, Accuracy = 0.9028339982032776
    Iter #1536512:  Learning rate = 0.002710:   Batch Loss = 0.171678, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5201129913330078, Accuracy = 0.9004048705101013
    Iter #1537024:  Learning rate = 0.002710:   Batch Loss = 0.169227, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5192615985870361, Accuracy = 0.9028339982032776
    Iter #1537536:  Learning rate = 0.002710:   Batch Loss = 0.178483, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.534321129322052, Accuracy = 0.898785412311554
    Iter #1538048:  Learning rate = 0.002710:   Batch Loss = 0.182338, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5793889760971069, Accuracy = 0.8842105269432068
    Iter #1538560:  Learning rate = 0.002710:   Batch Loss = 0.251079, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5971038937568665, Accuracy = 0.8834007978439331
    Iter #1539072:  Learning rate = 0.002710:   Batch Loss = 0.237244, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6410263180732727, Accuracy = 0.8672064542770386
    Iter #1539584:  Learning rate = 0.002710:   Batch Loss = 0.194758, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6512004137039185, Accuracy = 0.8635627627372742
    Iter #1540096:  Learning rate = 0.002710:   Batch Loss = 0.210983, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6165286302566528, Accuracy = 0.8753036260604858
    Iter #1540608:  Learning rate = 0.002710:   Batch Loss = 0.243317, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6373670101165771, Accuracy = 0.8639675974845886
    Iter #1541120:  Learning rate = 0.002710:   Batch Loss = 0.240345, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6381574869155884, Accuracy = 0.873279333114624
    Iter #1541632:  Learning rate = 0.002710:   Batch Loss = 0.314229, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6822150349617004, Accuracy = 0.8510121703147888
    Iter #1542144:  Learning rate = 0.002710:   Batch Loss = 0.249233, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.659223735332489, Accuracy = 0.8607287406921387
    Iter #1542656:  Learning rate = 0.002710:   Batch Loss = 0.269073, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7043250799179077, Accuracy = 0.8453441262245178
    Iter #1543168:  Learning rate = 0.002710:   Batch Loss = 0.219025, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.612696647644043, Accuracy = 0.8781376481056213
    Iter #1543680:  Learning rate = 0.002710:   Batch Loss = 0.218031, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6335881948471069, Accuracy = 0.8736842274665833
    Iter #1544192:  Learning rate = 0.002710:   Batch Loss = 0.201320, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6373781561851501, Accuracy = 0.8748987913131714
    Iter #1544704:  Learning rate = 0.002710:   Batch Loss = 0.257199, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6869775056838989, Accuracy = 0.8603239059448242
    Iter #1545216:  Learning rate = 0.002710:   Batch Loss = 0.223067, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6300111413002014, Accuracy = 0.8781376481056213
    Iter #1545728:  Learning rate = 0.002710:   Batch Loss = 0.242575, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6413220167160034, Accuracy = 0.8692307472229004
    Iter #1546240:  Learning rate = 0.002710:   Batch Loss = 0.241952, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7156924605369568, Accuracy = 0.848987877368927
    Iter #1546752:  Learning rate = 0.002710:   Batch Loss = 0.225347, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6191867589950562, Accuracy = 0.8842105269432068
    Iter #1547264:  Learning rate = 0.002710:   Batch Loss = 0.274817, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5963916778564453, Accuracy = 0.8850202560424805
    Iter #1547776:  Learning rate = 0.002710:   Batch Loss = 0.195388, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5613529682159424, Accuracy = 0.892307698726654
    Iter #1548288:  Learning rate = 0.002710:   Batch Loss = 0.203381, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5598272681236267, Accuracy = 0.8931174278259277
    Iter #1548800:  Learning rate = 0.002710:   Batch Loss = 0.196215, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5766087174415588, Accuracy = 0.8894736766815186
    Iter #1549312:  Learning rate = 0.002710:   Batch Loss = 0.212225, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5716221928596497, Accuracy = 0.8951417207717896
    Iter #1549824:  Learning rate = 0.002710:   Batch Loss = 0.201533, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6365224719047546, Accuracy = 0.8720647692680359
    Iter #1550336:  Learning rate = 0.002710:   Batch Loss = 0.208914, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5993419885635376, Accuracy = 0.8890688419342041
    Iter #1550848:  Learning rate = 0.002710:   Batch Loss = 0.189842, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6191576719284058, Accuracy = 0.8753036260604858
    Iter #1551360:  Learning rate = 0.002710:   Batch Loss = 0.243083, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6203094720840454, Accuracy = 0.878947377204895
    Iter #1551872:  Learning rate = 0.002710:   Batch Loss = 0.193099, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5778928399085999, Accuracy = 0.8854250907897949
    Iter #1552384:  Learning rate = 0.002710:   Batch Loss = 0.212659, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.653995931148529, Accuracy = 0.8720647692680359
    Iter #1552896:  Learning rate = 0.002710:   Batch Loss = 0.252433, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6039458513259888, Accuracy = 0.878947377204895
    Iter #1553408:  Learning rate = 0.002710:   Batch Loss = 0.224945, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5738819241523743, Accuracy = 0.8890688419342041
    Iter #1553920:  Learning rate = 0.002710:   Batch Loss = 0.186898, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5626178979873657, Accuracy = 0.8959513902664185
    Iter #1554432:  Learning rate = 0.002710:   Batch Loss = 0.192571, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5485491752624512, Accuracy = 0.8995951414108276
    Iter #1554944:  Learning rate = 0.002710:   Batch Loss = 0.182333, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5571897029876709, Accuracy = 0.9016194343566895
    Iter #1555456:  Learning rate = 0.002710:   Batch Loss = 0.176574, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.556239128112793, Accuracy = 0.9028339982032776
    Iter #1555968:  Learning rate = 0.002710:   Batch Loss = 0.179252, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5405062437057495, Accuracy = 0.9068825840950012
    Iter #1556480:  Learning rate = 0.002710:   Batch Loss = 0.177954, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5476335287094116, Accuracy = 0.9032388925552368
    Iter #1556992:  Learning rate = 0.002710:   Batch Loss = 0.175364, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5420733690261841, Accuracy = 0.9020242691040039
    Iter #1557504:  Learning rate = 0.002710:   Batch Loss = 0.180251, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.560573935508728, Accuracy = 0.8951417207717896
    Iter #1558016:  Learning rate = 0.002710:   Batch Loss = 0.180434, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5220782160758972, Accuracy = 0.9064777493476868
    Iter #1558528:  Learning rate = 0.002710:   Batch Loss = 0.173221, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.527990996837616, Accuracy = 0.9060728549957275
    Iter #1559040:  Learning rate = 0.002710:   Batch Loss = 0.173024, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5322085022926331, Accuracy = 0.9040485620498657
    Iter #1559552:  Learning rate = 0.002710:   Batch Loss = 0.170037, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5264925956726074, Accuracy = 0.9008097052574158
    Iter #1560064:  Learning rate = 0.002710:   Batch Loss = 0.168905, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5233213901519775, Accuracy = 0.9016194343566895
    Iter #1560576:  Learning rate = 0.002710:   Batch Loss = 0.172357, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5255534648895264, Accuracy = 0.9024291634559631
    Iter #1561088:  Learning rate = 0.002710:   Batch Loss = 0.167673, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5279912948608398, Accuracy = 0.9020242691040039
    Iter #1561600:  Learning rate = 0.002710:   Batch Loss = 0.167174, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5217485427856445, Accuracy = 0.904453456401825
    Iter #1562112:  Learning rate = 0.002710:   Batch Loss = 0.167224, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5164802074432373, Accuracy = 0.9040485620498657
    Iter #1562624:  Learning rate = 0.002710:   Batch Loss = 0.166424, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5158967971801758, Accuracy = 0.904453456401825
    Iter #1563136:  Learning rate = 0.002710:   Batch Loss = 0.163502, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5163357853889465, Accuracy = 0.901214599609375
    Iter #1563648:  Learning rate = 0.002710:   Batch Loss = 0.162265, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5147265195846558, Accuracy = 0.9004048705101013
    Iter #1564160:  Learning rate = 0.002710:   Batch Loss = 0.164598, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5131098628044128, Accuracy = 0.9032388925552368
    Iter #1564672:  Learning rate = 0.002710:   Batch Loss = 0.163197, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5132369995117188, Accuracy = 0.9028339982032776
    Iter #1565184:  Learning rate = 0.002710:   Batch Loss = 0.162757, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5157058238983154, Accuracy = 0.901214599609375
    Iter #1565696:  Learning rate = 0.002710:   Batch Loss = 0.164121, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5158933997154236, Accuracy = 0.9004048705101013
    Iter #1566208:  Learning rate = 0.002710:   Batch Loss = 0.160584, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5136851072311401, Accuracy = 0.901214599609375
    Iter #1566720:  Learning rate = 0.002710:   Batch Loss = 0.163830, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5092194080352783, Accuracy = 0.9028339982032776
    Iter #1567232:  Learning rate = 0.002710:   Batch Loss = 0.160436, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5084584355354309, Accuracy = 0.9032388925552368
    Iter #1567744:  Learning rate = 0.002710:   Batch Loss = 0.161270, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5117299556732178, Accuracy = 0.9016194343566895
    Iter #1568256:  Learning rate = 0.002710:   Batch Loss = 0.158949, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5135972499847412, Accuracy = 0.9020242691040039
    Iter #1568768:  Learning rate = 0.002710:   Batch Loss = 0.160221, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5145590901374817, Accuracy = 0.9008097052574158
    Iter #1569280:  Learning rate = 0.002710:   Batch Loss = 0.159400, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5177468061447144, Accuracy = 0.8995951414108276
    Iter #1569792:  Learning rate = 0.002710:   Batch Loss = 0.159922, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5154469609260559, Accuracy = 0.8999999761581421
    Iter #1570304:  Learning rate = 0.002710:   Batch Loss = 0.156714, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5116995573043823, Accuracy = 0.901214599609375
    Iter #1570816:  Learning rate = 0.002710:   Batch Loss = 0.156587, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5105536580085754, Accuracy = 0.9016194343566895
    Iter #1571328:  Learning rate = 0.002710:   Batch Loss = 0.154464, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5107794404029846, Accuracy = 0.9040485620498657
    Iter #1571840:  Learning rate = 0.002710:   Batch Loss = 0.156350, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.51200932264328, Accuracy = 0.9004048705101013
    Iter #1572352:  Learning rate = 0.002710:   Batch Loss = 0.156908, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5124272108078003, Accuracy = 0.9008097052574158
    Iter #1572864:  Learning rate = 0.002710:   Batch Loss = 0.155133, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5112489461898804, Accuracy = 0.9024291634559631
    Iter #1573376:  Learning rate = 0.002710:   Batch Loss = 0.154441, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5141767263412476, Accuracy = 0.9016194343566895
    Iter #1573888:  Learning rate = 0.002710:   Batch Loss = 0.154584, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.515791654586792, Accuracy = 0.9008097052574158
    Iter #1574400:  Learning rate = 0.002710:   Batch Loss = 0.154164, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5130476951599121, Accuracy = 0.8999999761581421
    Iter #1574912:  Learning rate = 0.002710:   Batch Loss = 0.153392, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.510657548904419, Accuracy = 0.9004048705101013
    Iter #1575424:  Learning rate = 0.002710:   Batch Loss = 0.155714, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5115090608596802, Accuracy = 0.9004048705101013
    Iter #1575936:  Learning rate = 0.002710:   Batch Loss = 0.153954, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5126776695251465, Accuracy = 0.9016194343566895
    Iter #1576448:  Learning rate = 0.002710:   Batch Loss = 0.150793, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5110538601875305, Accuracy = 0.9032388925552368
    Iter #1576960:  Learning rate = 0.002710:   Batch Loss = 0.149857, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5108273029327393, Accuracy = 0.9016194343566895
    Iter #1577472:  Learning rate = 0.002710:   Batch Loss = 0.151920, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5085103511810303, Accuracy = 0.9016194343566895
    Iter #1577984:  Learning rate = 0.002710:   Batch Loss = 0.152336, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5108893513679504, Accuracy = 0.9016194343566895
    Iter #1578496:  Learning rate = 0.002710:   Batch Loss = 0.151367, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5113778710365295, Accuracy = 0.9008097052574158
    Iter #1579008:  Learning rate = 0.002710:   Batch Loss = 0.151500, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5087475776672363, Accuracy = 0.9020242691040039
    Iter #1579520:  Learning rate = 0.002710:   Batch Loss = 0.150779, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5074544548988342, Accuracy = 0.9028339982032776
    Iter #1580032:  Learning rate = 0.002710:   Batch Loss = 0.149812, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5053924322128296, Accuracy = 0.9048582911491394
    Iter #1580544:  Learning rate = 0.002710:   Batch Loss = 0.149491, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5073176026344299, Accuracy = 0.9040485620498657
    Iter #1581056:  Learning rate = 0.002710:   Batch Loss = 0.148709, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5120365023612976, Accuracy = 0.9028339982032776
    Iter #1581568:  Learning rate = 0.002710:   Batch Loss = 0.153215, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5251397490501404, Accuracy = 0.9008097052574158
    Iter #1582080:  Learning rate = 0.002710:   Batch Loss = 0.148381, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5191598534584045, Accuracy = 0.9040485620498657
    Iter #1582592:  Learning rate = 0.002710:   Batch Loss = 0.150380, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5076305270195007, Accuracy = 0.9064777493476868
    Iter #1583104:  Learning rate = 0.002710:   Batch Loss = 0.146009, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5047667622566223, Accuracy = 0.9060728549957275
    Iter #1583616:  Learning rate = 0.002710:   Batch Loss = 0.147666, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5072516798973083, Accuracy = 0.9056680202484131
    Iter #1584128:  Learning rate = 0.002710:   Batch Loss = 0.145524, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5124069452285767, Accuracy = 0.9036437273025513
    Iter #1584640:  Learning rate = 0.002710:   Batch Loss = 0.146005, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.512818455696106, Accuracy = 0.9008097052574158
    Iter #1585152:  Learning rate = 0.002710:   Batch Loss = 0.146265, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5163364410400391, Accuracy = 0.898785412311554
    Iter #1585664:  Learning rate = 0.002710:   Batch Loss = 0.145198, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5164560079574585, Accuracy = 0.8975708484649658
    Iter #1586176:  Learning rate = 0.002710:   Batch Loss = 0.144523, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5146982669830322, Accuracy = 0.8999999761581421
    Iter #1586688:  Learning rate = 0.002710:   Batch Loss = 0.146028, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5098695755004883, Accuracy = 0.9016194343566895
    Iter #1587200:  Learning rate = 0.002710:   Batch Loss = 0.150155, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.513881266117096, Accuracy = 0.9016194343566895
    Iter #1587712:  Learning rate = 0.002710:   Batch Loss = 0.145744, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.515198290348053, Accuracy = 0.9008097052574158
    Iter #1588224:  Learning rate = 0.002710:   Batch Loss = 0.143296, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5161725282669067, Accuracy = 0.901214599609375
    Iter #1588736:  Learning rate = 0.002710:   Batch Loss = 0.148012, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5151723027229309, Accuracy = 0.9020242691040039
    Iter #1589248:  Learning rate = 0.002710:   Batch Loss = 0.144985, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5159093737602234, Accuracy = 0.9004048705101013
    Iter #1589760:  Learning rate = 0.002710:   Batch Loss = 0.143382, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5169199705123901, Accuracy = 0.8991903066635132
    Iter #1590272:  Learning rate = 0.002710:   Batch Loss = 0.145383, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.511812150478363, Accuracy = 0.901214599609375
    Iter #1590784:  Learning rate = 0.002710:   Batch Loss = 0.143330, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5045344829559326, Accuracy = 0.9028339982032776
    Iter #1591296:  Learning rate = 0.002710:   Batch Loss = 0.142889, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5127262473106384, Accuracy = 0.8967611193656921
    Iter #1591808:  Learning rate = 0.002710:   Batch Loss = 0.142176, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.513655960559845, Accuracy = 0.9020242691040039
    Iter #1592320:  Learning rate = 0.002710:   Batch Loss = 0.146175, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5102192163467407, Accuracy = 0.9056680202484131
    Iter #1592832:  Learning rate = 0.002710:   Batch Loss = 0.141922, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5125125646591187, Accuracy = 0.9024291634559631
    Iter #1593344:  Learning rate = 0.002710:   Batch Loss = 0.141440, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5225637555122375, Accuracy = 0.8995951414108276
    Iter #1593856:  Learning rate = 0.002710:   Batch Loss = 0.141386, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5116455554962158, Accuracy = 0.904453456401825
    Iter #1594368:  Learning rate = 0.002710:   Batch Loss = 0.141479, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.503747820854187, Accuracy = 0.9032388925552368
    Iter #1594880:  Learning rate = 0.002710:   Batch Loss = 0.140091, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5162999033927917, Accuracy = 0.8975708484649658
    Iter #1595392:  Learning rate = 0.002710:   Batch Loss = 0.142232, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5129865407943726, Accuracy = 0.9028339982032776
    Iter #1595904:  Learning rate = 0.002710:   Batch Loss = 0.145885, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5111048221588135, Accuracy = 0.8991903066635132
    Iter #1596416:  Learning rate = 0.002710:   Batch Loss = 0.137997, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5111302137374878, Accuracy = 0.9024291634559631
    Iter #1596928:  Learning rate = 0.002710:   Batch Loss = 0.145380, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5173951983451843, Accuracy = 0.8975708484649658
    Iter #1597440:  Learning rate = 0.002710:   Batch Loss = 0.141786, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5171459317207336, Accuracy = 0.895546555519104
    Iter #1597952:  Learning rate = 0.002710:   Batch Loss = 0.139154, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5224373936653137, Accuracy = 0.8971660137176514
    Iter #1598464:  Learning rate = 0.002710:   Batch Loss = 0.142934, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5177885293960571, Accuracy = 0.8995951414108276
    Iter #1598976:  Learning rate = 0.002710:   Batch Loss = 0.141432, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5094263553619385, Accuracy = 0.9008097052574158
    Iter #1599488:  Learning rate = 0.002710:   Batch Loss = 0.138600, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5085095167160034, Accuracy = 0.9004048705101013
    Iter #1600000:  Learning rate = 0.002602:   Batch Loss = 0.145651, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5171681642532349, Accuracy = 0.8971660137176514
    Iter #1600512:  Learning rate = 0.002602:   Batch Loss = 0.142996, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5202264785766602, Accuracy = 0.8967611193656921
    Iter #1601024:  Learning rate = 0.002602:   Batch Loss = 0.139012, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5256354808807373, Accuracy = 0.8951417207717896
    Iter #1601536:  Learning rate = 0.002602:   Batch Loss = 0.140626, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5181944370269775, Accuracy = 0.8975708484649658
    Iter #1602048:  Learning rate = 0.002602:   Batch Loss = 0.138887, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5192038416862488, Accuracy = 0.8943319916725159
    Iter #1602560:  Learning rate = 0.002602:   Batch Loss = 0.138299, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5106511116027832, Accuracy = 0.9008097052574158
    Iter #1603072:  Learning rate = 0.002602:   Batch Loss = 0.138258, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5175417065620422, Accuracy = 0.892307698726654
    Iter #1603584:  Learning rate = 0.002602:   Batch Loss = 0.137627, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5242166519165039, Accuracy = 0.8951417207717896
    Iter #1604096:  Learning rate = 0.002602:   Batch Loss = 0.136058, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5120000839233398, Accuracy = 0.8963562846183777
    Iter #1604608:  Learning rate = 0.002602:   Batch Loss = 0.142850, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5145528316497803, Accuracy = 0.8963562846183777
    Iter #1605120:  Learning rate = 0.002602:   Batch Loss = 0.137657, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5410110950469971, Accuracy = 0.8882591128349304
    Iter #1605632:  Learning rate = 0.002602:   Batch Loss = 0.138765, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5487798452377319, Accuracy = 0.8854250907897949
    Iter #1606144:  Learning rate = 0.002602:   Batch Loss = 0.137858, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5439721345901489, Accuracy = 0.8866396546363831
    Iter #1606656:  Learning rate = 0.002602:   Batch Loss = 0.152120, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5353173613548279, Accuracy = 0.8931174278259277
    Iter #1607168:  Learning rate = 0.002602:   Batch Loss = 0.159560, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6667081117630005, Accuracy = 0.8587044477462769
    Iter #1607680:  Learning rate = 0.002602:   Batch Loss = 0.180962, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6965261697769165, Accuracy = 0.8392712473869324
    Iter #1608192:  Learning rate = 0.002602:   Batch Loss = 0.328154, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6852543354034424, Accuracy = 0.8384615182876587
    Iter #1608704:  Learning rate = 0.002602:   Batch Loss = 0.331878, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7012429237365723, Accuracy = 0.8506072759628296
    Iter #1609216:  Learning rate = 0.002602:   Batch Loss = 0.587965, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8974769115447998, Accuracy = 0.7858299612998962
    Iter #1609728:  Learning rate = 0.002602:   Batch Loss = 0.517046, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7435271739959717, Accuracy = 0.8331983685493469
    Iter #1610240:  Learning rate = 0.002602:   Batch Loss = 0.326471, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7406230568885803, Accuracy = 0.8307692408561707
    Iter #1610752:  Learning rate = 0.002602:   Batch Loss = 0.298566, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7530385255813599, Accuracy = 0.8352226614952087
    Iter #1611264:  Learning rate = 0.002602:   Batch Loss = 0.504961, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7892343401908875, Accuracy = 0.8202429413795471
    Iter #1611776:  Learning rate = 0.002602:   Batch Loss = 0.308973, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8162758350372314, Accuracy = 0.8068826198577881
    Iter #1612288:  Learning rate = 0.002602:   Batch Loss = 0.461391, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7569184303283691, Accuracy = 0.826720654964447
    Iter #1612800:  Learning rate = 0.002602:   Batch Loss = 0.437407, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7678715586662292, Accuracy = 0.8287449479103088
    Iter #1613312:  Learning rate = 0.002602:   Batch Loss = 0.309829, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7447830438613892, Accuracy = 0.8336032629013062
    Iter #1613824:  Learning rate = 0.002602:   Batch Loss = 0.333583, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6976934671401978, Accuracy = 0.8514170050621033
    Iter #1614336:  Learning rate = 0.002602:   Batch Loss = 0.402373, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7672038078308105, Accuracy = 0.8376518487930298
    Iter #1614848:  Learning rate = 0.002602:   Batch Loss = 0.310164, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7552412152290344, Accuracy = 0.8384615182876587
    Iter #1615360:  Learning rate = 0.002602:   Batch Loss = 0.278719, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7166423797607422, Accuracy = 0.8453441262245178
    Iter #1615872:  Learning rate = 0.002602:   Batch Loss = 0.277563, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7569633722305298, Accuracy = 0.8319838047027588
    Iter #1616384:  Learning rate = 0.002602:   Batch Loss = 0.371265, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7375550270080566, Accuracy = 0.8461538553237915
    Iter #1616896:  Learning rate = 0.002602:   Batch Loss = 0.346979, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7395837903022766, Accuracy = 0.8449392914772034
    Iter #1617408:  Learning rate = 0.002602:   Batch Loss = 0.277941, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7481056451797485, Accuracy = 0.8368421196937561
    Iter #1617920:  Learning rate = 0.002602:   Batch Loss = 0.252173, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7053802013397217, Accuracy = 0.8570850491523743
    Iter #1618432:  Learning rate = 0.002602:   Batch Loss = 0.240941, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6690981388092041, Accuracy = 0.8684210777282715
    Iter #1618944:  Learning rate = 0.002602:   Batch Loss = 0.263192, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6978894472122192, Accuracy = 0.8587044477462769
    Iter #1619456:  Learning rate = 0.002602:   Batch Loss = 0.278615, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6473903656005859, Accuracy = 0.8684210777282715
    Iter #1619968:  Learning rate = 0.002602:   Batch Loss = 0.213964, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6693453788757324, Accuracy = 0.8627530336380005
    Iter #1620480:  Learning rate = 0.002602:   Batch Loss = 0.242272, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6280213594436646, Accuracy = 0.8805667757987976
    Iter #1620992:  Learning rate = 0.002602:   Batch Loss = 0.212477, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.611438512802124, Accuracy = 0.8862348198890686
    Iter #1621504:  Learning rate = 0.002602:   Batch Loss = 0.203215, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6434582471847534, Accuracy = 0.8753036260604858
    Iter #1622016:  Learning rate = 0.002602:   Batch Loss = 0.200940, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6533763408660889, Accuracy = 0.8753036260604858
    Iter #1622528:  Learning rate = 0.002602:   Batch Loss = 0.231011, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6623470783233643, Accuracy = 0.8570850491523743
    Iter #1623040:  Learning rate = 0.002602:   Batch Loss = 0.249292, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.638024091720581, Accuracy = 0.8704453706741333
    Iter #1623552:  Learning rate = 0.002602:   Batch Loss = 0.253482, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5966322422027588, Accuracy = 0.8854250907897949
    Iter #1624064:  Learning rate = 0.002602:   Batch Loss = 0.211061, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6031432151794434, Accuracy = 0.8809716701507568
    Iter #1624576:  Learning rate = 0.002602:   Batch Loss = 0.199189, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5951728820800781, Accuracy = 0.8858299851417542
    Iter #1625088:  Learning rate = 0.002602:   Batch Loss = 0.211133, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5789951086044312, Accuracy = 0.8935222625732422
    Iter #1625600:  Learning rate = 0.002602:   Batch Loss = 0.201420, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5922032594680786, Accuracy = 0.8874493837356567
    Iter #1626112:  Learning rate = 0.002602:   Batch Loss = 0.194246, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6017569303512573, Accuracy = 0.8785424828529358
    Iter #1626624:  Learning rate = 0.002602:   Batch Loss = 0.187795, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5933858752250671, Accuracy = 0.8862348198890686
    Iter #1627136:  Learning rate = 0.002602:   Batch Loss = 0.184567, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5702630281448364, Accuracy = 0.8910931348800659
    Iter #1627648:  Learning rate = 0.002602:   Batch Loss = 0.179774, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5589388608932495, Accuracy = 0.8943319916725159
    Iter #1628160:  Learning rate = 0.002602:   Batch Loss = 0.180670, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5690620541572571, Accuracy = 0.8963562846183777
    Iter #1628672:  Learning rate = 0.002602:   Batch Loss = 0.180300, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5588187575340271, Accuracy = 0.8971660137176514
    Iter #1629184:  Learning rate = 0.002602:   Batch Loss = 0.179103, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5428948998451233, Accuracy = 0.9028339982032776
    Iter #1629696:  Learning rate = 0.002602:   Batch Loss = 0.174023, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5338216423988342, Accuracy = 0.9008097052574158
    Iter #1630208:  Learning rate = 0.002602:   Batch Loss = 0.175563, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5328359603881836, Accuracy = 0.9008097052574158
    Iter #1630720:  Learning rate = 0.002602:   Batch Loss = 0.173198, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5361883640289307, Accuracy = 0.8975708484649658
    Iter #1631232:  Learning rate = 0.002602:   Batch Loss = 0.175084, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5374146103858948, Accuracy = 0.8979756832122803
    Iter #1631744:  Learning rate = 0.002602:   Batch Loss = 0.169544, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5324902534484863, Accuracy = 0.8971660137176514
    Iter #1632256:  Learning rate = 0.002602:   Batch Loss = 0.171211, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.533576250076294, Accuracy = 0.8947368264198303
    Iter #1632768:  Learning rate = 0.002602:   Batch Loss = 0.167334, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.531655490398407, Accuracy = 0.8979756832122803
    Iter #1633280:  Learning rate = 0.002602:   Batch Loss = 0.167005, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5291525721549988, Accuracy = 0.8967611193656921
    Iter #1633792:  Learning rate = 0.002602:   Batch Loss = 0.169883, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5262513756752014, Accuracy = 0.898785412311554
    Iter #1634304:  Learning rate = 0.002602:   Batch Loss = 0.167644, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5255405306816101, Accuracy = 0.9004048705101013
    Iter #1634816:  Learning rate = 0.002602:   Batch Loss = 0.168518, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5279308557510376, Accuracy = 0.8995951414108276
    Iter #1635328:  Learning rate = 0.002602:   Batch Loss = 0.165409, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5246487855911255, Accuracy = 0.8979756832122803
    Iter #1635840:  Learning rate = 0.002602:   Batch Loss = 0.165303, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5257831811904907, Accuracy = 0.8979756832122803
    Iter #1636352:  Learning rate = 0.002602:   Batch Loss = 0.163285, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5260173678398132, Accuracy = 0.8951417207717896
    Iter #1636864:  Learning rate = 0.002602:   Batch Loss = 0.162218, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.524075984954834, Accuracy = 0.8951417207717896
    Iter #1637376:  Learning rate = 0.002602:   Batch Loss = 0.166180, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.523970365524292, Accuracy = 0.8971660137176514
    Iter #1637888:  Learning rate = 0.002602:   Batch Loss = 0.160896, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5283814072608948, Accuracy = 0.898785412311554
    Iter #1638400:  Learning rate = 0.002602:   Batch Loss = 0.162211, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5294793248176575, Accuracy = 0.898785412311554
    Iter #1638912:  Learning rate = 0.002602:   Batch Loss = 0.164389, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5269293785095215, Accuracy = 0.8983805775642395
    Iter #1639424:  Learning rate = 0.002602:   Batch Loss = 0.158934, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5245450139045715, Accuracy = 0.8975708484649658
    Iter #1639936:  Learning rate = 0.002602:   Batch Loss = 0.161368, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5244448184967041, Accuracy = 0.8971660137176514
    Iter #1640448:  Learning rate = 0.002602:   Batch Loss = 0.158089, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5243474245071411, Accuracy = 0.8963562846183777
    Iter #1640960:  Learning rate = 0.002602:   Batch Loss = 0.158150, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5260063409805298, Accuracy = 0.895546555519104
    Iter #1641472:  Learning rate = 0.002602:   Batch Loss = 0.158323, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.527543842792511, Accuracy = 0.8963562846183777
    Iter #1641984:  Learning rate = 0.002602:   Batch Loss = 0.160327, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5238943099975586, Accuracy = 0.8951417207717896
    Iter #1642496:  Learning rate = 0.002602:   Batch Loss = 0.158834, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5226120948791504, Accuracy = 0.8951417207717896
    Iter #1643008:  Learning rate = 0.002602:   Batch Loss = 0.158453, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5293604731559753, Accuracy = 0.8935222625732422
    Iter #1643520:  Learning rate = 0.002602:   Batch Loss = 0.156839, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5335051417350769, Accuracy = 0.8919028043746948
    Iter #1644032:  Learning rate = 0.002602:   Batch Loss = 0.154560, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5309839248657227, Accuracy = 0.8931174278259277
    Iter #1644544:  Learning rate = 0.002602:   Batch Loss = 0.155747, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5290675759315491, Accuracy = 0.895546555519104
    Iter #1645056:  Learning rate = 0.002602:   Batch Loss = 0.155140, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5320125222206116, Accuracy = 0.8927125334739685
    Iter #1645568:  Learning rate = 0.002602:   Batch Loss = 0.152157, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5327245593070984, Accuracy = 0.8906882405281067
    Iter #1646080:  Learning rate = 0.002602:   Batch Loss = 0.156154, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5318098664283752, Accuracy = 0.895546555519104
    Iter #1646592:  Learning rate = 0.002602:   Batch Loss = 0.151572, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5344976186752319, Accuracy = 0.8935222625732422
    Iter #1647104:  Learning rate = 0.002602:   Batch Loss = 0.153624, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5361650586128235, Accuracy = 0.8914979696273804
    Iter #1647616:  Learning rate = 0.002602:   Batch Loss = 0.151539, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5351313352584839, Accuracy = 0.8902834057807922
    Iter #1648128:  Learning rate = 0.002602:   Batch Loss = 0.150394, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5317497849464417, Accuracy = 0.8902834057807922
    Iter #1648640:  Learning rate = 0.002602:   Batch Loss = 0.148302, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5214577913284302, Accuracy = 0.8931174278259277
    Iter #1649152:  Learning rate = 0.002602:   Batch Loss = 0.150068, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5203953981399536, Accuracy = 0.8971660137176514
    Iter #1649664:  Learning rate = 0.002602:   Batch Loss = 0.152193, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5251762866973877, Accuracy = 0.8947368264198303
    Iter #1650176:  Learning rate = 0.002602:   Batch Loss = 0.152088, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5340777635574341, Accuracy = 0.8906882405281067
    Iter #1650688:  Learning rate = 0.002602:   Batch Loss = 0.151081, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5289180278778076, Accuracy = 0.8898785710334778
    Iter #1651200:  Learning rate = 0.002602:   Batch Loss = 0.150440, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5256386995315552, Accuracy = 0.8906882405281067
    Iter #1651712:  Learning rate = 0.002602:   Batch Loss = 0.151252, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5297749042510986, Accuracy = 0.8910931348800659
    Iter #1652224:  Learning rate = 0.002602:   Batch Loss = 0.150914, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5331195592880249, Accuracy = 0.8910931348800659
    Iter #1652736:  Learning rate = 0.002602:   Batch Loss = 0.149838, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5288550853729248, Accuracy = 0.8939270973205566
    Iter #1653248:  Learning rate = 0.002602:   Batch Loss = 0.148738, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.529704213142395, Accuracy = 0.8898785710334778
    Iter #1653760:  Learning rate = 0.002602:   Batch Loss = 0.147342, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5391313433647156, Accuracy = 0.8910931348800659
    Iter #1654272:  Learning rate = 0.002602:   Batch Loss = 0.150604, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5418249368667603, Accuracy = 0.8858299851417542
    Iter #1654784:  Learning rate = 0.002602:   Batch Loss = 0.148300, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5348436236381531, Accuracy = 0.8902834057807922
    Iter #1655296:  Learning rate = 0.002602:   Batch Loss = 0.144912, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5270788669586182, Accuracy = 0.8947368264198303
    Iter #1655808:  Learning rate = 0.002602:   Batch Loss = 0.145726, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5247368812561035, Accuracy = 0.8935222625732422
    Iter #1656320:  Learning rate = 0.002602:   Batch Loss = 0.146022, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5295672416687012, Accuracy = 0.892307698726654
    Iter #1656832:  Learning rate = 0.002602:   Batch Loss = 0.146058, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5263263583183289, Accuracy = 0.892307698726654
    Iter #1657344:  Learning rate = 0.002602:   Batch Loss = 0.145715, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5315307974815369, Accuracy = 0.8910931348800659
    Iter #1657856:  Learning rate = 0.002602:   Batch Loss = 0.148089, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5290696620941162, Accuracy = 0.8919028043746948
    Iter #1658368:  Learning rate = 0.002602:   Batch Loss = 0.146510, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5282589197158813, Accuracy = 0.8898785710334778
    Iter #1658880:  Learning rate = 0.002602:   Batch Loss = 0.145690, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.527198076248169, Accuracy = 0.8902834057807922
    Iter #1659392:  Learning rate = 0.002602:   Batch Loss = 0.144441, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5274494290351868, Accuracy = 0.887854278087616
    Iter #1659904:  Learning rate = 0.002602:   Batch Loss = 0.143328, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5320552587509155, Accuracy = 0.8906882405281067
    Iter #1660416:  Learning rate = 0.002602:   Batch Loss = 0.142640, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5286022424697876, Accuracy = 0.8890688419342041
    Iter #1660928:  Learning rate = 0.002602:   Batch Loss = 0.142730, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5310219526290894, Accuracy = 0.8886639475822449
    Iter #1661440:  Learning rate = 0.002602:   Batch Loss = 0.144698, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5301570892333984, Accuracy = 0.8862348198890686
    Iter #1661952:  Learning rate = 0.002602:   Batch Loss = 0.146003, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5315625667572021, Accuracy = 0.8874493837356567
    Iter #1662464:  Learning rate = 0.002602:   Batch Loss = 0.143127, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5365286469459534, Accuracy = 0.8902834057807922
    Iter #1662976:  Learning rate = 0.002602:   Batch Loss = 0.144819, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5319175124168396, Accuracy = 0.8890688419342041
    Iter #1663488:  Learning rate = 0.002602:   Batch Loss = 0.143207, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5387395024299622, Accuracy = 0.8870445489883423
    Iter #1664000:  Learning rate = 0.002602:   Batch Loss = 0.139136, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5305481553077698, Accuracy = 0.8910931348800659
    Iter #1664512:  Learning rate = 0.002602:   Batch Loss = 0.141475, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5256707072257996, Accuracy = 0.8959513902664185
    Iter #1665024:  Learning rate = 0.002602:   Batch Loss = 0.144989, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5317850708961487, Accuracy = 0.8919028043746948
    Iter #1665536:  Learning rate = 0.002602:   Batch Loss = 0.145519, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5383129119873047, Accuracy = 0.8846153616905212
    Iter #1666048:  Learning rate = 0.002602:   Batch Loss = 0.142409, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5229232311248779, Accuracy = 0.8927125334739685
    Iter #1666560:  Learning rate = 0.002602:   Batch Loss = 0.143895, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5226563215255737, Accuracy = 0.8939270973205566
    Iter #1667072:  Learning rate = 0.002602:   Batch Loss = 0.143485, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.536436915397644, Accuracy = 0.8886639475822449
    Iter #1667584:  Learning rate = 0.002602:   Batch Loss = 0.146310, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5344176888465881, Accuracy = 0.8910931348800659
    Iter #1668096:  Learning rate = 0.002602:   Batch Loss = 0.141020, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5229654312133789, Accuracy = 0.8890688419342041
    Iter #1668608:  Learning rate = 0.002602:   Batch Loss = 0.146779, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5387846231460571, Accuracy = 0.887854278087616
    Iter #1669120:  Learning rate = 0.002602:   Batch Loss = 0.141472, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5578107237815857, Accuracy = 0.8850202560424805
    Iter #1669632:  Learning rate = 0.002602:   Batch Loss = 0.139730, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5319406986236572, Accuracy = 0.8890688419342041
    Iter #1670144:  Learning rate = 0.002602:   Batch Loss = 0.141143, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5310835838317871, Accuracy = 0.8902834057807922
    Iter #1670656:  Learning rate = 0.002602:   Batch Loss = 0.136865, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5453741550445557, Accuracy = 0.8854250907897949
    Iter #1671168:  Learning rate = 0.002602:   Batch Loss = 0.138611, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5413033962249756, Accuracy = 0.8906882405281067
    Iter #1671680:  Learning rate = 0.002602:   Batch Loss = 0.143273, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5363088250160217, Accuracy = 0.8882591128349304
    Iter #1672192:  Learning rate = 0.002602:   Batch Loss = 0.138844, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5334746837615967, Accuracy = 0.8894736766815186
    Iter #1672704:  Learning rate = 0.002602:   Batch Loss = 0.137517, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5425853729248047, Accuracy = 0.8870445489883423
    Iter #1673216:  Learning rate = 0.002602:   Batch Loss = 0.234277, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5881060361862183, Accuracy = 0.8809716701507568
    Iter #1673728:  Learning rate = 0.002602:   Batch Loss = 0.157086, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6469793319702148, Accuracy = 0.8587044477462769
    Iter #1674240:  Learning rate = 0.002602:   Batch Loss = 0.206491, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6348446607589722, Accuracy = 0.8603239059448242
    Iter #1674752:  Learning rate = 0.002602:   Batch Loss = 0.389484, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6637327075004578, Accuracy = 0.8554655909538269
    Iter #1675264:  Learning rate = 0.002602:   Batch Loss = 0.346598, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8090950846672058, Accuracy = 0.807692289352417
    Iter #1675776:  Learning rate = 0.002602:   Batch Loss = 0.406695, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7131881713867188, Accuracy = 0.8352226614952087
    Iter #1676288:  Learning rate = 0.002602:   Batch Loss = 0.351100, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.746289074420929, Accuracy = 0.8323886394500732
    Iter #1676800:  Learning rate = 0.002602:   Batch Loss = 0.251644, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.736129105091095, Accuracy = 0.8380566835403442
    Iter #1677312:  Learning rate = 0.002602:   Batch Loss = 0.227692, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8028841018676758, Accuracy = 0.8113360404968262
    Iter #1677824:  Learning rate = 0.002602:   Batch Loss = 0.390321, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7036078572273254, Accuracy = 0.8441295623779297
    Iter #1678336:  Learning rate = 0.002602:   Batch Loss = 0.244341, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.679349422454834, Accuracy = 0.8502024412155151
    Iter #1678848:  Learning rate = 0.002602:   Batch Loss = 0.282510, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6827106475830078, Accuracy = 0.8477732539176941
    Iter #1679360:  Learning rate = 0.002602:   Batch Loss = 0.230152, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7326355576515198, Accuracy = 0.8469635844230652
    Iter #1679872:  Learning rate = 0.002602:   Batch Loss = 0.419503, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9599113464355469, Accuracy = 0.7951416969299316
    Iter #1680384:  Learning rate = 0.002602:   Batch Loss = 0.695736, Accuracy = 0.875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9182044863700867, Accuracy = 0.7809716463088989
    Iter #1680896:  Learning rate = 0.002602:   Batch Loss = 0.355798, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8513221144676208, Accuracy = 0.7959514260292053
    Iter #1681408:  Learning rate = 0.002602:   Batch Loss = 0.297925, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7038365602493286, Accuracy = 0.8481781482696533
    Iter #1681920:  Learning rate = 0.002602:   Batch Loss = 0.377226, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6917340159416199, Accuracy = 0.8429149985313416
    Iter #1682432:  Learning rate = 0.002602:   Batch Loss = 0.254257, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7597364783287048, Accuracy = 0.8295546770095825
    Iter #1682944:  Learning rate = 0.002602:   Batch Loss = 0.308113, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6595928072929382, Accuracy = 0.8578947186470032
    Iter #1683456:  Learning rate = 0.002602:   Batch Loss = 0.295075, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6973551511764526, Accuracy = 0.8461538553237915
    Iter #1683968:  Learning rate = 0.002602:   Batch Loss = 0.411486, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6692878007888794, Accuracy = 0.8615384697914124
    Iter #1684480:  Learning rate = 0.002602:   Batch Loss = 0.349817, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.695614218711853, Accuracy = 0.8562753200531006
    Iter #1684992:  Learning rate = 0.002602:   Batch Loss = 0.250486, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6385775804519653, Accuracy = 0.8736842274665833
    Iter #1685504:  Learning rate = 0.002602:   Batch Loss = 0.206079, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6264567375183105, Accuracy = 0.8809716701507568
    Iter #1686016:  Learning rate = 0.002602:   Batch Loss = 0.226739, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6570313572883606, Accuracy = 0.8700404763221741
    Iter #1686528:  Learning rate = 0.002602:   Batch Loss = 0.205131, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6135266423225403, Accuracy = 0.878947377204895
    Iter #1687040:  Learning rate = 0.002602:   Batch Loss = 0.216132, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6072930097579956, Accuracy = 0.8793522119522095
    Iter #1687552:  Learning rate = 0.002602:   Batch Loss = 0.210050, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5965070724487305, Accuracy = 0.8842105269432068
    Iter #1688064:  Learning rate = 0.002602:   Batch Loss = 0.226761, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5924867987632751, Accuracy = 0.887854278087616
    Iter #1688576:  Learning rate = 0.002602:   Batch Loss = 0.298575, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5944879055023193, Accuracy = 0.8817813992500305
    Iter #1689088:  Learning rate = 0.002602:   Batch Loss = 0.278549, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5863175392150879, Accuracy = 0.8919028043746948
    Iter #1689600:  Learning rate = 0.002602:   Batch Loss = 0.270678, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5993654727935791, Accuracy = 0.8894736766815186
    Iter #1690112:  Learning rate = 0.002602:   Batch Loss = 0.195580, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6021783351898193, Accuracy = 0.8882591128349304
    Iter #1690624:  Learning rate = 0.002602:   Batch Loss = 0.216398, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5991579294204712, Accuracy = 0.8910931348800659
    Iter #1691136:  Learning rate = 0.002602:   Batch Loss = 0.189692, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5956553816795349, Accuracy = 0.8858299851417542
    Iter #1691648:  Learning rate = 0.002602:   Batch Loss = 0.186181, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5791128277778625, Accuracy = 0.8919028043746948
    Iter #1692160:  Learning rate = 0.002602:   Batch Loss = 0.206384, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5986640453338623, Accuracy = 0.8846153616905212
    Iter #1692672:  Learning rate = 0.002602:   Batch Loss = 0.206803, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5897345542907715, Accuracy = 0.8902834057807922
    Iter #1693184:  Learning rate = 0.002602:   Batch Loss = 0.184482, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5863179564476013, Accuracy = 0.8927125334739685
    Iter #1693696:  Learning rate = 0.002602:   Batch Loss = 0.184891, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5724985003471375, Accuracy = 0.8971660137176514
    Iter #1694208:  Learning rate = 0.002602:   Batch Loss = 0.184814, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5675971508026123, Accuracy = 0.8979756832122803
    Iter #1694720:  Learning rate = 0.002602:   Batch Loss = 0.201197, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5574192404747009, Accuracy = 0.895546555519104
    Iter #1695232:  Learning rate = 0.002602:   Batch Loss = 0.180903, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5628703832626343, Accuracy = 0.8919028043746948
    Iter #1695744:  Learning rate = 0.002602:   Batch Loss = 0.207391, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5518752336502075, Accuracy = 0.8979756832122803
    Iter #1696256:  Learning rate = 0.002602:   Batch Loss = 0.189853, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5563362836837769, Accuracy = 0.8967611193656921
    Iter #1696768:  Learning rate = 0.002602:   Batch Loss = 0.178373, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5680577754974365, Accuracy = 0.892307698726654
    Iter #1697280:  Learning rate = 0.002602:   Batch Loss = 0.179855, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5540165901184082, Accuracy = 0.8939270973205566
    Iter #1697792:  Learning rate = 0.002602:   Batch Loss = 0.172513, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5410968661308289, Accuracy = 0.9004048705101013
    Iter #1698304:  Learning rate = 0.002602:   Batch Loss = 0.173358, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5353221297264099, Accuracy = 0.9048582911491394
    Iter #1698816:  Learning rate = 0.002602:   Batch Loss = 0.174184, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5243878960609436, Accuracy = 0.9040485620498657
    Iter #1699328:  Learning rate = 0.002602:   Batch Loss = 0.171822, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.522765040397644, Accuracy = 0.9052631855010986
    Iter #1699840:  Learning rate = 0.002602:   Batch Loss = 0.171606, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5226390361785889, Accuracy = 0.904453456401825
    Iter #1700352:  Learning rate = 0.002498:   Batch Loss = 0.169460, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5247957706451416, Accuracy = 0.9032388925552368
    Iter #1700864:  Learning rate = 0.002498:   Batch Loss = 0.168473, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5263385772705078, Accuracy = 0.9028339982032776
    Iter #1701376:  Learning rate = 0.002498:   Batch Loss = 0.168385, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.529525876045227, Accuracy = 0.901214599609375
    Iter #1701888:  Learning rate = 0.002498:   Batch Loss = 0.170095, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.529553234577179, Accuracy = 0.9008097052574158
    Iter #1702400:  Learning rate = 0.002498:   Batch Loss = 0.166333, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.528214693069458, Accuracy = 0.9008097052574158
    Iter #1702912:  Learning rate = 0.002498:   Batch Loss = 0.170367, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5290766954421997, Accuracy = 0.8991903066635132
    Iter #1703424:  Learning rate = 0.002498:   Batch Loss = 0.166294, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5277349948883057, Accuracy = 0.9004048705101013
    Iter #1703936:  Learning rate = 0.002498:   Batch Loss = 0.165932, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5297516584396362, Accuracy = 0.898785412311554
    Iter #1704448:  Learning rate = 0.002498:   Batch Loss = 0.167095, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5290603041648865, Accuracy = 0.8995951414108276
    Iter #1704960:  Learning rate = 0.002498:   Batch Loss = 0.165239, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.529991090297699, Accuracy = 0.8971660137176514
    Iter #1705472:  Learning rate = 0.002498:   Batch Loss = 0.161406, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5264866352081299, Accuracy = 0.8983805775642395
    Iter #1705984:  Learning rate = 0.002498:   Batch Loss = 0.162203, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5258195400238037, Accuracy = 0.8971660137176514
    Iter #1706496:  Learning rate = 0.002498:   Batch Loss = 0.163778, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.523478090763092, Accuracy = 0.9004048705101013
    Iter #1707008:  Learning rate = 0.002498:   Batch Loss = 0.160521, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5198906064033508, Accuracy = 0.8999999761581421
    Iter #1707520:  Learning rate = 0.002498:   Batch Loss = 0.162518, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5206584930419922, Accuracy = 0.9016194343566895
    Iter #1708032:  Learning rate = 0.002498:   Batch Loss = 0.159776, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5202337503433228, Accuracy = 0.8999999761581421
    Iter #1708544:  Learning rate = 0.002498:   Batch Loss = 0.157828, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5224931836128235, Accuracy = 0.8963562846183777
    Iter #1709056:  Learning rate = 0.002498:   Batch Loss = 0.159302, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5226513147354126, Accuracy = 0.9004048705101013
    Iter #1709568:  Learning rate = 0.002498:   Batch Loss = 0.158161, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5225536227226257, Accuracy = 0.8999999761581421
    Iter #1710080:  Learning rate = 0.002498:   Batch Loss = 0.160090, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5242116451263428, Accuracy = 0.8983805775642395
    Iter #1710592:  Learning rate = 0.002498:   Batch Loss = 0.159500, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5211334824562073, Accuracy = 0.9008097052574158
    Iter #1711104:  Learning rate = 0.002498:   Batch Loss = 0.159816, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5190718770027161, Accuracy = 0.9016194343566895
    Iter #1711616:  Learning rate = 0.002498:   Batch Loss = 0.159251, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5200042724609375, Accuracy = 0.8967611193656921
    Iter #1712128:  Learning rate = 0.002498:   Batch Loss = 0.157800, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5206519365310669, Accuracy = 0.8906882405281067
    Iter #1712640:  Learning rate = 0.002498:   Batch Loss = 0.157107, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5229495167732239, Accuracy = 0.8943319916725159
    Iter #1713152:  Learning rate = 0.002498:   Batch Loss = 0.157806, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5248576402664185, Accuracy = 0.8991903066635132
    Iter #1713664:  Learning rate = 0.002498:   Batch Loss = 0.157009, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5259225964546204, Accuracy = 0.8963562846183777
    Iter #1714176:  Learning rate = 0.002498:   Batch Loss = 0.158487, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5239938497543335, Accuracy = 0.8963562846183777
    Iter #1714688:  Learning rate = 0.002498:   Batch Loss = 0.155361, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5238782167434692, Accuracy = 0.895546555519104
    Iter #1715200:  Learning rate = 0.002498:   Batch Loss = 0.153250, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5223472118377686, Accuracy = 0.9008097052574158
    Iter #1715712:  Learning rate = 0.002498:   Batch Loss = 0.153639, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5233367085456848, Accuracy = 0.8943319916725159
    Iter #1716224:  Learning rate = 0.002498:   Batch Loss = 0.152780, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5213215947151184, Accuracy = 0.8959513902664185
    Iter #1716736:  Learning rate = 0.002498:   Batch Loss = 0.153712, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5203903317451477, Accuracy = 0.8999999761581421
    Iter #1717248:  Learning rate = 0.002498:   Batch Loss = 0.153231, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5218589305877686, Accuracy = 0.8983805775642395
    Iter #1717760:  Learning rate = 0.002498:   Batch Loss = 0.153240, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5203298330307007, Accuracy = 0.8975708484649658
    Iter #1718272:  Learning rate = 0.002498:   Batch Loss = 0.151408, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.519798219203949, Accuracy = 0.8951417207717896
    Iter #1718784:  Learning rate = 0.002498:   Batch Loss = 0.150839, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5188577175140381, Accuracy = 0.8963562846183777
    Iter #1719296:  Learning rate = 0.002498:   Batch Loss = 0.150702, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5191107392311096, Accuracy = 0.8971660137176514
    Iter #1719808:  Learning rate = 0.002498:   Batch Loss = 0.153131, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5164906978607178, Accuracy = 0.8995951414108276
    Iter #1720320:  Learning rate = 0.002498:   Batch Loss = 0.149309, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.516406774520874, Accuracy = 0.895546555519104
    Iter #1720832:  Learning rate = 0.002498:   Batch Loss = 0.150517, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5200099945068359, Accuracy = 0.8947368264198303
    Iter #1721344:  Learning rate = 0.002498:   Batch Loss = 0.149680, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5206423997879028, Accuracy = 0.8914979696273804
    Iter #1721856:  Learning rate = 0.002498:   Batch Loss = 0.151012, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5191313028335571, Accuracy = 0.8959513902664185
    Iter #1722368:  Learning rate = 0.002498:   Batch Loss = 0.149653, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5194825530052185, Accuracy = 0.8898785710334778
    Iter #1722880:  Learning rate = 0.002498:   Batch Loss = 0.149395, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5195033550262451, Accuracy = 0.8939270973205566
    Iter #1723392:  Learning rate = 0.002498:   Batch Loss = 0.151111, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5177353620529175, Accuracy = 0.8951417207717896
    Iter #1723904:  Learning rate = 0.002498:   Batch Loss = 0.148342, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5228059887886047, Accuracy = 0.8874493837356567
    Iter #1724416:  Learning rate = 0.002498:   Batch Loss = 0.147065, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.524986743927002, Accuracy = 0.8890688419342041
    Iter #1724928:  Learning rate = 0.002498:   Batch Loss = 0.148082, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5196874737739563, Accuracy = 0.8935222625732422
    Iter #1725440:  Learning rate = 0.002498:   Batch Loss = 0.148987, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5160950422286987, Accuracy = 0.8971660137176514
    Iter #1725952:  Learning rate = 0.002498:   Batch Loss = 0.146229, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5200104713439941, Accuracy = 0.8919028043746948
    Iter #1726464:  Learning rate = 0.002498:   Batch Loss = 0.145631, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5204602479934692, Accuracy = 0.8910931348800659
    Iter #1726976:  Learning rate = 0.002498:   Batch Loss = 0.145747, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5197726488113403, Accuracy = 0.8906882405281067
    Iter #1727488:  Learning rate = 0.002498:   Batch Loss = 0.146283, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5212436318397522, Accuracy = 0.8894736766815186
    Iter #1728000:  Learning rate = 0.002498:   Batch Loss = 0.145246, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5176687836647034, Accuracy = 0.8943319916725159
    Iter #1728512:  Learning rate = 0.002498:   Batch Loss = 0.143618, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5198181867599487, Accuracy = 0.8939270973205566
    Iter #1729024:  Learning rate = 0.002498:   Batch Loss = 0.146452, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.518195629119873, Accuracy = 0.892307698726654
    Iter #1729536:  Learning rate = 0.002498:   Batch Loss = 0.143556, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5188153386116028, Accuracy = 0.8902834057807922
    Iter #1730048:  Learning rate = 0.002498:   Batch Loss = 0.142977, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5184406638145447, Accuracy = 0.8906882405281067
    Iter #1730560:  Learning rate = 0.002498:   Batch Loss = 0.143124, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5172804594039917, Accuracy = 0.8943319916725159
    Iter #1731072:  Learning rate = 0.002498:   Batch Loss = 0.145625, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5193750858306885, Accuracy = 0.8927125334739685
    Iter #1731584:  Learning rate = 0.002498:   Batch Loss = 0.146316, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5327885746955872, Accuracy = 0.8870445489883423
    Iter #1732096:  Learning rate = 0.002498:   Batch Loss = 0.145985, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5221571326255798, Accuracy = 0.8882591128349304
    Iter #1732608:  Learning rate = 0.002498:   Batch Loss = 0.159030, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5445700883865356, Accuracy = 0.8898785710334778
    Iter #1733120:  Learning rate = 0.002498:   Batch Loss = 0.241660, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5602190494537354, Accuracy = 0.882186233997345
    Iter #1733632:  Learning rate = 0.002498:   Batch Loss = 0.249011, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6477469205856323, Accuracy = 0.8582996129989624
    Iter #1734144:  Learning rate = 0.002498:   Batch Loss = 0.225906, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6315276622772217, Accuracy = 0.8562753200531006
    Iter #1734656:  Learning rate = 0.002498:   Batch Loss = 0.188728, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6926336288452148, Accuracy = 0.8388664126396179
    Iter #1735168:  Learning rate = 0.002498:   Batch Loss = 0.272902, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6686580181121826, Accuracy = 0.8457489609718323
    Iter #1735680:  Learning rate = 0.002498:   Batch Loss = 0.487588, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6502015590667725, Accuracy = 0.8510121703147888
    Iter #1736192:  Learning rate = 0.002498:   Batch Loss = 0.265733, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.694281280040741, Accuracy = 0.8485829830169678
    Iter #1736704:  Learning rate = 0.002498:   Batch Loss = 0.239331, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6693942546844482, Accuracy = 0.848987877368927
    Iter #1737216:  Learning rate = 0.002498:   Batch Loss = 0.267254, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7450821399688721, Accuracy = 0.8340080976486206
    Iter #1737728:  Learning rate = 0.002498:   Batch Loss = 0.299463, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.663496732711792, Accuracy = 0.8485829830169678
    Iter #1738240:  Learning rate = 0.002498:   Batch Loss = 0.279692, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6832682490348816, Accuracy = 0.8518218398094177
    Iter #1738752:  Learning rate = 0.002498:   Batch Loss = 0.343949, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6864702701568604, Accuracy = 0.8530364632606506
    Iter #1739264:  Learning rate = 0.002498:   Batch Loss = 0.227922, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6617262959480286, Accuracy = 0.8639675974845886
    Iter #1739776:  Learning rate = 0.002498:   Batch Loss = 0.233827, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6644695997238159, Accuracy = 0.8663967847824097
    Iter #1740288:  Learning rate = 0.002498:   Batch Loss = 0.324149, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6802796721458435, Accuracy = 0.8587044477462769
    Iter #1740800:  Learning rate = 0.002498:   Batch Loss = 0.357050, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6418659090995789, Accuracy = 0.8643724918365479
    Iter #1741312:  Learning rate = 0.002498:   Batch Loss = 0.260975, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6688488721847534, Accuracy = 0.8550607562065125
    Iter #1741824:  Learning rate = 0.002498:   Batch Loss = 0.220260, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6325325965881348, Accuracy = 0.8680161833763123
    Iter #1742336:  Learning rate = 0.002498:   Batch Loss = 0.229924, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6179000735282898, Accuracy = 0.8781376481056213
    Iter #1742848:  Learning rate = 0.002498:   Batch Loss = 0.237127, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6502439975738525, Accuracy = 0.8663967847824097
    Iter #1743360:  Learning rate = 0.002498:   Batch Loss = 0.225325, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6416088938713074, Accuracy = 0.8740890622138977
    Iter #1743872:  Learning rate = 0.002498:   Batch Loss = 0.236056, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6477763652801514, Accuracy = 0.878947377204895
    Iter #1744384:  Learning rate = 0.002498:   Batch Loss = 0.302130, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6151201128959656, Accuracy = 0.878947377204895
    Iter #1744896:  Learning rate = 0.002498:   Batch Loss = 0.255321, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6401901245117188, Accuracy = 0.8668016195297241
    Iter #1745408:  Learning rate = 0.002498:   Batch Loss = 0.207988, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6550356149673462, Accuracy = 0.8757085204124451
    Iter #1745920:  Learning rate = 0.002498:   Batch Loss = 0.221095, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6393152475357056, Accuracy = 0.8748987913131714
    Iter #1746432:  Learning rate = 0.002498:   Batch Loss = 0.210340, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.580721378326416, Accuracy = 0.8935222625732422
    Iter #1746944:  Learning rate = 0.002498:   Batch Loss = 0.194511, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6106159687042236, Accuracy = 0.8874493837356567
    Iter #1747456:  Learning rate = 0.002498:   Batch Loss = 0.183479, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5625830292701721, Accuracy = 0.8919028043746948
    Iter #1747968:  Learning rate = 0.002498:   Batch Loss = 0.191868, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5580657720565796, Accuracy = 0.8910931348800659
    Iter #1748480:  Learning rate = 0.002498:   Batch Loss = 0.189641, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5770941972732544, Accuracy = 0.8931174278259277
    Iter #1748992:  Learning rate = 0.002498:   Batch Loss = 0.176337, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5568317174911499, Accuracy = 0.8951417207717896
    Iter #1749504:  Learning rate = 0.002498:   Batch Loss = 0.184063, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5609196424484253, Accuracy = 0.8959513902664185
    Iter #1750016:  Learning rate = 0.002498:   Batch Loss = 0.174171, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.545417308807373, Accuracy = 0.901214599609375
    Iter #1750528:  Learning rate = 0.002498:   Batch Loss = 0.174309, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5280492901802063, Accuracy = 0.9068825840950012
    Iter #1751040:  Learning rate = 0.002498:   Batch Loss = 0.170628, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5179963111877441, Accuracy = 0.9080971479415894
    Iter #1751552:  Learning rate = 0.002498:   Batch Loss = 0.170879, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.516973614692688, Accuracy = 0.9048582911491394
    Iter #1752064:  Learning rate = 0.002498:   Batch Loss = 0.169280, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5166192054748535, Accuracy = 0.9036437273025513
    Iter #1752576:  Learning rate = 0.002498:   Batch Loss = 0.166492, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5117889046669006, Accuracy = 0.9068825840950012
    Iter #1753088:  Learning rate = 0.002498:   Batch Loss = 0.167302, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5087590217590332, Accuracy = 0.9072874784469604
    Iter #1753600:  Learning rate = 0.002498:   Batch Loss = 0.165046, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5059337615966797, Accuracy = 0.9105263352394104
    Iter #1754112:  Learning rate = 0.002498:   Batch Loss = 0.166387, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5026917457580566, Accuracy = 0.9105263352394104
    Iter #1754624:  Learning rate = 0.002498:   Batch Loss = 0.163266, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5000879168510437, Accuracy = 0.9105263352394104
    Iter #1755136:  Learning rate = 0.002498:   Batch Loss = 0.163975, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49918287992477417, Accuracy = 0.9093117117881775
    Iter #1755648:  Learning rate = 0.002498:   Batch Loss = 0.164047, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4999590814113617, Accuracy = 0.908906877040863
    Iter #1756160:  Learning rate = 0.002498:   Batch Loss = 0.162284, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5020332932472229, Accuracy = 0.908906877040863
    Iter #1756672:  Learning rate = 0.002498:   Batch Loss = 0.163889, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5068213939666748, Accuracy = 0.9060728549957275
    Iter #1757184:  Learning rate = 0.002498:   Batch Loss = 0.159680, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5132607221603394, Accuracy = 0.9032388925552368
    Iter #1757696:  Learning rate = 0.002498:   Batch Loss = 0.161683, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.503253698348999, Accuracy = 0.9068825840950012
    Iter #1758208:  Learning rate = 0.002498:   Batch Loss = 0.159821, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49875926971435547, Accuracy = 0.9093117117881775
    Iter #1758720:  Learning rate = 0.002498:   Batch Loss = 0.159940, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5007864832878113, Accuracy = 0.9076923131942749
    Iter #1759232:  Learning rate = 0.002498:   Batch Loss = 0.160519, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5045547485351562, Accuracy = 0.9072874784469604
    Iter #1759744:  Learning rate = 0.002498:   Batch Loss = 0.158443, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5055381059646606, Accuracy = 0.9064777493476868
    Iter #1760256:  Learning rate = 0.002498:   Batch Loss = 0.158725, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5030252933502197, Accuracy = 0.9076923131942749
    Iter #1760768:  Learning rate = 0.002498:   Batch Loss = 0.156541, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5024300813674927, Accuracy = 0.9080971479415894
    Iter #1761280:  Learning rate = 0.002498:   Batch Loss = 0.154886, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5019598603248596, Accuracy = 0.9064777493476868
    Iter #1761792:  Learning rate = 0.002498:   Batch Loss = 0.156407, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5015349388122559, Accuracy = 0.9056680202484131
    Iter #1762304:  Learning rate = 0.002498:   Batch Loss = 0.157441, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5004063248634338, Accuracy = 0.9076923131942749
    Iter #1762816:  Learning rate = 0.002498:   Batch Loss = 0.153746, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5015599131584167, Accuracy = 0.904453456401825
    Iter #1763328:  Learning rate = 0.002498:   Batch Loss = 0.153782, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5034244656562805, Accuracy = 0.901214599609375
    Iter #1763840:  Learning rate = 0.002498:   Batch Loss = 0.153909, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5051627159118652, Accuracy = 0.901214599609375
    Iter #1764352:  Learning rate = 0.002498:   Batch Loss = 0.152317, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5090018510818481, Accuracy = 0.901214599609375
    Iter #1764864:  Learning rate = 0.002498:   Batch Loss = 0.152712, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5098501443862915, Accuracy = 0.9008097052574158
    Iter #1765376:  Learning rate = 0.002498:   Batch Loss = 0.152279, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5019971132278442, Accuracy = 0.9056680202484131
    Iter #1765888:  Learning rate = 0.002498:   Batch Loss = 0.152514, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5044740438461304, Accuracy = 0.9032388925552368
    Iter #1766400:  Learning rate = 0.002498:   Batch Loss = 0.149972, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.506449818611145, Accuracy = 0.8999999761581421
    Iter #1766912:  Learning rate = 0.002498:   Batch Loss = 0.152130, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5064256191253662, Accuracy = 0.9024291634559631
    Iter #1767424:  Learning rate = 0.002498:   Batch Loss = 0.149390, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5062109231948853, Accuracy = 0.9024291634559631
    Iter #1767936:  Learning rate = 0.002498:   Batch Loss = 0.150206, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5074403882026672, Accuracy = 0.9004048705101013
    Iter #1768448:  Learning rate = 0.002498:   Batch Loss = 0.147587, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5075311660766602, Accuracy = 0.9036437273025513
    Iter #1768960:  Learning rate = 0.002498:   Batch Loss = 0.145500, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5092468857765198, Accuracy = 0.901214599609375
    Iter #1769472:  Learning rate = 0.002498:   Batch Loss = 0.146869, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5122023224830627, Accuracy = 0.898785412311554
    Iter #1769984:  Learning rate = 0.002498:   Batch Loss = 0.147461, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5093114376068115, Accuracy = 0.9020242691040039
    Iter #1770496:  Learning rate = 0.002498:   Batch Loss = 0.147359, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5121573209762573, Accuracy = 0.9008097052574158
    Iter #1771008:  Learning rate = 0.002498:   Batch Loss = 0.146236, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5124444961547852, Accuracy = 0.9016194343566895
    Iter #1771520:  Learning rate = 0.002498:   Batch Loss = 0.146956, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5078414082527161, Accuracy = 0.8999999761581421
    Iter #1772032:  Learning rate = 0.002498:   Batch Loss = 0.146028, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.510281503200531, Accuracy = 0.8983805775642395
    Iter #1772544:  Learning rate = 0.002498:   Batch Loss = 0.147287, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5099676847457886, Accuracy = 0.9008097052574158
    Iter #1773056:  Learning rate = 0.002498:   Batch Loss = 0.144651, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5028249025344849, Accuracy = 0.9036437273025513
    Iter #1773568:  Learning rate = 0.002498:   Batch Loss = 0.144375, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5055170059204102, Accuracy = 0.9020242691040039
    Iter #1774080:  Learning rate = 0.002498:   Batch Loss = 0.144059, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5090686082839966, Accuracy = 0.8999999761581421
    Iter #1774592:  Learning rate = 0.002498:   Batch Loss = 0.145756, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.508396565914154, Accuracy = 0.9016194343566895
    Iter #1775104:  Learning rate = 0.002498:   Batch Loss = 0.145660, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5074869990348816, Accuracy = 0.9024291634559631
    Iter #1775616:  Learning rate = 0.002498:   Batch Loss = 0.144286, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5081269145011902, Accuracy = 0.9016194343566895
    Iter #1776128:  Learning rate = 0.002498:   Batch Loss = 0.142310, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5078845024108887, Accuracy = 0.8991903066635132
    Iter #1776640:  Learning rate = 0.002498:   Batch Loss = 0.143764, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5087907314300537, Accuracy = 0.8995951414108276
    Iter #1777152:  Learning rate = 0.002498:   Batch Loss = 0.141369, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5079051852226257, Accuracy = 0.901214599609375
    Iter #1777664:  Learning rate = 0.002498:   Batch Loss = 0.143241, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5108620524406433, Accuracy = 0.8991903066635132
    Iter #1778176:  Learning rate = 0.002498:   Batch Loss = 0.142495, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5124748945236206, Accuracy = 0.9024291634559631
    Iter #1778688:  Learning rate = 0.002498:   Batch Loss = 0.143508, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5090166330337524, Accuracy = 0.9024291634559631
    Iter #1779200:  Learning rate = 0.002498:   Batch Loss = 0.140351, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5069544911384583, Accuracy = 0.9004048705101013
    Iter #1779712:  Learning rate = 0.002498:   Batch Loss = 0.141055, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5131082534790039, Accuracy = 0.8979756832122803
    Iter #1780224:  Learning rate = 0.002498:   Batch Loss = 0.139074, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5134215950965881, Accuracy = 0.8983805775642395
    Iter #1780736:  Learning rate = 0.002498:   Batch Loss = 0.140902, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5112185478210449, Accuracy = 0.8995951414108276
    Iter #1781248:  Learning rate = 0.002498:   Batch Loss = 0.141352, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5067219138145447, Accuracy = 0.8999999761581421
    Iter #1781760:  Learning rate = 0.002498:   Batch Loss = 0.141057, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.508345365524292, Accuracy = 0.8983805775642395
    Iter #1782272:  Learning rate = 0.002498:   Batch Loss = 0.140498, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5103601813316345, Accuracy = 0.8975708484649658
    Iter #1782784:  Learning rate = 0.002498:   Batch Loss = 0.139371, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5164523720741272, Accuracy = 0.898785412311554
    Iter #1783296:  Learning rate = 0.002498:   Batch Loss = 0.139322, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5127639174461365, Accuracy = 0.8979756832122803
    Iter #1783808:  Learning rate = 0.002498:   Batch Loss = 0.138266, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5128180384635925, Accuracy = 0.8963562846183777
    Iter #1784320:  Learning rate = 0.002498:   Batch Loss = 0.138216, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5134756565093994, Accuracy = 0.8975708484649658
    Iter #1784832:  Learning rate = 0.002498:   Batch Loss = 0.140920, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5098999738693237, Accuracy = 0.8971660137176514
    Iter #1785344:  Learning rate = 0.002498:   Batch Loss = 0.137507, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5129111409187317, Accuracy = 0.8971660137176514
    Iter #1785856:  Learning rate = 0.002498:   Batch Loss = 0.138326, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5153947472572327, Accuracy = 0.8943319916725159
    Iter #1786368:  Learning rate = 0.002498:   Batch Loss = 0.135065, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5118266344070435, Accuracy = 0.8959513902664185
    Iter #1786880:  Learning rate = 0.002498:   Batch Loss = 0.136901, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5136227607727051, Accuracy = 0.8967611193656921
    Iter #1787392:  Learning rate = 0.002498:   Batch Loss = 0.139528, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5116511583328247, Accuracy = 0.8975708484649658
    Iter #1787904:  Learning rate = 0.002498:   Batch Loss = 0.136756, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5184153914451599, Accuracy = 0.8947368264198303
    Iter #1788416:  Learning rate = 0.002498:   Batch Loss = 0.136638, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5166556239128113, Accuracy = 0.895546555519104
    Iter #1788928:  Learning rate = 0.002498:   Batch Loss = 0.137109, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.530450165271759, Accuracy = 0.8914979696273804
    Iter #1789440:  Learning rate = 0.002498:   Batch Loss = 0.137472, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5133681297302246, Accuracy = 0.8991903066635132
    Iter #1789952:  Learning rate = 0.002498:   Batch Loss = 0.136219, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5230048894882202, Accuracy = 0.8935222625732422
    Iter #1790464:  Learning rate = 0.002498:   Batch Loss = 0.137731, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5248825550079346, Accuracy = 0.8914979696273804
    Iter #1790976:  Learning rate = 0.002498:   Batch Loss = 0.142074, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5381273031234741, Accuracy = 0.8919028043746948
    Iter #1791488:  Learning rate = 0.002498:   Batch Loss = 0.137562, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5535826683044434, Accuracy = 0.8838056921958923
    Iter #1792000:  Learning rate = 0.002498:   Batch Loss = 0.137634, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.547755241394043, Accuracy = 0.8874493837356567
    Iter #1792512:  Learning rate = 0.002498:   Batch Loss = 0.151064, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6009619235992432, Accuracy = 0.8724696636199951
    Iter #1793024:  Learning rate = 0.002498:   Batch Loss = 0.148674, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5348854660987854, Accuracy = 0.8947368264198303
    Iter #1793536:  Learning rate = 0.002498:   Batch Loss = 0.148706, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6403045058250427, Accuracy = 0.859919011592865
    Iter #1794048:  Learning rate = 0.002498:   Batch Loss = 0.252327, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6554502248764038, Accuracy = 0.8530364632606506
    Iter #1794560:  Learning rate = 0.002498:   Batch Loss = 0.226141, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6579940915107727, Accuracy = 0.8546558618545532
    Iter #1795072:  Learning rate = 0.002498:   Batch Loss = 0.307138, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7099355459213257, Accuracy = 0.843319833278656
    Iter #1795584:  Learning rate = 0.002498:   Batch Loss = 0.589111, Accuracy = 0.84375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6784005165100098, Accuracy = 0.8542510271072388
    Iter #1796096:  Learning rate = 0.002498:   Batch Loss = 0.284046, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7485647797584534, Accuracy = 0.8348178267478943
    Iter #1796608:  Learning rate = 0.002498:   Batch Loss = 0.400297, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6911230683326721, Accuracy = 0.8506072759628296
    Iter #1797120:  Learning rate = 0.002498:   Batch Loss = 0.237188, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6676961183547974, Accuracy = 0.8574898838996887
    Iter #1797632:  Learning rate = 0.002498:   Batch Loss = 0.245426, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6824960708618164, Accuracy = 0.8453441262245178
    Iter #1798144:  Learning rate = 0.002498:   Batch Loss = 0.365205, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7488396763801575, Accuracy = 0.8283400535583496
    Iter #1798656:  Learning rate = 0.002498:   Batch Loss = 0.311394, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7069987058639526, Accuracy = 0.8425101041793823
    Iter #1799168:  Learning rate = 0.002498:   Batch Loss = 0.351801, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7252388000488281, Accuracy = 0.8441295623779297
    Iter #1799680:  Learning rate = 0.002498:   Batch Loss = 0.237266, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.702882707118988, Accuracy = 0.8493927121162415
    Iter #1800192:  Learning rate = 0.002398:   Batch Loss = 0.285692, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.709097146987915, Accuracy = 0.8396761417388916
    Iter #1800704:  Learning rate = 0.002398:   Batch Loss = 0.314540, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6554310321807861, Accuracy = 0.8700404763221741
    Iter #1801216:  Learning rate = 0.002398:   Batch Loss = 0.266545, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6954878568649292, Accuracy = 0.8477732539176941
    Iter #1801728:  Learning rate = 0.002398:   Batch Loss = 0.369602, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6633472442626953, Accuracy = 0.8582996129989624
    Iter #1802240:  Learning rate = 0.002398:   Batch Loss = 0.264437, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6677547097206116, Accuracy = 0.8587044477462769
    Iter #1802752:  Learning rate = 0.002398:   Batch Loss = 0.233181, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6078873872756958, Accuracy = 0.8817813992500305
    Iter #1803264:  Learning rate = 0.002398:   Batch Loss = 0.235474, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5951990485191345, Accuracy = 0.8882591128349304
    Iter #1803776:  Learning rate = 0.002398:   Batch Loss = 0.206854, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5993157029151917, Accuracy = 0.8862348198890686
    Iter #1804288:  Learning rate = 0.002398:   Batch Loss = 0.226028, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5941551327705383, Accuracy = 0.8886639475822449
    Iter #1804800:  Learning rate = 0.002398:   Batch Loss = 0.193915, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6296412348747253, Accuracy = 0.8753036260604858
    Iter #1805312:  Learning rate = 0.002398:   Batch Loss = 0.191830, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6194014549255371, Accuracy = 0.8797571063041687
    Iter #1805824:  Learning rate = 0.002398:   Batch Loss = 0.197583, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5731932520866394, Accuracy = 0.8890688419342041
    Iter #1806336:  Learning rate = 0.002398:   Batch Loss = 0.194486, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5889262557029724, Accuracy = 0.8870445489883423
    Iter #1806848:  Learning rate = 0.002398:   Batch Loss = 0.181263, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6074674725532532, Accuracy = 0.878947377204895
    Iter #1807360:  Learning rate = 0.002398:   Batch Loss = 0.205824, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6118419766426086, Accuracy = 0.8809716701507568
    Iter #1807872:  Learning rate = 0.002398:   Batch Loss = 0.205442, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6070805788040161, Accuracy = 0.8834007978439331
    Iter #1808384:  Learning rate = 0.002398:   Batch Loss = 0.183970, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6050553321838379, Accuracy = 0.887854278087616
    Iter #1808896:  Learning rate = 0.002398:   Batch Loss = 0.187239, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6139658689498901, Accuracy = 0.8854250907897949
    Iter #1809408:  Learning rate = 0.002398:   Batch Loss = 0.194443, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.602347731590271, Accuracy = 0.8842105269432068
    Iter #1809920:  Learning rate = 0.002398:   Batch Loss = 0.175133, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5571205019950867, Accuracy = 0.8979756832122803
    Iter #1810432:  Learning rate = 0.002398:   Batch Loss = 0.169088, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5511415004730225, Accuracy = 0.9024291634559631
    Iter #1810944:  Learning rate = 0.002398:   Batch Loss = 0.171145, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5428826808929443, Accuracy = 0.8995951414108276
    Iter #1811456:  Learning rate = 0.002398:   Batch Loss = 0.166329, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5397893190383911, Accuracy = 0.9016194343566895
    Iter #1811968:  Learning rate = 0.002398:   Batch Loss = 0.171815, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5319327116012573, Accuracy = 0.9020242691040039
    Iter #1812480:  Learning rate = 0.002398:   Batch Loss = 0.175831, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5270029306411743, Accuracy = 0.9064777493476868
    Iter #1812992:  Learning rate = 0.002398:   Batch Loss = 0.162960, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.528931200504303, Accuracy = 0.9048582911491394
    Iter #1813504:  Learning rate = 0.002398:   Batch Loss = 0.165409, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5263040065765381, Accuracy = 0.9032388925552368
    Iter #1814016:  Learning rate = 0.002398:   Batch Loss = 0.162390, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5263341069221497, Accuracy = 0.9036437273025513
    Iter #1814528:  Learning rate = 0.002398:   Batch Loss = 0.160991, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5273681879043579, Accuracy = 0.9068825840950012
    Iter #1815040:  Learning rate = 0.002398:   Batch Loss = 0.159883, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5240421891212463, Accuracy = 0.9064777493476868
    Iter #1815552:  Learning rate = 0.002398:   Batch Loss = 0.157830, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5200906991958618, Accuracy = 0.9093117117881775
    Iter #1816064:  Learning rate = 0.002398:   Batch Loss = 0.158154, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5181938409805298, Accuracy = 0.9093117117881775
    Iter #1816576:  Learning rate = 0.002398:   Batch Loss = 0.159524, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5187949538230896, Accuracy = 0.9093117117881775
    Iter #1817088:  Learning rate = 0.002398:   Batch Loss = 0.157124, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5182914137840271, Accuracy = 0.9101214408874512
    Iter #1817600:  Learning rate = 0.002398:   Batch Loss = 0.155193, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5218376517295837, Accuracy = 0.9056680202484131
    Iter #1818112:  Learning rate = 0.002398:   Batch Loss = 0.153971, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5246462821960449, Accuracy = 0.9060728549957275
    Iter #1818624:  Learning rate = 0.002398:   Batch Loss = 0.153867, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5255534648895264, Accuracy = 0.9048582911491394
    Iter #1819136:  Learning rate = 0.002398:   Batch Loss = 0.153841, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5194091200828552, Accuracy = 0.9036437273025513
    Iter #1819648:  Learning rate = 0.002398:   Batch Loss = 0.154869, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5175125002861023, Accuracy = 0.9052631855010986
    Iter #1820160:  Learning rate = 0.002398:   Batch Loss = 0.154905, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5161644220352173, Accuracy = 0.9097166061401367
    Iter #1820672:  Learning rate = 0.002398:   Batch Loss = 0.151891, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5151866674423218, Accuracy = 0.9101214408874512
    Iter #1821184:  Learning rate = 0.002398:   Batch Loss = 0.154686, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5151244401931763, Accuracy = 0.9076923131942749
    Iter #1821696:  Learning rate = 0.002398:   Batch Loss = 0.150612, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5119237303733826, Accuracy = 0.9076923131942749
    Iter #1822208:  Learning rate = 0.002398:   Batch Loss = 0.151177, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5117071866989136, Accuracy = 0.9056680202484131
    Iter #1822720:  Learning rate = 0.002398:   Batch Loss = 0.151098, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5127546787261963, Accuracy = 0.9056680202484131
    Iter #1823232:  Learning rate = 0.002398:   Batch Loss = 0.150574, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5139980316162109, Accuracy = 0.9060728549957275
    Iter #1823744:  Learning rate = 0.002398:   Batch Loss = 0.149704, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5161943435668945, Accuracy = 0.9056680202484131
    Iter #1824256:  Learning rate = 0.002398:   Batch Loss = 0.148237, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5203977823257446, Accuracy = 0.9008097052574158
    Iter #1824768:  Learning rate = 0.002398:   Batch Loss = 0.148604, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5172191858291626, Accuracy = 0.904453456401825
    Iter #1825280:  Learning rate = 0.002398:   Batch Loss = 0.148374, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5119345188140869, Accuracy = 0.9068825840950012
    Iter #1825792:  Learning rate = 0.002398:   Batch Loss = 0.146759, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5115047693252563, Accuracy = 0.908906877040863
    Iter #1826304:  Learning rate = 0.002398:   Batch Loss = 0.147380, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.513088047504425, Accuracy = 0.9068825840950012
    Iter #1826816:  Learning rate = 0.002398:   Batch Loss = 0.146945, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5125694274902344, Accuracy = 0.9076923131942749
    Iter #1827328:  Learning rate = 0.002398:   Batch Loss = 0.147824, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5169845819473267, Accuracy = 0.9056680202484131
    Iter #1827840:  Learning rate = 0.002398:   Batch Loss = 0.147973, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5153188109397888, Accuracy = 0.904453456401825
    Iter #1828352:  Learning rate = 0.002398:   Batch Loss = 0.145260, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5103633999824524, Accuracy = 0.9080971479415894
    Iter #1828864:  Learning rate = 0.002398:   Batch Loss = 0.146208, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5081342458724976, Accuracy = 0.9085020422935486
    Iter #1829376:  Learning rate = 0.002398:   Batch Loss = 0.146450, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5077972412109375, Accuracy = 0.9105263352394104
    Iter #1829888:  Learning rate = 0.002398:   Batch Loss = 0.143885, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5139824151992798, Accuracy = 0.9020242691040039
    Iter #1830400:  Learning rate = 0.002398:   Batch Loss = 0.144651, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.515413224697113, Accuracy = 0.8999999761581421
    Iter #1830912:  Learning rate = 0.002398:   Batch Loss = 0.142787, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5087203979492188, Accuracy = 0.9072874784469604
    Iter #1831424:  Learning rate = 0.002398:   Batch Loss = 0.143578, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5071606040000916, Accuracy = 0.9052631855010986
    Iter #1831936:  Learning rate = 0.002398:   Batch Loss = 0.143738, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.51519775390625, Accuracy = 0.9032388925552368
    Iter #1832448:  Learning rate = 0.002398:   Batch Loss = 0.141256, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5137678384780884, Accuracy = 0.9056680202484131
    Iter #1832960:  Learning rate = 0.002398:   Batch Loss = 0.143060, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5148301720619202, Accuracy = 0.9048582911491394
    Iter #1833472:  Learning rate = 0.002398:   Batch Loss = 0.144138, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5133612751960754, Accuracy = 0.904453456401825
    Iter #1833984:  Learning rate = 0.002398:   Batch Loss = 0.142682, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5117481350898743, Accuracy = 0.9064777493476868
    Iter #1834496:  Learning rate = 0.002398:   Batch Loss = 0.141835, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.511161208152771, Accuracy = 0.9060728549957275
    Iter #1835008:  Learning rate = 0.002398:   Batch Loss = 0.141635, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5136368870735168, Accuracy = 0.9048582911491394
    Iter #1835520:  Learning rate = 0.002398:   Batch Loss = 0.143772, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5156375765800476, Accuracy = 0.9032388925552368
    Iter #1836032:  Learning rate = 0.002398:   Batch Loss = 0.143440, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5146588087081909, Accuracy = 0.904453456401825
    Iter #1836544:  Learning rate = 0.002398:   Batch Loss = 0.142339, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5160012245178223, Accuracy = 0.9024291634559631
    Iter #1837056:  Learning rate = 0.002398:   Batch Loss = 0.142469, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5147183537483215, Accuracy = 0.9020242691040039
    Iter #1837568:  Learning rate = 0.002398:   Batch Loss = 0.140798, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5184446573257446, Accuracy = 0.8991903066635132
    Iter #1838080:  Learning rate = 0.002398:   Batch Loss = 0.141534, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5177572965621948, Accuracy = 0.9060728549957275
    Iter #1838592:  Learning rate = 0.002398:   Batch Loss = 0.140422, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5178790092468262, Accuracy = 0.9032388925552368
    Iter #1839104:  Learning rate = 0.002398:   Batch Loss = 0.139800, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5168430209159851, Accuracy = 0.8979756832122803
    Iter #1839616:  Learning rate = 0.002398:   Batch Loss = 0.139377, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5111562609672546, Accuracy = 0.9024291634559631
    Iter #1840128:  Learning rate = 0.002398:   Batch Loss = 0.139660, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5078692436218262, Accuracy = 0.9036437273025513
    Iter #1840640:  Learning rate = 0.002398:   Batch Loss = 0.142825, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5134038329124451, Accuracy = 0.9028339982032776
    Iter #1841152:  Learning rate = 0.002398:   Batch Loss = 0.137791, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5134627223014832, Accuracy = 0.9032388925552368
    Iter #1841664:  Learning rate = 0.002398:   Batch Loss = 0.136100, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5126492977142334, Accuracy = 0.9004048705101013
    Iter #1842176:  Learning rate = 0.002398:   Batch Loss = 0.138573, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.517073929309845, Accuracy = 0.8967611193656921
    Iter #1842688:  Learning rate = 0.002398:   Batch Loss = 0.138207, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5188273191452026, Accuracy = 0.8963562846183777
    Iter #1843200:  Learning rate = 0.002398:   Batch Loss = 0.137293, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5169466137886047, Accuracy = 0.901214599609375
    Iter #1843712:  Learning rate = 0.002398:   Batch Loss = 0.135346, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.510837972164154, Accuracy = 0.8991903066635132
    Iter #1844224:  Learning rate = 0.002398:   Batch Loss = 0.137500, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5136830806732178, Accuracy = 0.8971660137176514
    Iter #1844736:  Learning rate = 0.002398:   Batch Loss = 0.138187, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5191267728805542, Accuracy = 0.8963562846183777
    Iter #1845248:  Learning rate = 0.002398:   Batch Loss = 0.137087, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5128747224807739, Accuracy = 0.8979756832122803
    Iter #1845760:  Learning rate = 0.002398:   Batch Loss = 0.136042, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5151256918907166, Accuracy = 0.8959513902664185
    Iter #1846272:  Learning rate = 0.002398:   Batch Loss = 0.138570, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5176923274993896, Accuracy = 0.8963562846183777
    Iter #1846784:  Learning rate = 0.002398:   Batch Loss = 0.135769, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5189803242683411, Accuracy = 0.8943319916725159
    Iter #1847296:  Learning rate = 0.002398:   Batch Loss = 0.135503, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5233258605003357, Accuracy = 0.8963562846183777
    Iter #1847808:  Learning rate = 0.002398:   Batch Loss = 0.134928, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5198255181312561, Accuracy = 0.8959513902664185
    Iter #1848320:  Learning rate = 0.002398:   Batch Loss = 0.134252, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5229820013046265, Accuracy = 0.8939270973205566
    Iter #1848832:  Learning rate = 0.002398:   Batch Loss = 0.135455, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5194575786590576, Accuracy = 0.8943319916725159
    Iter #1849344:  Learning rate = 0.002398:   Batch Loss = 0.140248, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5160195827484131, Accuracy = 0.9008097052574158
    Iter #1849856:  Learning rate = 0.002398:   Batch Loss = 0.136626, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5299674868583679, Accuracy = 0.8939270973205566
    Iter #1850368:  Learning rate = 0.002398:   Batch Loss = 0.134429, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5206841230392456, Accuracy = 0.892307698726654
    Iter #1850880:  Learning rate = 0.002398:   Batch Loss = 0.135246, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5118384957313538, Accuracy = 0.9020242691040039
    Iter #1851392:  Learning rate = 0.002398:   Batch Loss = 0.133380, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5191470980644226, Accuracy = 0.898785412311554
    Iter #1851904:  Learning rate = 0.002398:   Batch Loss = 0.131996, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.528194785118103, Accuracy = 0.8902834057807922
    Iter #1852416:  Learning rate = 0.002398:   Batch Loss = 0.132183, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5284386873245239, Accuracy = 0.8947368264198303
    Iter #1852928:  Learning rate = 0.002398:   Batch Loss = 0.131643, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5276787877082825, Accuracy = 0.8947368264198303
    Iter #1853440:  Learning rate = 0.002398:   Batch Loss = 0.133052, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5249647498130798, Accuracy = 0.8919028043746948
    Iter #1853952:  Learning rate = 0.002398:   Batch Loss = 0.133445, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5180948972702026, Accuracy = 0.8971660137176514
    Iter #1854464:  Learning rate = 0.002398:   Batch Loss = 0.133915, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5190408229827881, Accuracy = 0.8935222625732422
    Iter #1854976:  Learning rate = 0.002398:   Batch Loss = 0.131407, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.518043577671051, Accuracy = 0.8971660137176514
    Iter #1855488:  Learning rate = 0.002398:   Batch Loss = 0.133983, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.516862154006958, Accuracy = 0.8991903066635132
    Iter #1856000:  Learning rate = 0.002398:   Batch Loss = 0.133546, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5223388671875, Accuracy = 0.8935222625732422
    Iter #1856512:  Learning rate = 0.002398:   Batch Loss = 0.131143, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5248004794120789, Accuracy = 0.8919028043746948
    Iter #1857024:  Learning rate = 0.002398:   Batch Loss = 0.130821, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5271077752113342, Accuracy = 0.8866396546363831
    Iter #1857536:  Learning rate = 0.002398:   Batch Loss = 0.130440, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5250492691993713, Accuracy = 0.8910931348800659
    Iter #1858048:  Learning rate = 0.002398:   Batch Loss = 0.130016, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5212854743003845, Accuracy = 0.8959513902664185
    Iter #1858560:  Learning rate = 0.002398:   Batch Loss = 0.131350, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5223116874694824, Accuracy = 0.8906882405281067
    Iter #1859072:  Learning rate = 0.002398:   Batch Loss = 0.130101, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5245656371116638, Accuracy = 0.8906882405281067
    Iter #1859584:  Learning rate = 0.002398:   Batch Loss = 0.132000, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5302464962005615, Accuracy = 0.8886639475822449
    Iter #1860096:  Learning rate = 0.002398:   Batch Loss = 0.129396, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5259806513786316, Accuracy = 0.8874493837356567
    Iter #1860608:  Learning rate = 0.002398:   Batch Loss = 0.130258, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5370194315910339, Accuracy = 0.8870445489883423
    Iter #1861120:  Learning rate = 0.002398:   Batch Loss = 0.130371, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5288518667221069, Accuracy = 0.8894736766815186
    Iter #1861632:  Learning rate = 0.002398:   Batch Loss = 0.133170, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.531758189201355, Accuracy = 0.8931174278259277
    Iter #1862144:  Learning rate = 0.002398:   Batch Loss = 0.132621, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5337041616439819, Accuracy = 0.8866396546363831
    Iter #1862656:  Learning rate = 0.002398:   Batch Loss = 0.128632, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.53662109375, Accuracy = 0.8882591128349304
    Iter #1863168:  Learning rate = 0.002398:   Batch Loss = 0.126465, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5302625894546509, Accuracy = 0.8874493837356567
    Iter #1863680:  Learning rate = 0.002398:   Batch Loss = 0.130717, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5347873568534851, Accuracy = 0.8886639475822449
    Iter #1864192:  Learning rate = 0.002398:   Batch Loss = 0.127942, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5256112217903137, Accuracy = 0.8902834057807922
    Iter #1864704:  Learning rate = 0.002398:   Batch Loss = 0.128982, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5209834575653076, Accuracy = 0.8939270973205566
    Iter #1865216:  Learning rate = 0.002398:   Batch Loss = 0.126540, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5226927399635315, Accuracy = 0.8919028043746948
    Iter #1865728:  Learning rate = 0.002398:   Batch Loss = 0.126180, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5244802236557007, Accuracy = 0.8927125334739685
    Iter #1866240:  Learning rate = 0.002398:   Batch Loss = 0.125845, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5294188261032104, Accuracy = 0.887854278087616
    Iter #1866752:  Learning rate = 0.002398:   Batch Loss = 0.125326, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5240673422813416, Accuracy = 0.8902834057807922
    Iter #1867264:  Learning rate = 0.002398:   Batch Loss = 0.130422, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5252585411071777, Accuracy = 0.8902834057807922
    Iter #1867776:  Learning rate = 0.002398:   Batch Loss = 0.128858, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5264642238616943, Accuracy = 0.8927125334739685
    Iter #1868288:  Learning rate = 0.002398:   Batch Loss = 0.124805, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5291581153869629, Accuracy = 0.8882591128349304
    Iter #1868800:  Learning rate = 0.002398:   Batch Loss = 0.128457, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5298938751220703, Accuracy = 0.8870445489883423
    Iter #1869312:  Learning rate = 0.002398:   Batch Loss = 0.125149, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5248936414718628, Accuracy = 0.8894736766815186
    Iter #1869824:  Learning rate = 0.002398:   Batch Loss = 0.124841, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5289770364761353, Accuracy = 0.8919028043746948
    Iter #1870336:  Learning rate = 0.002398:   Batch Loss = 0.127512, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5401772260665894, Accuracy = 0.8890688419342041
    Iter #1870848:  Learning rate = 0.002398:   Batch Loss = 0.126544, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5132979154586792, Accuracy = 0.8963562846183777
    Iter #1871360:  Learning rate = 0.002398:   Batch Loss = 0.127263, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.527275562286377, Accuracy = 0.8858299851417542
    Iter #1871872:  Learning rate = 0.002398:   Batch Loss = 0.126383, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.538368284702301, Accuracy = 0.8882591128349304
    Iter #1872384:  Learning rate = 0.002398:   Batch Loss = 0.132249, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5452954173088074, Accuracy = 0.8874493837356567
    Iter #1872896:  Learning rate = 0.002398:   Batch Loss = 0.178374, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5953256487846375, Accuracy = 0.8724696636199951
    Iter #1873408:  Learning rate = 0.002398:   Batch Loss = 0.332778, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6677899360656738, Accuracy = 0.8514170050621033
    Iter #1873920:  Learning rate = 0.002398:   Batch Loss = 0.334778, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7882653474807739, Accuracy = 0.8032388687133789
    Iter #1874432:  Learning rate = 0.002398:   Batch Loss = 0.415843, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7520690560340881, Accuracy = 0.8230769038200378
    Iter #1874944:  Learning rate = 0.002398:   Batch Loss = 0.255867, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8359812498092651, Accuracy = 0.8060728907585144
    Iter #1875456:  Learning rate = 0.002398:   Batch Loss = 0.482795, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7895963191986084, Accuracy = 0.8157894611358643
    Iter #1875968:  Learning rate = 0.002398:   Batch Loss = 0.624013, Accuracy = 0.84375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8399502038955688, Accuracy = 0.8024291396141052
    Iter #1876480:  Learning rate = 0.002398:   Batch Loss = 0.372592, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7835621237754822, Accuracy = 0.8121457695960999
    Iter #1876992:  Learning rate = 0.002398:   Batch Loss = 0.348428, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7324510812759399, Accuracy = 0.8372469544410706
    Iter #1877504:  Learning rate = 0.002398:   Batch Loss = 0.363048, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6728619933128357, Accuracy = 0.8611335754394531
    Iter #1878016:  Learning rate = 0.002398:   Batch Loss = 0.367508, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.736566424369812, Accuracy = 0.8380566835403442
    Iter #1878528:  Learning rate = 0.002398:   Batch Loss = 0.428860, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7655388712882996, Accuracy = 0.8295546770095825
    Iter #1879040:  Learning rate = 0.002398:   Batch Loss = 0.408409, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6969047784805298, Accuracy = 0.8570850491523743
    Iter #1879552:  Learning rate = 0.002398:   Batch Loss = 0.388320, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7076221108436584, Accuracy = 0.8595141768455505
    Iter #1880064:  Learning rate = 0.002398:   Batch Loss = 0.319772, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6566846370697021, Accuracy = 0.8615384697914124
    Iter #1880576:  Learning rate = 0.002398:   Batch Loss = 0.212359, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6613961458206177, Accuracy = 0.856680154800415
    Iter #1881088:  Learning rate = 0.002398:   Batch Loss = 0.234865, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6878073215484619, Accuracy = 0.8562753200531006
    Iter #1881600:  Learning rate = 0.002398:   Batch Loss = 0.202112, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6566702127456665, Accuracy = 0.8708502054214478
    Iter #1882112:  Learning rate = 0.002398:   Batch Loss = 0.273608, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6495265960693359, Accuracy = 0.8688259124755859
    Iter #1882624:  Learning rate = 0.002398:   Batch Loss = 0.228587, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7200443744659424, Accuracy = 0.8526315689086914
    Iter #1883136:  Learning rate = 0.002398:   Batch Loss = 0.311434, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6684095859527588, Accuracy = 0.8704453706741333
    Iter #1883648:  Learning rate = 0.002398:   Batch Loss = 0.253798, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6584826707839966, Accuracy = 0.8672064542770386
    Iter #1884160:  Learning rate = 0.002398:   Batch Loss = 0.240276, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5897356271743774, Accuracy = 0.8862348198890686
    Iter #1884672:  Learning rate = 0.002398:   Batch Loss = 0.239891, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6235426068305969, Accuracy = 0.8805667757987976
    Iter #1885184:  Learning rate = 0.002398:   Batch Loss = 0.328480, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.630317211151123, Accuracy = 0.8736842274665833
    Iter #1885696:  Learning rate = 0.002398:   Batch Loss = 0.192591, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5999823808670044, Accuracy = 0.8825910687446594
    Iter #1886208:  Learning rate = 0.002398:   Batch Loss = 0.237091, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6373190879821777, Accuracy = 0.876518189907074
    Iter #1886720:  Learning rate = 0.002398:   Batch Loss = 0.209025, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6415615677833557, Accuracy = 0.8692307472229004
    Iter #1887232:  Learning rate = 0.002398:   Batch Loss = 0.197457, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6559897661209106, Accuracy = 0.8696356415748596
    Iter #1887744:  Learning rate = 0.002398:   Batch Loss = 0.217792, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6452588438987732, Accuracy = 0.8797571063041687
    Iter #1888256:  Learning rate = 0.002398:   Batch Loss = 0.185017, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6190198659896851, Accuracy = 0.8809716701507568
    Iter #1888768:  Learning rate = 0.002398:   Batch Loss = 0.196043, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5982979536056519, Accuracy = 0.8882591128349304
    Iter #1889280:  Learning rate = 0.002398:   Batch Loss = 0.176698, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5765675902366638, Accuracy = 0.8947368264198303
    Iter #1889792:  Learning rate = 0.002398:   Batch Loss = 0.180815, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5769968628883362, Accuracy = 0.892307698726654
    Iter #1890304:  Learning rate = 0.002398:   Batch Loss = 0.179382, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5810665488243103, Accuracy = 0.8874493837356567
    Iter #1890816:  Learning rate = 0.002398:   Batch Loss = 0.171926, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5736886262893677, Accuracy = 0.8890688419342041
    Iter #1891328:  Learning rate = 0.002398:   Batch Loss = 0.173057, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5641365051269531, Accuracy = 0.8902834057807922
    Iter #1891840:  Learning rate = 0.002398:   Batch Loss = 0.175216, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5654892921447754, Accuracy = 0.8874493837356567
    Iter #1892352:  Learning rate = 0.002398:   Batch Loss = 0.183243, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5720139741897583, Accuracy = 0.8906882405281067
    Iter #1892864:  Learning rate = 0.002398:   Batch Loss = 0.186535, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5922052264213562, Accuracy = 0.8894736766815186
    Iter #1893376:  Learning rate = 0.002398:   Batch Loss = 0.171187, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6025928258895874, Accuracy = 0.8801619410514832
    Iter #1893888:  Learning rate = 0.002398:   Batch Loss = 0.181322, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5947829484939575, Accuracy = 0.8805667757987976
    Iter #1894400:  Learning rate = 0.002398:   Batch Loss = 0.173890, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6031821966171265, Accuracy = 0.8813765048980713
    Iter #1894912:  Learning rate = 0.002398:   Batch Loss = 0.176782, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5666005611419678, Accuracy = 0.8931174278259277
    Iter #1895424:  Learning rate = 0.002398:   Batch Loss = 0.169939, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5596054792404175, Accuracy = 0.8886639475822449
    Iter #1895936:  Learning rate = 0.002398:   Batch Loss = 0.192248, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5922952890396118, Accuracy = 0.8910931348800659
    Iter #1896448:  Learning rate = 0.002398:   Batch Loss = 0.176881, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6137294769287109, Accuracy = 0.878947377204895
    Iter #1896960:  Learning rate = 0.002398:   Batch Loss = 0.178082, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5587226748466492, Accuracy = 0.8914979696273804
    Iter #1897472:  Learning rate = 0.002398:   Batch Loss = 0.307712, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5701936483383179, Accuracy = 0.8842105269432068
    Iter #1897984:  Learning rate = 0.002398:   Batch Loss = 0.237448, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5511625409126282, Accuracy = 0.8931174278259277
    Iter #1898496:  Learning rate = 0.002398:   Batch Loss = 0.196536, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6024132966995239, Accuracy = 0.8785424828529358
    Iter #1899008:  Learning rate = 0.002398:   Batch Loss = 0.242096, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.633568525314331, Accuracy = 0.8684210777282715
    Iter #1899520:  Learning rate = 0.002398:   Batch Loss = 0.188057, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5744483470916748, Accuracy = 0.8886639475822449
    Iter #1900032:  Learning rate = 0.002302:   Batch Loss = 0.230501, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5740848183631897, Accuracy = 0.8850202560424805
    Iter #1900544:  Learning rate = 0.002302:   Batch Loss = 0.182339, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5868597030639648, Accuracy = 0.8834007978439331
    Iter #1901056:  Learning rate = 0.002302:   Batch Loss = 0.174170, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5586020350456238, Accuracy = 0.8939270973205566
    Iter #1901568:  Learning rate = 0.002302:   Batch Loss = 0.180845, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5733452439308167, Accuracy = 0.8902834057807922
    Iter #1902080:  Learning rate = 0.002302:   Batch Loss = 0.173318, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5521572828292847, Accuracy = 0.895546555519104
    Iter #1902592:  Learning rate = 0.002302:   Batch Loss = 0.175297, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5299643278121948, Accuracy = 0.9020242691040039
    Iter #1903104:  Learning rate = 0.002302:   Batch Loss = 0.162866, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5296258926391602, Accuracy = 0.9020242691040039
    Iter #1903616:  Learning rate = 0.002302:   Batch Loss = 0.165104, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.521481990814209, Accuracy = 0.901214599609375
    Iter #1904128:  Learning rate = 0.002302:   Batch Loss = 0.163910, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5215970277786255, Accuracy = 0.9016194343566895
    Iter #1904640:  Learning rate = 0.002302:   Batch Loss = 0.166265, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5251364707946777, Accuracy = 0.8995951414108276
    Iter #1905152:  Learning rate = 0.002302:   Batch Loss = 0.163083, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5125808715820312, Accuracy = 0.9048582911491394
    Iter #1905664:  Learning rate = 0.002302:   Batch Loss = 0.163135, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5096367001533508, Accuracy = 0.9032388925552368
    Iter #1906176:  Learning rate = 0.002302:   Batch Loss = 0.160672, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.510002851486206, Accuracy = 0.9024291634559631
    Iter #1906688:  Learning rate = 0.002302:   Batch Loss = 0.156621, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5100826621055603, Accuracy = 0.9016194343566895
    Iter #1907200:  Learning rate = 0.002302:   Batch Loss = 0.156611, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5069446563720703, Accuracy = 0.9036437273025513
    Iter #1907712:  Learning rate = 0.002302:   Batch Loss = 0.155245, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5037503242492676, Accuracy = 0.9040485620498657
    Iter #1908224:  Learning rate = 0.002302:   Batch Loss = 0.159065, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.503817081451416, Accuracy = 0.9024291634559631
    Iter #1908736:  Learning rate = 0.002302:   Batch Loss = 0.157154, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5050311088562012, Accuracy = 0.9028339982032776
    Iter #1909248:  Learning rate = 0.002302:   Batch Loss = 0.154314, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5055170655250549, Accuracy = 0.9024291634559631
    Iter #1909760:  Learning rate = 0.002302:   Batch Loss = 0.154768, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5017122626304626, Accuracy = 0.9004048705101013
    Iter #1910272:  Learning rate = 0.002302:   Batch Loss = 0.154071, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4984147548675537, Accuracy = 0.9024291634559631
    Iter #1910784:  Learning rate = 0.002302:   Batch Loss = 0.153831, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4983672499656677, Accuracy = 0.9024291634559631
    Iter #1911296:  Learning rate = 0.002302:   Batch Loss = 0.152401, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49947965145111084, Accuracy = 0.9016194343566895
    Iter #1911808:  Learning rate = 0.002302:   Batch Loss = 0.153299, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5007631182670593, Accuracy = 0.9020242691040039
    Iter #1912320:  Learning rate = 0.002302:   Batch Loss = 0.150744, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5017582774162292, Accuracy = 0.9024291634559631
    Iter #1912832:  Learning rate = 0.002302:   Batch Loss = 0.152525, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.502231240272522, Accuracy = 0.9020242691040039
    Iter #1913344:  Learning rate = 0.002302:   Batch Loss = 0.152323, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4981732666492462, Accuracy = 0.9032388925552368
    Iter #1913856:  Learning rate = 0.002302:   Batch Loss = 0.150534, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49490785598754883, Accuracy = 0.9004048705101013
    Iter #1914368:  Learning rate = 0.002302:   Batch Loss = 0.151928, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4962999224662781, Accuracy = 0.904453456401825
    Iter #1914880:  Learning rate = 0.002302:   Batch Loss = 0.149659, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.499218225479126, Accuracy = 0.9036437273025513
    Iter #1915392:  Learning rate = 0.002302:   Batch Loss = 0.149885, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49747297167778015, Accuracy = 0.901214599609375
    Iter #1915904:  Learning rate = 0.002302:   Batch Loss = 0.151878, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.493686318397522, Accuracy = 0.9004048705101013
    Iter #1916416:  Learning rate = 0.002302:   Batch Loss = 0.148261, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4905964732170105, Accuracy = 0.9016194343566895
    Iter #1916928:  Learning rate = 0.002302:   Batch Loss = 0.149158, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49266040325164795, Accuracy = 0.9036437273025513
    Iter #1917440:  Learning rate = 0.002302:   Batch Loss = 0.147740, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4929565191268921, Accuracy = 0.9040485620498657
    Iter #1917952:  Learning rate = 0.002302:   Batch Loss = 0.147521, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49292829632759094, Accuracy = 0.901214599609375
    Iter #1918464:  Learning rate = 0.002302:   Batch Loss = 0.149347, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4915122389793396, Accuracy = 0.9024291634559631
    Iter #1918976:  Learning rate = 0.002302:   Batch Loss = 0.148636, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49167752265930176, Accuracy = 0.904453456401825
    Iter #1919488:  Learning rate = 0.002302:   Batch Loss = 0.148113, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4956473112106323, Accuracy = 0.9036437273025513
    Iter #1920000:  Learning rate = 0.002302:   Batch Loss = 0.144012, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5009949207305908, Accuracy = 0.9008097052574158
    Iter #1920512:  Learning rate = 0.002302:   Batch Loss = 0.148443, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.501027524471283, Accuracy = 0.898785412311554
    Iter #1921024:  Learning rate = 0.002302:   Batch Loss = 0.147484, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49870187044143677, Accuracy = 0.901214599609375
    Iter #1921536:  Learning rate = 0.002302:   Batch Loss = 0.149029, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4933251142501831, Accuracy = 0.9028339982032776
    Iter #1922048:  Learning rate = 0.002302:   Batch Loss = 0.145192, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4913841485977173, Accuracy = 0.9024291634559631
    Iter #1922560:  Learning rate = 0.002302:   Batch Loss = 0.144219, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49328893423080444, Accuracy = 0.9036437273025513
    Iter #1923072:  Learning rate = 0.002302:   Batch Loss = 0.145128, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4945797920227051, Accuracy = 0.9020242691040039
    Iter #1923584:  Learning rate = 0.002302:   Batch Loss = 0.142994, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.494193971157074, Accuracy = 0.8999999761581421
    Iter #1924096:  Learning rate = 0.002302:   Batch Loss = 0.143163, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49569031596183777, Accuracy = 0.9016194343566895
    Iter #1924608:  Learning rate = 0.002302:   Batch Loss = 0.145197, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49722471833229065, Accuracy = 0.9016194343566895
    Iter #1925120:  Learning rate = 0.002302:   Batch Loss = 0.144328, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49862682819366455, Accuracy = 0.9020242691040039
    Iter #1925632:  Learning rate = 0.002302:   Batch Loss = 0.142745, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4971354603767395, Accuracy = 0.9052631855010986
    Iter #1926144:  Learning rate = 0.002302:   Batch Loss = 0.142846, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49773165583610535, Accuracy = 0.9036437273025513
    Iter #1926656:  Learning rate = 0.002302:   Batch Loss = 0.143139, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49819910526275635, Accuracy = 0.901214599609375
    Iter #1927168:  Learning rate = 0.002302:   Batch Loss = 0.142064, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5014533996582031, Accuracy = 0.898785412311554
    Iter #1927680:  Learning rate = 0.002302:   Batch Loss = 0.142775, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4975320100784302, Accuracy = 0.9024291634559631
    Iter #1928192:  Learning rate = 0.002302:   Batch Loss = 0.144247, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5035898685455322, Accuracy = 0.9004048705101013
    Iter #1928704:  Learning rate = 0.002302:   Batch Loss = 0.139536, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4967915713787079, Accuracy = 0.8999999761581421
    Iter #1929216:  Learning rate = 0.002302:   Batch Loss = 0.143993, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49619054794311523, Accuracy = 0.9024291634559631
    Iter #1929728:  Learning rate = 0.002302:   Batch Loss = 0.142428, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49963316321372986, Accuracy = 0.8999999761581421
    Iter #1930240:  Learning rate = 0.002302:   Batch Loss = 0.141738, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4976406693458557, Accuracy = 0.9008097052574158
    Iter #1930752:  Learning rate = 0.002302:   Batch Loss = 0.141447, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.496952623128891, Accuracy = 0.9008097052574158
    Iter #1931264:  Learning rate = 0.002302:   Batch Loss = 0.139294, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49640294909477234, Accuracy = 0.9004048705101013
    Iter #1931776:  Learning rate = 0.002302:   Batch Loss = 0.140807, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49623918533325195, Accuracy = 0.8975708484649658
    Iter #1932288:  Learning rate = 0.002302:   Batch Loss = 0.138764, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49521493911743164, Accuracy = 0.9008097052574158
    Iter #1932800:  Learning rate = 0.002302:   Batch Loss = 0.139347, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49784326553344727, Accuracy = 0.8999999761581421
    Iter #1933312:  Learning rate = 0.002302:   Batch Loss = 0.140223, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5007092952728271, Accuracy = 0.8979756832122803
    Iter #1933824:  Learning rate = 0.002302:   Batch Loss = 0.139816, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5002729892730713, Accuracy = 0.8991903066635132
    Iter #1934336:  Learning rate = 0.002302:   Batch Loss = 0.137188, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.503733217716217, Accuracy = 0.898785412311554
    Iter #1934848:  Learning rate = 0.002302:   Batch Loss = 0.137717, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5036757588386536, Accuracy = 0.898785412311554
    Iter #1935360:  Learning rate = 0.002302:   Batch Loss = 0.143704, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.503739595413208, Accuracy = 0.8995951414108276
    Iter #1935872:  Learning rate = 0.002302:   Batch Loss = 0.138608, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5035802721977234, Accuracy = 0.9004048705101013
    Iter #1936384:  Learning rate = 0.002302:   Batch Loss = 0.139451, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.503409206867218, Accuracy = 0.898785412311554
    Iter #1936896:  Learning rate = 0.002302:   Batch Loss = 0.134979, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5003619194030762, Accuracy = 0.8983805775642395
    Iter #1937408:  Learning rate = 0.002302:   Batch Loss = 0.139890, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5023655295372009, Accuracy = 0.8991903066635132
    Iter #1937920:  Learning rate = 0.002302:   Batch Loss = 0.135334, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5076724290847778, Accuracy = 0.8975708484649658
    Iter #1938432:  Learning rate = 0.002302:   Batch Loss = 0.135498, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5081636309623718, Accuracy = 0.8983805775642395
    Iter #1938944:  Learning rate = 0.002302:   Batch Loss = 0.134353, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5070351362228394, Accuracy = 0.8979756832122803
    Iter #1939456:  Learning rate = 0.002302:   Batch Loss = 0.135927, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.508935809135437, Accuracy = 0.8983805775642395
    Iter #1939968:  Learning rate = 0.002302:   Batch Loss = 0.136802, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5080289244651794, Accuracy = 0.8963562846183777
    Iter #1940480:  Learning rate = 0.002302:   Batch Loss = 0.133774, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5070311427116394, Accuracy = 0.8971660137176514
    Iter #1940992:  Learning rate = 0.002302:   Batch Loss = 0.135225, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5069879293441772, Accuracy = 0.8971660137176514
    Iter #1941504:  Learning rate = 0.002302:   Batch Loss = 0.133579, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5062260627746582, Accuracy = 0.8983805775642395
    Iter #1942016:  Learning rate = 0.002302:   Batch Loss = 0.140852, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5059131383895874, Accuracy = 0.8963562846183777
    Iter #1942528:  Learning rate = 0.002302:   Batch Loss = 0.138365, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5125923752784729, Accuracy = 0.8979756832122803
    Iter #1943040:  Learning rate = 0.002302:   Batch Loss = 0.134643, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5011385083198547, Accuracy = 0.8979756832122803
    Iter #1943552:  Learning rate = 0.002302:   Batch Loss = 0.137130, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5049575567245483, Accuracy = 0.895546555519104
    Iter #1944064:  Learning rate = 0.002302:   Batch Loss = 0.137776, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5320550203323364, Accuracy = 0.8906882405281067
    Iter #1944576:  Learning rate = 0.002302:   Batch Loss = 0.139683, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5345959067344666, Accuracy = 0.8943319916725159
    Iter #1945088:  Learning rate = 0.002302:   Batch Loss = 0.139678, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.619654655456543, Accuracy = 0.8676113486289978
    Iter #1945600:  Learning rate = 0.002302:   Batch Loss = 0.196093, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6255909204483032, Accuracy = 0.8700404763221741
    Iter #1946112:  Learning rate = 0.002302:   Batch Loss = 0.340683, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7910103797912598, Accuracy = 0.8206477761268616
    Iter #1946624:  Learning rate = 0.002302:   Batch Loss = 0.286807, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6615198850631714, Accuracy = 0.8542510271072388
    Iter #1947136:  Learning rate = 0.002302:   Batch Loss = 0.258464, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.689031183719635, Accuracy = 0.8412955403327942
    Iter #1947648:  Learning rate = 0.002302:   Batch Loss = 0.241754, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.682958722114563, Accuracy = 0.8554655909538269
    Iter #1948160:  Learning rate = 0.002302:   Batch Loss = 0.336648, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7746942639350891, Accuracy = 0.8190283179283142
    Iter #1948672:  Learning rate = 0.002302:   Batch Loss = 0.545407, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8240777254104614, Accuracy = 0.8004048466682434
    Iter #1949184:  Learning rate = 0.002302:   Batch Loss = 0.257652, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7716473937034607, Accuracy = 0.8255060911178589
    Iter #1949696:  Learning rate = 0.002302:   Batch Loss = 0.399379, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7455892562866211, Accuracy = 0.8376518487930298
    Iter #1950208:  Learning rate = 0.002302:   Batch Loss = 0.320332, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6508028507232666, Accuracy = 0.8659918904304504
    Iter #1950720:  Learning rate = 0.002302:   Batch Loss = 0.244744, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6680318117141724, Accuracy = 0.862348198890686
    Iter #1951232:  Learning rate = 0.002302:   Batch Loss = 0.281088, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6940290927886963, Accuracy = 0.8518218398094177
    Iter #1951744:  Learning rate = 0.002302:   Batch Loss = 0.236659, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6331890225410461, Accuracy = 0.8700404763221741
    Iter #1952256:  Learning rate = 0.002302:   Batch Loss = 0.287515, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6231303215026855, Accuracy = 0.876518189907074
    Iter #1952768:  Learning rate = 0.002302:   Batch Loss = 0.209748, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6060440540313721, Accuracy = 0.8757085204124451
    Iter #1953280:  Learning rate = 0.002302:   Batch Loss = 0.200840, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6427380442619324, Accuracy = 0.8773279190063477
    Iter #1953792:  Learning rate = 0.002302:   Batch Loss = 0.234155, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6631582975387573, Accuracy = 0.8716599345207214
    Iter #1954304:  Learning rate = 0.002302:   Batch Loss = 0.238284, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6022998094558716, Accuracy = 0.8846153616905212
    Iter #1954816:  Learning rate = 0.002302:   Batch Loss = 0.293833, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6125016808509827, Accuracy = 0.8813765048980713
    Iter #1955328:  Learning rate = 0.002302:   Batch Loss = 0.309525, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6069952249526978, Accuracy = 0.8813765048980713
    Iter #1955840:  Learning rate = 0.002302:   Batch Loss = 0.275595, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6359281539916992, Accuracy = 0.8720647692680359
    Iter #1956352:  Learning rate = 0.002302:   Batch Loss = 0.199452, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6693031191825867, Accuracy = 0.8631578683853149
    Iter #1956864:  Learning rate = 0.002302:   Batch Loss = 0.238064, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6430704593658447, Accuracy = 0.8712550401687622
    Iter #1957376:  Learning rate = 0.002302:   Batch Loss = 0.215820, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.628437340259552, Accuracy = 0.8797571063041687
    Iter #1957888:  Learning rate = 0.002302:   Batch Loss = 0.275481, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6608759164810181, Accuracy = 0.8700404763221741
    Iter #1958400:  Learning rate = 0.002302:   Batch Loss = 0.209453, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6666921377182007, Accuracy = 0.8631578683853149
    Iter #1958912:  Learning rate = 0.002302:   Batch Loss = 0.231309, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6216784119606018, Accuracy = 0.8748987913131714
    Iter #1959424:  Learning rate = 0.002302:   Batch Loss = 0.274191, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6348813772201538, Accuracy = 0.8740890622138977
    Iter #1959936:  Learning rate = 0.002302:   Batch Loss = 0.181786, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.591277539730072, Accuracy = 0.8797571063041687
    Iter #1960448:  Learning rate = 0.002302:   Batch Loss = 0.178252, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5639989972114563, Accuracy = 0.892307698726654
    Iter #1960960:  Learning rate = 0.002302:   Batch Loss = 0.187546, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.570188581943512, Accuracy = 0.8939270973205566
    Iter #1961472:  Learning rate = 0.002302:   Batch Loss = 0.173768, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5571399927139282, Accuracy = 0.8979756832122803
    Iter #1961984:  Learning rate = 0.002302:   Batch Loss = 0.175687, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5428830981254578, Accuracy = 0.901214599609375
    Iter #1962496:  Learning rate = 0.002302:   Batch Loss = 0.168297, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5334984660148621, Accuracy = 0.9024291634559631
    Iter #1963008:  Learning rate = 0.002302:   Batch Loss = 0.167316, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5351465940475464, Accuracy = 0.9028339982032776
    Iter #1963520:  Learning rate = 0.002302:   Batch Loss = 0.169315, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5347469449043274, Accuracy = 0.9028339982032776
    Iter #1964032:  Learning rate = 0.002302:   Batch Loss = 0.165632, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5348504781723022, Accuracy = 0.9052631855010986
    Iter #1964544:  Learning rate = 0.002302:   Batch Loss = 0.163294, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5367410778999329, Accuracy = 0.9016194343566895
    Iter #1965056:  Learning rate = 0.002302:   Batch Loss = 0.163021, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5246648192405701, Accuracy = 0.9048582911491394
    Iter #1965568:  Learning rate = 0.002302:   Batch Loss = 0.159338, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5240873694419861, Accuracy = 0.9028339982032776
    Iter #1966080:  Learning rate = 0.002302:   Batch Loss = 0.159431, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5189549922943115, Accuracy = 0.9028339982032776
    Iter #1966592:  Learning rate = 0.002302:   Batch Loss = 0.161317, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.521853506565094, Accuracy = 0.9008097052574158
    Iter #1967104:  Learning rate = 0.002302:   Batch Loss = 0.158534, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5251003503799438, Accuracy = 0.9032388925552368
    Iter #1967616:  Learning rate = 0.002302:   Batch Loss = 0.156515, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5256744623184204, Accuracy = 0.901214599609375
    Iter #1968128:  Learning rate = 0.002302:   Batch Loss = 0.160194, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5158833265304565, Accuracy = 0.904453456401825
    Iter #1968640:  Learning rate = 0.002302:   Batch Loss = 0.157325, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5121288895606995, Accuracy = 0.9048582911491394
    Iter #1969152:  Learning rate = 0.002302:   Batch Loss = 0.157282, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5110191106796265, Accuracy = 0.9060728549957275
    Iter #1969664:  Learning rate = 0.002302:   Batch Loss = 0.152783, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5123048424720764, Accuracy = 0.9052631855010986
    Iter #1970176:  Learning rate = 0.002302:   Batch Loss = 0.153965, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5172474384307861, Accuracy = 0.9024291634559631
    Iter #1970688:  Learning rate = 0.002302:   Batch Loss = 0.154269, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5139510631561279, Accuracy = 0.9040485620498657
    Iter #1971200:  Learning rate = 0.002302:   Batch Loss = 0.154977, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5096083879470825, Accuracy = 0.9028339982032776
    Iter #1971712:  Learning rate = 0.002302:   Batch Loss = 0.152276, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5101743936538696, Accuracy = 0.9040485620498657
    Iter #1972224:  Learning rate = 0.002302:   Batch Loss = 0.151755, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5115464925765991, Accuracy = 0.9048582911491394
    Iter #1972736:  Learning rate = 0.002302:   Batch Loss = 0.150558, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5131139755249023, Accuracy = 0.9040485620498657
    Iter #1973248:  Learning rate = 0.002302:   Batch Loss = 0.154257, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5128070116043091, Accuracy = 0.904453456401825
    Iter #1973760:  Learning rate = 0.002302:   Batch Loss = 0.152290, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5145159959793091, Accuracy = 0.9048582911491394
    Iter #1974272:  Learning rate = 0.002302:   Batch Loss = 0.149083, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5100919008255005, Accuracy = 0.9040485620498657
    Iter #1974784:  Learning rate = 0.002302:   Batch Loss = 0.152545, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5106520652770996, Accuracy = 0.9052631855010986
    Iter #1975296:  Learning rate = 0.002302:   Batch Loss = 0.151498, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.512227475643158, Accuracy = 0.9028339982032776
    Iter #1975808:  Learning rate = 0.002302:   Batch Loss = 0.148414, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5095022916793823, Accuracy = 0.9024291634559631
    Iter #1976320:  Learning rate = 0.002302:   Batch Loss = 0.148191, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5118448734283447, Accuracy = 0.9052631855010986
    Iter #1976832:  Learning rate = 0.002302:   Batch Loss = 0.148575, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5086840987205505, Accuracy = 0.9020242691040039
    Iter #1977344:  Learning rate = 0.002302:   Batch Loss = 0.145610, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5105941891670227, Accuracy = 0.9020242691040039
    Iter #1977856:  Learning rate = 0.002302:   Batch Loss = 0.145907, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5068575739860535, Accuracy = 0.904453456401825
    Iter #1978368:  Learning rate = 0.002302:   Batch Loss = 0.149076, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5101606845855713, Accuracy = 0.9032388925552368
    Iter #1978880:  Learning rate = 0.002302:   Batch Loss = 0.147812, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5143914818763733, Accuracy = 0.898785412311554
    Iter #1979392:  Learning rate = 0.002302:   Batch Loss = 0.147796, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5184088349342346, Accuracy = 0.898785412311554
    Iter #1979904:  Learning rate = 0.002302:   Batch Loss = 0.148292, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5133272409439087, Accuracy = 0.9004048705101013
    Iter #1980416:  Learning rate = 0.002302:   Batch Loss = 0.147439, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5084238052368164, Accuracy = 0.901214599609375
    Iter #1980928:  Learning rate = 0.002302:   Batch Loss = 0.143677, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5113304257392883, Accuracy = 0.9040485620498657
    Iter #1981440:  Learning rate = 0.002302:   Batch Loss = 0.144125, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.512010931968689, Accuracy = 0.8995951414108276
    Iter #1981952:  Learning rate = 0.002302:   Batch Loss = 0.144109, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5222107172012329, Accuracy = 0.8963562846183777
    Iter #1982464:  Learning rate = 0.002302:   Batch Loss = 0.142332, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5162496566772461, Accuracy = 0.8967611193656921
    Iter #1982976:  Learning rate = 0.002302:   Batch Loss = 0.142729, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5188254117965698, Accuracy = 0.8979756832122803
    Iter #1983488:  Learning rate = 0.002302:   Batch Loss = 0.142664, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5171193480491638, Accuracy = 0.901214599609375
    Iter #1984000:  Learning rate = 0.002302:   Batch Loss = 0.146071, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5124121904373169, Accuracy = 0.9028339982032776
    Iter #1984512:  Learning rate = 0.002302:   Batch Loss = 0.141518, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5164221525192261, Accuracy = 0.9016194343566895
    Iter #1985024:  Learning rate = 0.002302:   Batch Loss = 0.142745, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5209977030754089, Accuracy = 0.895546555519104
    Iter #1985536:  Learning rate = 0.002302:   Batch Loss = 0.141478, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5195221900939941, Accuracy = 0.898785412311554
    Iter #1986048:  Learning rate = 0.002302:   Batch Loss = 0.137887, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5184993743896484, Accuracy = 0.9008097052574158
    Iter #1986560:  Learning rate = 0.002302:   Batch Loss = 0.137410, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5255948305130005, Accuracy = 0.8959513902664185
    Iter #1987072:  Learning rate = 0.002302:   Batch Loss = 0.139443, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5220762491226196, Accuracy = 0.8951417207717896
    Iter #1987584:  Learning rate = 0.002302:   Batch Loss = 0.140750, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5264148712158203, Accuracy = 0.8914979696273804
    Iter #1988096:  Learning rate = 0.002302:   Batch Loss = 0.137800, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5234848260879517, Accuracy = 0.8947368264198303
    Iter #1988608:  Learning rate = 0.002302:   Batch Loss = 0.135525, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5206461548805237, Accuracy = 0.8979756832122803
    Iter #1989120:  Learning rate = 0.002302:   Batch Loss = 0.136549, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5222245454788208, Accuracy = 0.8963562846183777
    Iter #1989632:  Learning rate = 0.002302:   Batch Loss = 0.138223, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5242531299591064, Accuracy = 0.8943319916725159
    Iter #1990144:  Learning rate = 0.002302:   Batch Loss = 0.137302, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5145930051803589, Accuracy = 0.8983805775642395
    Iter #1990656:  Learning rate = 0.002302:   Batch Loss = 0.141159, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.51746666431427, Accuracy = 0.8983805775642395
    Iter #1991168:  Learning rate = 0.002302:   Batch Loss = 0.135025, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.520817756652832, Accuracy = 0.8967611193656921
    Iter #1991680:  Learning rate = 0.002302:   Batch Loss = 0.136130, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5226660370826721, Accuracy = 0.8947368264198303
    Iter #1992192:  Learning rate = 0.002302:   Batch Loss = 0.139067, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5182766318321228, Accuracy = 0.8979756832122803
    Iter #1992704:  Learning rate = 0.002302:   Batch Loss = 0.136037, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5162686109542847, Accuracy = 0.8971660137176514
    Iter #1993216:  Learning rate = 0.002302:   Batch Loss = 0.136865, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5170090198516846, Accuracy = 0.895546555519104
    Iter #1993728:  Learning rate = 0.002302:   Batch Loss = 0.135258, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5264136791229248, Accuracy = 0.8919028043746948
    Iter #1994240:  Learning rate = 0.002302:   Batch Loss = 0.133318, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5301063060760498, Accuracy = 0.887854278087616
    Iter #1994752:  Learning rate = 0.002302:   Batch Loss = 0.134950, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5208169221878052, Accuracy = 0.8931174278259277
    Iter #1995264:  Learning rate = 0.002302:   Batch Loss = 0.132798, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5159880518913269, Accuracy = 0.8943319916725159
    Iter #1995776:  Learning rate = 0.002302:   Batch Loss = 0.132886, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5209712386131287, Accuracy = 0.8931174278259277
    Iter #1996288:  Learning rate = 0.002302:   Batch Loss = 0.133524, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5193426012992859, Accuracy = 0.895546555519104
    Iter #1996800:  Learning rate = 0.002302:   Batch Loss = 0.134230, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5194739699363708, Accuracy = 0.8971660137176514
    Iter #1997312:  Learning rate = 0.002302:   Batch Loss = 0.132109, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5297355651855469, Accuracy = 0.8951417207717896
    Iter #1997824:  Learning rate = 0.002302:   Batch Loss = 0.130078, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.530376136302948, Accuracy = 0.8902834057807922
    Iter #1998336:  Learning rate = 0.002302:   Batch Loss = 0.132113, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5309764742851257, Accuracy = 0.8894736766815186
    Iter #1998848:  Learning rate = 0.002302:   Batch Loss = 0.130622, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5281125903129578, Accuracy = 0.8902834057807922
    Iter #1999360:  Learning rate = 0.002302:   Batch Loss = 0.133232, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5299252867698669, Accuracy = 0.8882591128349304
    Iter #1999872:  Learning rate = 0.002302:   Batch Loss = 0.132933, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5197304487228394, Accuracy = 0.8931174278259277
    Iter #2000384:  Learning rate = 0.002210:   Batch Loss = 0.132122, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5235018730163574, Accuracy = 0.8967611193656921
    Iter #2000896:  Learning rate = 0.002210:   Batch Loss = 0.133970, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5329317450523376, Accuracy = 0.8935222625732422
    Iter #2001408:  Learning rate = 0.002210:   Batch Loss = 0.129745, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5247758626937866, Accuracy = 0.8931174278259277
    Iter #2001920:  Learning rate = 0.002210:   Batch Loss = 0.131269, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5267281532287598, Accuracy = 0.8947368264198303
    Iter #2002432:  Learning rate = 0.002210:   Batch Loss = 0.130207, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.527816116809845, Accuracy = 0.8902834057807922
    Iter #2002944:  Learning rate = 0.002210:   Batch Loss = 0.131442, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5250754356384277, Accuracy = 0.8947368264198303
    Iter #2003456:  Learning rate = 0.002210:   Batch Loss = 0.132130, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5332993865013123, Accuracy = 0.8914979696273804
    Iter #2003968:  Learning rate = 0.002210:   Batch Loss = 0.129671, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5265594720840454, Accuracy = 0.8947368264198303
    Iter #2004480:  Learning rate = 0.002210:   Batch Loss = 0.128398, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5281205177307129, Accuracy = 0.8939270973205566
    Iter #2004992:  Learning rate = 0.002210:   Batch Loss = 0.128602, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5290147066116333, Accuracy = 0.8927125334739685
    Iter #2005504:  Learning rate = 0.002210:   Batch Loss = 0.133453, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5242965221405029, Accuracy = 0.8947368264198303
    Iter #2006016:  Learning rate = 0.002210:   Batch Loss = 0.127526, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5243522524833679, Accuracy = 0.8943319916725159
    Iter #2006528:  Learning rate = 0.002210:   Batch Loss = 0.129513, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5227009057998657, Accuracy = 0.8927125334739685
    Iter #2007040:  Learning rate = 0.002210:   Batch Loss = 0.127276, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5336877703666687, Accuracy = 0.8842105269432068
    Iter #2007552:  Learning rate = 0.002210:   Batch Loss = 0.128248, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5311300158500671, Accuracy = 0.8914979696273804
    Iter #2008064:  Learning rate = 0.002210:   Batch Loss = 0.126861, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5310067534446716, Accuracy = 0.8914979696273804
    Iter #2008576:  Learning rate = 0.002210:   Batch Loss = 0.127868, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5275869369506836, Accuracy = 0.8935222625732422
    Iter #2009088:  Learning rate = 0.002210:   Batch Loss = 0.129153, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5276683568954468, Accuracy = 0.8910931348800659
    Iter #2009600:  Learning rate = 0.002210:   Batch Loss = 0.128227, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5298234224319458, Accuracy = 0.8886639475822449
    Iter #2010112:  Learning rate = 0.002210:   Batch Loss = 0.127087, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5284602046012878, Accuracy = 0.8894736766815186
    Iter #2010624:  Learning rate = 0.002210:   Batch Loss = 0.127206, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5244124531745911, Accuracy = 0.8927125334739685
    Iter #2011136:  Learning rate = 0.002210:   Batch Loss = 0.126869, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5266449451446533, Accuracy = 0.8927125334739685
    Iter #2011648:  Learning rate = 0.002210:   Batch Loss = 0.130747, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5329419374465942, Accuracy = 0.8874493837356567
    Iter #2012160:  Learning rate = 0.002210:   Batch Loss = 0.126560, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5214071273803711, Accuracy = 0.8943319916725159
    Iter #2012672:  Learning rate = 0.002210:   Batch Loss = 0.126381, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5226778984069824, Accuracy = 0.8927125334739685
    Iter #2013184:  Learning rate = 0.002210:   Batch Loss = 0.126260, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5245376825332642, Accuracy = 0.8927125334739685
    Iter #2013696:  Learning rate = 0.002210:   Batch Loss = 0.125094, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5288559794425964, Accuracy = 0.8898785710334778
    Iter #2014208:  Learning rate = 0.002210:   Batch Loss = 0.126726, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5295432209968567, Accuracy = 0.8886639475822449
    Iter #2014720:  Learning rate = 0.002210:   Batch Loss = 0.127357, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5295259952545166, Accuracy = 0.887854278087616
    Iter #2015232:  Learning rate = 0.002210:   Batch Loss = 0.124034, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5260306000709534, Accuracy = 0.8947368264198303
    Iter #2015744:  Learning rate = 0.002210:   Batch Loss = 0.125299, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5295727849006653, Accuracy = 0.8890688419342041
    Iter #2016256:  Learning rate = 0.002210:   Batch Loss = 0.124192, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5265724062919617, Accuracy = 0.8914979696273804
    Iter #2016768:  Learning rate = 0.002210:   Batch Loss = 0.125454, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5259127020835876, Accuracy = 0.8919028043746948
    Iter #2017280:  Learning rate = 0.002210:   Batch Loss = 0.127380, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5275213122367859, Accuracy = 0.8898785710334778
    Iter #2017792:  Learning rate = 0.002210:   Batch Loss = 0.126427, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5269687175750732, Accuracy = 0.8902834057807922
    Iter #2018304:  Learning rate = 0.002210:   Batch Loss = 0.126092, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5303536653518677, Accuracy = 0.8874493837356567
    Iter #2018816:  Learning rate = 0.002210:   Batch Loss = 0.125428, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5352024435997009, Accuracy = 0.8906882405281067
    Iter #2019328:  Learning rate = 0.002210:   Batch Loss = 0.124151, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5279276967048645, Accuracy = 0.8898785710334778
    Iter #2019840:  Learning rate = 0.002210:   Batch Loss = 0.125121, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5330297946929932, Accuracy = 0.8902834057807922
    Iter #2020352:  Learning rate = 0.002210:   Batch Loss = 0.126637, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5232600569725037, Accuracy = 0.8951417207717896
    Iter #2020864:  Learning rate = 0.002210:   Batch Loss = 0.124259, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5275245904922485, Accuracy = 0.8902834057807922
    Iter #2021376:  Learning rate = 0.002210:   Batch Loss = 0.126030, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5264054536819458, Accuracy = 0.892307698726654
    Iter #2021888:  Learning rate = 0.002210:   Batch Loss = 0.123989, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5271750092506409, Accuracy = 0.8914979696273804
    Iter #2022400:  Learning rate = 0.002210:   Batch Loss = 0.123337, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5237899422645569, Accuracy = 0.8902834057807922
    Iter #2022912:  Learning rate = 0.002210:   Batch Loss = 0.126665, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5287960171699524, Accuracy = 0.895546555519104
    Iter #2023424:  Learning rate = 0.002210:   Batch Loss = 0.128112, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5294470191001892, Accuracy = 0.887854278087616
    Iter #2023936:  Learning rate = 0.002210:   Batch Loss = 0.125060, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5325167775154114, Accuracy = 0.8914979696273804
    Iter #2024448:  Learning rate = 0.002210:   Batch Loss = 0.122050, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5463253259658813, Accuracy = 0.8834007978439331
    Iter #2024960:  Learning rate = 0.002210:   Batch Loss = 0.126729, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5362672209739685, Accuracy = 0.8886639475822449
    Iter #2025472:  Learning rate = 0.002210:   Batch Loss = 0.122120, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5314846038818359, Accuracy = 0.887854278087616
    Iter #2025984:  Learning rate = 0.002210:   Batch Loss = 0.122571, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5429337620735168, Accuracy = 0.8882591128349304
    Iter #2026496:  Learning rate = 0.002210:   Batch Loss = 0.124985, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5319951772689819, Accuracy = 0.8890688419342041
    Iter #2027008:  Learning rate = 0.002210:   Batch Loss = 0.122090, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5411766767501831, Accuracy = 0.8890688419342041
    Iter #2027520:  Learning rate = 0.002210:   Batch Loss = 0.123426, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5339256525039673, Accuracy = 0.8906882405281067
    Iter #2028032:  Learning rate = 0.002210:   Batch Loss = 0.123180, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5262296199798584, Accuracy = 0.8927125334739685
    Iter #2028544:  Learning rate = 0.002210:   Batch Loss = 0.122588, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5290833711624146, Accuracy = 0.8939270973205566
    Iter #2029056:  Learning rate = 0.002210:   Batch Loss = 0.122564, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5325414538383484, Accuracy = 0.8898785710334778
    Iter #2029568:  Learning rate = 0.002210:   Batch Loss = 0.121936, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5293835997581482, Accuracy = 0.892307698726654
    Iter #2030080:  Learning rate = 0.002210:   Batch Loss = 0.123620, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5274028182029724, Accuracy = 0.8910931348800659
    Iter #2030592:  Learning rate = 0.002210:   Batch Loss = 0.121632, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5357805490493774, Accuracy = 0.8910931348800659
    Iter #2031104:  Learning rate = 0.002210:   Batch Loss = 0.122603, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5384433269500732, Accuracy = 0.8894736766815186
    Iter #2031616:  Learning rate = 0.002210:   Batch Loss = 0.122181, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.540992021560669, Accuracy = 0.8890688419342041
    Iter #2032128:  Learning rate = 0.002210:   Batch Loss = 0.125462, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5287134647369385, Accuracy = 0.892307698726654
    Iter #2032640:  Learning rate = 0.002210:   Batch Loss = 0.121756, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5619595050811768, Accuracy = 0.8829959630966187
    Iter #2033152:  Learning rate = 0.002210:   Batch Loss = 0.195169, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6652401685714722, Accuracy = 0.8493927121162415
    Iter #2033664:  Learning rate = 0.002210:   Batch Loss = 0.310396, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6721123456954956, Accuracy = 0.8461538553237915
    Iter #2034176:  Learning rate = 0.002210:   Batch Loss = 0.686631, Accuracy = 0.8125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.908504068851471, Accuracy = 0.7801619172096252
    Iter #2034688:  Learning rate = 0.002210:   Batch Loss = 0.461491, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8303565979003906, Accuracy = 0.7846153974533081
    Iter #2035200:  Learning rate = 0.002210:   Batch Loss = 0.458672, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7918519973754883, Accuracy = 0.804453432559967
    Iter #2035712:  Learning rate = 0.002210:   Batch Loss = 0.297110, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.793851912021637, Accuracy = 0.810121476650238
    Iter #2036224:  Learning rate = 0.002210:   Batch Loss = 0.312645, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6979673504829407, Accuracy = 0.8506072759628296
    Iter #2036736:  Learning rate = 0.002210:   Batch Loss = 0.409255, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.704145073890686, Accuracy = 0.8477732539176941
    Iter #2037248:  Learning rate = 0.002210:   Batch Loss = 0.445432, Accuracy = 0.875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7432342767715454, Accuracy = 0.8275303840637207
    Iter #2037760:  Learning rate = 0.002210:   Batch Loss = 0.273307, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.739206075668335, Accuracy = 0.843319833278656
    Iter #2038272:  Learning rate = 0.002210:   Batch Loss = 0.426022, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7235963344573975, Accuracy = 0.8388664126396179
    Iter #2038784:  Learning rate = 0.002210:   Batch Loss = 0.313060, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.710071861743927, Accuracy = 0.8506072759628296
    Iter #2039296:  Learning rate = 0.002210:   Batch Loss = 0.334023, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6798397898674011, Accuracy = 0.8607287406921387
    Iter #2039808:  Learning rate = 0.002210:   Batch Loss = 0.231864, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6407402753829956, Accuracy = 0.8631578683853149
    Iter #2040320:  Learning rate = 0.002210:   Batch Loss = 0.317109, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6167021989822388, Accuracy = 0.8757085204124451
    Iter #2040832:  Learning rate = 0.002210:   Batch Loss = 0.268286, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6202887892723083, Accuracy = 0.873279333114624
    Iter #2041344:  Learning rate = 0.002210:   Batch Loss = 0.305575, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6402081251144409, Accuracy = 0.8684210777282715
    Iter #2041856:  Learning rate = 0.002210:   Batch Loss = 0.208585, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6721091270446777, Accuracy = 0.8558704257011414
    Iter #2042368:  Learning rate = 0.002210:   Batch Loss = 0.181398, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5782549381256104, Accuracy = 0.8858299851417542
    Iter #2042880:  Learning rate = 0.002210:   Batch Loss = 0.208632, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5876966714859009, Accuracy = 0.8898785710334778
    Iter #2043392:  Learning rate = 0.002210:   Batch Loss = 0.198730, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6250443458557129, Accuracy = 0.8724696636199951
    Iter #2043904:  Learning rate = 0.002210:   Batch Loss = 0.236027, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6343985199928284, Accuracy = 0.878947377204895
    Iter #2044416:  Learning rate = 0.002210:   Batch Loss = 0.221204, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5848528146743774, Accuracy = 0.8809716701507568
    Iter #2044928:  Learning rate = 0.002210:   Batch Loss = 0.247886, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6122000813484192, Accuracy = 0.8829959630966187
    Iter #2045440:  Learning rate = 0.002210:   Batch Loss = 0.207283, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5737805366516113, Accuracy = 0.8854250907897949
    Iter #2045952:  Learning rate = 0.002210:   Batch Loss = 0.274814, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.589339554309845, Accuracy = 0.8850202560424805
    Iter #2046464:  Learning rate = 0.002210:   Batch Loss = 0.191754, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5538119077682495, Accuracy = 0.8910931348800659
    Iter #2046976:  Learning rate = 0.002210:   Batch Loss = 0.173120, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5696563124656677, Accuracy = 0.8898785710334778
    Iter #2047488:  Learning rate = 0.002210:   Batch Loss = 0.168217, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5840139985084534, Accuracy = 0.8910931348800659
    Iter #2048000:  Learning rate = 0.002210:   Batch Loss = 0.170952, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5586915016174316, Accuracy = 0.8906882405281067
    Iter #2048512:  Learning rate = 0.002210:   Batch Loss = 0.182411, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.55205899477005, Accuracy = 0.8935222625732422
    Iter #2049024:  Learning rate = 0.002210:   Batch Loss = 0.172981, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5733091831207275, Accuracy = 0.8882591128349304
    Iter #2049536:  Learning rate = 0.002210:   Batch Loss = 0.166072, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5665885806083679, Accuracy = 0.8927125334739685
    Iter #2050048:  Learning rate = 0.002210:   Batch Loss = 0.171645, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5502475500106812, Accuracy = 0.8890688419342041
    Iter #2050560:  Learning rate = 0.002210:   Batch Loss = 0.171725, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5413841009140015, Accuracy = 0.898785412311554
    Iter #2051072:  Learning rate = 0.002210:   Batch Loss = 0.172556, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5433864593505859, Accuracy = 0.8967611193656921
    Iter #2051584:  Learning rate = 0.002210:   Batch Loss = 0.208501, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5341551899909973, Accuracy = 0.895546555519104
    Iter #2052096:  Learning rate = 0.002210:   Batch Loss = 0.162995, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5238342881202698, Accuracy = 0.9032388925552368
    Iter #2052608:  Learning rate = 0.002210:   Batch Loss = 0.217316, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5423445701599121, Accuracy = 0.8943319916725159
    Iter #2053120:  Learning rate = 0.002210:   Batch Loss = 0.165190, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5364023447036743, Accuracy = 0.8947368264198303
    Iter #2053632:  Learning rate = 0.002210:   Batch Loss = 0.163706, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5576764345169067, Accuracy = 0.8890688419342041
    Iter #2054144:  Learning rate = 0.002210:   Batch Loss = 0.161050, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5316537618637085, Accuracy = 0.898785412311554
    Iter #2054656:  Learning rate = 0.002210:   Batch Loss = 0.164703, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5603882670402527, Accuracy = 0.8927125334739685
    Iter #2055168:  Learning rate = 0.002210:   Batch Loss = 0.165815, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5368693470954895, Accuracy = 0.8971660137176514
    Iter #2055680:  Learning rate = 0.002210:   Batch Loss = 0.165134, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5059638023376465, Accuracy = 0.9117408990859985
    Iter #2056192:  Learning rate = 0.002210:   Batch Loss = 0.158191, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5166178941726685, Accuracy = 0.9097166061401367
    Iter #2056704:  Learning rate = 0.002210:   Batch Loss = 0.165058, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5281468629837036, Accuracy = 0.9004048705101013
    Iter #2057216:  Learning rate = 0.002210:   Batch Loss = 0.163670, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5344027280807495, Accuracy = 0.9004048705101013
    Iter #2057728:  Learning rate = 0.002210:   Batch Loss = 0.160719, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.506565511226654, Accuracy = 0.9048582911491394
    Iter #2058240:  Learning rate = 0.002210:   Batch Loss = 0.156624, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5003727078437805, Accuracy = 0.9085020422935486
    Iter #2058752:  Learning rate = 0.002210:   Batch Loss = 0.154994, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4939762353897095, Accuracy = 0.9109311699867249
    Iter #2059264:  Learning rate = 0.002210:   Batch Loss = 0.153608, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48917800188064575, Accuracy = 0.9105263352394104
    Iter #2059776:  Learning rate = 0.002210:   Batch Loss = 0.156714, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48987215757369995, Accuracy = 0.912145733833313
    Iter #2060288:  Learning rate = 0.002210:   Batch Loss = 0.151923, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48591530323028564, Accuracy = 0.9101214408874512
    Iter #2060800:  Learning rate = 0.002210:   Batch Loss = 0.149370, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48491787910461426, Accuracy = 0.9097166061401367
    Iter #2061312:  Learning rate = 0.002210:   Batch Loss = 0.150604, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48527994751930237, Accuracy = 0.9117408990859985
    Iter #2061824:  Learning rate = 0.002210:   Batch Loss = 0.147843, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4850093722343445, Accuracy = 0.9113360047340393
    Iter #2062336:  Learning rate = 0.002210:   Batch Loss = 0.151130, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48517972230911255, Accuracy = 0.9093117117881775
    Iter #2062848:  Learning rate = 0.002210:   Batch Loss = 0.147847, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4846488833427429, Accuracy = 0.9105263352394104
    Iter #2063360:  Learning rate = 0.002210:   Batch Loss = 0.146082, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4829298257827759, Accuracy = 0.9101214408874512
    Iter #2063872:  Learning rate = 0.002210:   Batch Loss = 0.147558, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48320817947387695, Accuracy = 0.9093117117881775
    Iter #2064384:  Learning rate = 0.002210:   Batch Loss = 0.147595, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4840279221534729, Accuracy = 0.9105263352394104
    Iter #2064896:  Learning rate = 0.002210:   Batch Loss = 0.147709, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4821411967277527, Accuracy = 0.9093117117881775
    Iter #2065408:  Learning rate = 0.002210:   Batch Loss = 0.146446, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48165762424468994, Accuracy = 0.9097166061401367
    Iter #2065920:  Learning rate = 0.002210:   Batch Loss = 0.142945, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48501166701316833, Accuracy = 0.908906877040863
    Iter #2066432:  Learning rate = 0.002210:   Batch Loss = 0.147017, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4849697947502136, Accuracy = 0.9068825840950012
    Iter #2066944:  Learning rate = 0.002210:   Batch Loss = 0.142006, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48495060205459595, Accuracy = 0.9072874784469604
    Iter #2067456:  Learning rate = 0.002210:   Batch Loss = 0.143081, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48425450921058655, Accuracy = 0.9080971479415894
    Iter #2067968:  Learning rate = 0.002210:   Batch Loss = 0.143118, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4852110743522644, Accuracy = 0.9072874784469604
    Iter #2068480:  Learning rate = 0.002210:   Batch Loss = 0.141083, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48959454894065857, Accuracy = 0.9076923131942749
    Iter #2068992:  Learning rate = 0.002210:   Batch Loss = 0.141397, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48822999000549316, Accuracy = 0.9080971479415894
    Iter #2069504:  Learning rate = 0.002210:   Batch Loss = 0.140411, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48544231057167053, Accuracy = 0.9080971479415894
    Iter #2070016:  Learning rate = 0.002210:   Batch Loss = 0.139935, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4868776798248291, Accuracy = 0.908906877040863
    Iter #2070528:  Learning rate = 0.002210:   Batch Loss = 0.140651, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4858235716819763, Accuracy = 0.9076923131942749
    Iter #2071040:  Learning rate = 0.002210:   Batch Loss = 0.140523, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4867434501647949, Accuracy = 0.9072874784469604
    Iter #2071552:  Learning rate = 0.002210:   Batch Loss = 0.141824, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.494622141122818, Accuracy = 0.9048582911491394
    Iter #2072064:  Learning rate = 0.002210:   Batch Loss = 0.141222, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4968808889389038, Accuracy = 0.904453456401825
    Iter #2072576:  Learning rate = 0.002210:   Batch Loss = 0.138440, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48536986112594604, Accuracy = 0.9093117117881775
    Iter #2073088:  Learning rate = 0.002210:   Batch Loss = 0.138917, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48529815673828125, Accuracy = 0.9072874784469604
    Iter #2073600:  Learning rate = 0.002210:   Batch Loss = 0.136921, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4887591004371643, Accuracy = 0.9048582911491394
    Iter #2074112:  Learning rate = 0.002210:   Batch Loss = 0.138251, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4884239137172699, Accuracy = 0.9056680202484131
    Iter #2074624:  Learning rate = 0.002210:   Batch Loss = 0.138377, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48974353075027466, Accuracy = 0.9060728549957275
    Iter #2075136:  Learning rate = 0.002210:   Batch Loss = 0.137163, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4910573363304138, Accuracy = 0.9036437273025513
    Iter #2075648:  Learning rate = 0.002210:   Batch Loss = 0.138207, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48958656191825867, Accuracy = 0.9060728549957275
    Iter #2076160:  Learning rate = 0.002210:   Batch Loss = 0.137888, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48902183771133423, Accuracy = 0.9068825840950012
    Iter #2076672:  Learning rate = 0.002210:   Batch Loss = 0.133839, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4895566999912262, Accuracy = 0.9060728549957275
    Iter #2077184:  Learning rate = 0.002210:   Batch Loss = 0.138329, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4902000427246094, Accuracy = 0.9060728549957275
    Iter #2077696:  Learning rate = 0.002210:   Batch Loss = 0.133989, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49001580476760864, Accuracy = 0.9068825840950012
    Iter #2078208:  Learning rate = 0.002210:   Batch Loss = 0.136844, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4884950816631317, Accuracy = 0.9048582911491394
    Iter #2078720:  Learning rate = 0.002210:   Batch Loss = 0.136107, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49063917994499207, Accuracy = 0.9052631855010986
    Iter #2079232:  Learning rate = 0.002210:   Batch Loss = 0.137052, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4922969937324524, Accuracy = 0.9060728549957275
    Iter #2079744:  Learning rate = 0.002210:   Batch Loss = 0.133575, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49699926376342773, Accuracy = 0.9036437273025513
    Iter #2080256:  Learning rate = 0.002210:   Batch Loss = 0.132011, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.500281035900116, Accuracy = 0.8999999761581421
    Iter #2080768:  Learning rate = 0.002210:   Batch Loss = 0.133428, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49169740080833435, Accuracy = 0.9036437273025513
    Iter #2081280:  Learning rate = 0.002210:   Batch Loss = 0.134928, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49199873208999634, Accuracy = 0.9064777493476868
    Iter #2081792:  Learning rate = 0.002210:   Batch Loss = 0.133725, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4953036904335022, Accuracy = 0.9028339982032776
    Iter #2082304:  Learning rate = 0.002210:   Batch Loss = 0.130686, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49677497148513794, Accuracy = 0.9024291634559631
    Iter #2082816:  Learning rate = 0.002210:   Batch Loss = 0.134186, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49819356203079224, Accuracy = 0.9016194343566895
    Iter #2083328:  Learning rate = 0.002210:   Batch Loss = 0.131520, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5011191964149475, Accuracy = 0.9004048705101013
    Iter #2083840:  Learning rate = 0.002210:   Batch Loss = 0.132162, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5004788041114807, Accuracy = 0.9016194343566895
    Iter #2084352:  Learning rate = 0.002210:   Batch Loss = 0.132732, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4996246099472046, Accuracy = 0.9016194343566895
    Iter #2084864:  Learning rate = 0.002210:   Batch Loss = 0.132800, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.498114675283432, Accuracy = 0.904453456401825
    Iter #2085376:  Learning rate = 0.002210:   Batch Loss = 0.133153, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4983815550804138, Accuracy = 0.9016194343566895
    Iter #2085888:  Learning rate = 0.002210:   Batch Loss = 0.130855, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4965217709541321, Accuracy = 0.904453456401825
    Iter #2086400:  Learning rate = 0.002210:   Batch Loss = 0.132821, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4942406117916107, Accuracy = 0.9068825840950012
    Iter #2086912:  Learning rate = 0.002210:   Batch Loss = 0.133708, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5014150142669678, Accuracy = 0.8999999761581421
    Iter #2087424:  Learning rate = 0.002210:   Batch Loss = 0.131072, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5018562078475952, Accuracy = 0.9032388925552368
    Iter #2087936:  Learning rate = 0.002210:   Batch Loss = 0.130340, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49954381585121155, Accuracy = 0.9028339982032776
    Iter #2088448:  Learning rate = 0.002210:   Batch Loss = 0.129760, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5004859566688538, Accuracy = 0.9024291634559631
    Iter #2088960:  Learning rate = 0.002210:   Batch Loss = 0.129763, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5025901794433594, Accuracy = 0.901214599609375
    Iter #2089472:  Learning rate = 0.002210:   Batch Loss = 0.129730, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5011769533157349, Accuracy = 0.8995951414108276
    Iter #2089984:  Learning rate = 0.002210:   Batch Loss = 0.129786, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5047455430030823, Accuracy = 0.9020242691040039
    Iter #2090496:  Learning rate = 0.002210:   Batch Loss = 0.129736, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5023685097694397, Accuracy = 0.9020242691040039
    Iter #2091008:  Learning rate = 0.002210:   Batch Loss = 0.128818, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5001050233840942, Accuracy = 0.8999999761581421
    Iter #2091520:  Learning rate = 0.002210:   Batch Loss = 0.128847, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49818283319473267, Accuracy = 0.9008097052574158
    Iter #2092032:  Learning rate = 0.002210:   Batch Loss = 0.129584, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49721163511276245, Accuracy = 0.9020242691040039
    Iter #2092544:  Learning rate = 0.002210:   Batch Loss = 0.126226, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49491995573043823, Accuracy = 0.904453456401825
    Iter #2093056:  Learning rate = 0.002210:   Batch Loss = 0.127351, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.501925528049469, Accuracy = 0.8999999761581421
    Iter #2093568:  Learning rate = 0.002210:   Batch Loss = 0.128703, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5010987520217896, Accuracy = 0.9020242691040039
    Iter #2094080:  Learning rate = 0.002210:   Batch Loss = 0.127536, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49802929162979126, Accuracy = 0.9032388925552368
    Iter #2094592:  Learning rate = 0.002210:   Batch Loss = 0.128675, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.501426100730896, Accuracy = 0.9016194343566895
    Iter #2095104:  Learning rate = 0.002210:   Batch Loss = 0.129321, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5049121379852295, Accuracy = 0.9016194343566895
    Iter #2095616:  Learning rate = 0.002210:   Batch Loss = 0.127028, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5020443797111511, Accuracy = 0.9036437273025513
    Iter #2096128:  Learning rate = 0.002210:   Batch Loss = 0.128090, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5089797973632812, Accuracy = 0.8979756832122803
    Iter #2096640:  Learning rate = 0.002210:   Batch Loss = 0.126828, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5071801543235779, Accuracy = 0.8999999761581421
    Iter #2097152:  Learning rate = 0.002210:   Batch Loss = 0.126240, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5027987360954285, Accuracy = 0.8979756832122803
    Iter #2097664:  Learning rate = 0.002210:   Batch Loss = 0.124009, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.502935528755188, Accuracy = 0.8995951414108276
    Iter #2098176:  Learning rate = 0.002210:   Batch Loss = 0.127245, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5040687322616577, Accuracy = 0.9020242691040039
    Iter #2098688:  Learning rate = 0.002210:   Batch Loss = 0.126338, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5083797574043274, Accuracy = 0.8983805775642395
    Iter #2099200:  Learning rate = 0.002210:   Batch Loss = 0.125110, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5075826048851013, Accuracy = 0.8975708484649658
    Iter #2099712:  Learning rate = 0.002210:   Batch Loss = 0.123724, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5040667057037354, Accuracy = 0.8983805775642395
    Iter #2100224:  Learning rate = 0.002122:   Batch Loss = 0.125794, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5080724954605103, Accuracy = 0.8979756832122803
    Iter #2100736:  Learning rate = 0.002122:   Batch Loss = 0.123627, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5163825154304504, Accuracy = 0.8959513902664185
    Iter #2101248:  Learning rate = 0.002122:   Batch Loss = 0.124411, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5121781229972839, Accuracy = 0.8943319916725159
    Iter #2101760:  Learning rate = 0.002122:   Batch Loss = 0.123825, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5033999681472778, Accuracy = 0.8979756832122803
    Iter #2102272:  Learning rate = 0.002122:   Batch Loss = 0.126569, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5013173818588257, Accuracy = 0.901214599609375
    Iter #2102784:  Learning rate = 0.002122:   Batch Loss = 0.123903, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5068053007125854, Accuracy = 0.898785412311554
    Iter #2103296:  Learning rate = 0.002122:   Batch Loss = 0.123377, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5079032182693481, Accuracy = 0.8991903066635132
    Iter #2103808:  Learning rate = 0.002122:   Batch Loss = 0.125325, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5053771734237671, Accuracy = 0.8995951414108276
    Iter #2104320:  Learning rate = 0.002122:   Batch Loss = 0.125304, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5088390111923218, Accuracy = 0.8991903066635132
    Iter #2104832:  Learning rate = 0.002122:   Batch Loss = 0.124144, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5099422335624695, Accuracy = 0.8991903066635132
    Iter #2105344:  Learning rate = 0.002122:   Batch Loss = 0.121543, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5077193975448608, Accuracy = 0.8983805775642395
    Iter #2105856:  Learning rate = 0.002122:   Batch Loss = 0.122243, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5098037123680115, Accuracy = 0.8991903066635132
    Iter #2106368:  Learning rate = 0.002122:   Batch Loss = 0.122171, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5097461342811584, Accuracy = 0.8967611193656921
    Iter #2106880:  Learning rate = 0.002122:   Batch Loss = 0.122500, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.508402943611145, Accuracy = 0.8975708484649658
    Iter #2107392:  Learning rate = 0.002122:   Batch Loss = 0.122497, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.513984203338623, Accuracy = 0.8971660137176514
    Iter #2107904:  Learning rate = 0.002122:   Batch Loss = 0.122610, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5173102021217346, Accuracy = 0.8975708484649658
    Iter #2108416:  Learning rate = 0.002122:   Batch Loss = 0.121030, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5125361680984497, Accuracy = 0.8983805775642395
    Iter #2108928:  Learning rate = 0.002122:   Batch Loss = 0.120395, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5099008083343506, Accuracy = 0.898785412311554
    Iter #2109440:  Learning rate = 0.002122:   Batch Loss = 0.122542, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5124806761741638, Accuracy = 0.8967611193656921
    Iter #2109952:  Learning rate = 0.002122:   Batch Loss = 0.122184, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5141075849533081, Accuracy = 0.8939270973205566
    Iter #2110464:  Learning rate = 0.002122:   Batch Loss = 0.156685, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6295180320739746, Accuracy = 0.8684210777282715
    Iter #2110976:  Learning rate = 0.002122:   Batch Loss = 0.314233, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6207364797592163, Accuracy = 0.8570850491523743
    Iter #2111488:  Learning rate = 0.002122:   Batch Loss = 0.350557, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6321984529495239, Accuracy = 0.8510121703147888
    Iter #2112000:  Learning rate = 0.002122:   Batch Loss = 0.413565, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.730070173740387, Accuracy = 0.8319838047027588
    Iter #2112512:  Learning rate = 0.002122:   Batch Loss = 0.311428, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.726874053478241, Accuracy = 0.8255060911178589
    Iter #2113024:  Learning rate = 0.002122:   Batch Loss = 0.249585, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6522986888885498, Accuracy = 0.8582996129989624
    Iter #2113536:  Learning rate = 0.002122:   Batch Loss = 0.415015, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6944680213928223, Accuracy = 0.8477732539176941
    Iter #2114048:  Learning rate = 0.002122:   Batch Loss = 0.325049, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6851069927215576, Accuracy = 0.8497975468635559
    Iter #2114560:  Learning rate = 0.002122:   Batch Loss = 0.236500, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6556110978126526, Accuracy = 0.8554655909538269
    Iter #2115072:  Learning rate = 0.002122:   Batch Loss = 0.446278, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7011052370071411, Accuracy = 0.8502024412155151
    Iter #2115584:  Learning rate = 0.002122:   Batch Loss = 0.385759, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7267366647720337, Accuracy = 0.8384615182876587
    Iter #2116096:  Learning rate = 0.002122:   Batch Loss = 0.388402, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7310572266578674, Accuracy = 0.8417003750801086
    Iter #2116608:  Learning rate = 0.002122:   Batch Loss = 0.456102, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.694405198097229, Accuracy = 0.852226734161377
    Iter #2117120:  Learning rate = 0.002122:   Batch Loss = 0.368568, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6334556937217712, Accuracy = 0.8672064542770386
    Iter #2117632:  Learning rate = 0.002122:   Batch Loss = 0.255573, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6705390810966492, Accuracy = 0.8607287406921387
    Iter #2118144:  Learning rate = 0.002122:   Batch Loss = 0.277469, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6435422897338867, Accuracy = 0.8704453706741333
    Iter #2118656:  Learning rate = 0.002122:   Batch Loss = 0.210441, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6463321447372437, Accuracy = 0.8651821613311768
    Iter #2119168:  Learning rate = 0.002122:   Batch Loss = 0.218323, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6353065371513367, Accuracy = 0.878947377204895
    Iter #2119680:  Learning rate = 0.002122:   Batch Loss = 0.245330, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6628909111022949, Accuracy = 0.8627530336380005
    Iter #2120192:  Learning rate = 0.002122:   Batch Loss = 0.236247, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7182923555374146, Accuracy = 0.8453441262245178
    Iter #2120704:  Learning rate = 0.002122:   Batch Loss = 0.376248, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6707682609558105, Accuracy = 0.8550607562065125
    Iter #2121216:  Learning rate = 0.002122:   Batch Loss = 0.236745, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.691622793674469, Accuracy = 0.8506072759628296
    Iter #2121728:  Learning rate = 0.002122:   Batch Loss = 0.250268, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6743748188018799, Accuracy = 0.8514170050621033
    Iter #2122240:  Learning rate = 0.002122:   Batch Loss = 0.305855, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6503155827522278, Accuracy = 0.8619433045387268
    Iter #2122752:  Learning rate = 0.002122:   Batch Loss = 0.233530, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5727322697639465, Accuracy = 0.8829959630966187
    Iter #2123264:  Learning rate = 0.002122:   Batch Loss = 0.247231, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5752637982368469, Accuracy = 0.8862348198890686
    Iter #2123776:  Learning rate = 0.002122:   Batch Loss = 0.242609, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6173375248908997, Accuracy = 0.8785424828529358
    Iter #2124288:  Learning rate = 0.002122:   Batch Loss = 0.206216, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6404485106468201, Accuracy = 0.8672064542770386
    Iter #2124800:  Learning rate = 0.002122:   Batch Loss = 0.227665, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6482582092285156, Accuracy = 0.8659918904304504
    Iter #2125312:  Learning rate = 0.002122:   Batch Loss = 0.239767, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6144604682922363, Accuracy = 0.8736842274665833
    Iter #2125824:  Learning rate = 0.002122:   Batch Loss = 0.215733, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5976527333259583, Accuracy = 0.8829959630966187
    Iter #2126336:  Learning rate = 0.002122:   Batch Loss = 0.199273, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6604751944541931, Accuracy = 0.8647773265838623
    Iter #2126848:  Learning rate = 0.002122:   Batch Loss = 0.246286, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6415600776672363, Accuracy = 0.8736842274665833
    Iter #2127360:  Learning rate = 0.002122:   Batch Loss = 0.170672, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.616438090801239, Accuracy = 0.8809716701507568
    Iter #2127872:  Learning rate = 0.002122:   Batch Loss = 0.230775, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5941258072853088, Accuracy = 0.8842105269432068
    Iter #2128384:  Learning rate = 0.002122:   Batch Loss = 0.173314, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.568198025226593, Accuracy = 0.8842105269432068
    Iter #2128896:  Learning rate = 0.002122:   Batch Loss = 0.168612, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5718957185745239, Accuracy = 0.8914979696273804
    Iter #2129408:  Learning rate = 0.002122:   Batch Loss = 0.185351, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5669501423835754, Accuracy = 0.8882591128349304
    Iter #2129920:  Learning rate = 0.002122:   Batch Loss = 0.206853, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5725179314613342, Accuracy = 0.8947368264198303
    Iter #2130432:  Learning rate = 0.002122:   Batch Loss = 0.188724, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5538726449012756, Accuracy = 0.8979756832122803
    Iter #2130944:  Learning rate = 0.002122:   Batch Loss = 0.174123, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.575154185295105, Accuracy = 0.8906882405281067
    Iter #2131456:  Learning rate = 0.002122:   Batch Loss = 0.176846, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5623007416725159, Accuracy = 0.8971660137176514
    Iter #2131968:  Learning rate = 0.002122:   Batch Loss = 0.172936, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5348967909812927, Accuracy = 0.8927125334739685
    Iter #2132480:  Learning rate = 0.002122:   Batch Loss = 0.177400, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.515596330165863, Accuracy = 0.9093117117881775
    Iter #2132992:  Learning rate = 0.002122:   Batch Loss = 0.168895, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.533274233341217, Accuracy = 0.9024291634559631
    Iter #2133504:  Learning rate = 0.002122:   Batch Loss = 0.167235, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5316987633705139, Accuracy = 0.9056680202484131
    Iter #2134016:  Learning rate = 0.002122:   Batch Loss = 0.162863, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5255473852157593, Accuracy = 0.9036437273025513
    Iter #2134528:  Learning rate = 0.002122:   Batch Loss = 0.156690, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5120027661323547, Accuracy = 0.9072874784469604
    Iter #2135040:  Learning rate = 0.002122:   Batch Loss = 0.161885, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5205246210098267, Accuracy = 0.9040485620498657
    Iter #2135552:  Learning rate = 0.002122:   Batch Loss = 0.156038, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5224243402481079, Accuracy = 0.9020242691040039
    Iter #2136064:  Learning rate = 0.002122:   Batch Loss = 0.153782, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5041718482971191, Accuracy = 0.9072874784469604
    Iter #2136576:  Learning rate = 0.002122:   Batch Loss = 0.154207, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49667447805404663, Accuracy = 0.9068825840950012
    Iter #2137088:  Learning rate = 0.002122:   Batch Loss = 0.151819, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4925093650817871, Accuracy = 0.9080971479415894
    Iter #2137600:  Learning rate = 0.002122:   Batch Loss = 0.150946, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4938409924507141, Accuracy = 0.9109311699867249
    Iter #2138112:  Learning rate = 0.002122:   Batch Loss = 0.153862, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49529507756233215, Accuracy = 0.9072874784469604
    Iter #2138624:  Learning rate = 0.002122:   Batch Loss = 0.150442, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4954720735549927, Accuracy = 0.9056680202484131
    Iter #2139136:  Learning rate = 0.002122:   Batch Loss = 0.150407, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4907652735710144, Accuracy = 0.9064777493476868
    Iter #2139648:  Learning rate = 0.002122:   Batch Loss = 0.149829, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4881271719932556, Accuracy = 0.9101214408874512
    Iter #2140160:  Learning rate = 0.002122:   Batch Loss = 0.149084, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4901033043861389, Accuracy = 0.9093117117881775
    Iter #2140672:  Learning rate = 0.002122:   Batch Loss = 0.148527, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49100950360298157, Accuracy = 0.9064777493476868
    Iter #2141184:  Learning rate = 0.002122:   Batch Loss = 0.148327, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4923892021179199, Accuracy = 0.9060728549957275
    Iter #2141696:  Learning rate = 0.002122:   Batch Loss = 0.147035, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49237892031669617, Accuracy = 0.9052631855010986
    Iter #2142208:  Learning rate = 0.002122:   Batch Loss = 0.149536, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49041396379470825, Accuracy = 0.9060728549957275
    Iter #2142720:  Learning rate = 0.002122:   Batch Loss = 0.147290, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4882071018218994, Accuracy = 0.9085020422935486
    Iter #2143232:  Learning rate = 0.002122:   Batch Loss = 0.144521, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4870855212211609, Accuracy = 0.9085020422935486
    Iter #2143744:  Learning rate = 0.002122:   Batch Loss = 0.144807, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4900984764099121, Accuracy = 0.9060728549957275
    Iter #2144256:  Learning rate = 0.002122:   Batch Loss = 0.142751, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49095433950424194, Accuracy = 0.9036437273025513
    Iter #2144768:  Learning rate = 0.002122:   Batch Loss = 0.146356, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.487570196390152, Accuracy = 0.9052631855010986
    Iter #2145280:  Learning rate = 0.002122:   Batch Loss = 0.142664, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48542892932891846, Accuracy = 0.908906877040863
    Iter #2145792:  Learning rate = 0.002122:   Batch Loss = 0.145913, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4854050874710083, Accuracy = 0.9085020422935486
    Iter #2146304:  Learning rate = 0.002122:   Batch Loss = 0.142998, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48614320158958435, Accuracy = 0.9068825840950012
    Iter #2146816:  Learning rate = 0.002122:   Batch Loss = 0.145655, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4884534478187561, Accuracy = 0.9036437273025513
    Iter #2147328:  Learning rate = 0.002122:   Batch Loss = 0.143448, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4884750247001648, Accuracy = 0.9048582911491394
    Iter #2147840:  Learning rate = 0.002122:   Batch Loss = 0.142957, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48789069056510925, Accuracy = 0.9072874784469604
    Iter #2148352:  Learning rate = 0.002122:   Batch Loss = 0.142075, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48697757720947266, Accuracy = 0.9064777493476868
    Iter #2148864:  Learning rate = 0.002122:   Batch Loss = 0.141683, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49129289388656616, Accuracy = 0.904453456401825
    Iter #2149376:  Learning rate = 0.002122:   Batch Loss = 0.141304, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4913090765476227, Accuracy = 0.9048582911491394
    Iter #2149888:  Learning rate = 0.002122:   Batch Loss = 0.140509, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49101150035858154, Accuracy = 0.9032388925552368
    Iter #2150400:  Learning rate = 0.002122:   Batch Loss = 0.140715, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4893845319747925, Accuracy = 0.9052631855010986
    Iter #2150912:  Learning rate = 0.002122:   Batch Loss = 0.139468, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.493222713470459, Accuracy = 0.9048582911491394
    Iter #2151424:  Learning rate = 0.002122:   Batch Loss = 0.140690, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4943111538887024, Accuracy = 0.9032388925552368
    Iter #2151936:  Learning rate = 0.002122:   Batch Loss = 0.139930, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4921952784061432, Accuracy = 0.9040485620498657
    Iter #2152448:  Learning rate = 0.002122:   Batch Loss = 0.137880, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4916633367538452, Accuracy = 0.9028339982032776
    Iter #2152960:  Learning rate = 0.002122:   Batch Loss = 0.136922, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4956614375114441, Accuracy = 0.9040485620498657
    Iter #2153472:  Learning rate = 0.002122:   Batch Loss = 0.138230, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49402350187301636, Accuracy = 0.9032388925552368
    Iter #2153984:  Learning rate = 0.002122:   Batch Loss = 0.136352, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4923863410949707, Accuracy = 0.9020242691040039
    Iter #2154496:  Learning rate = 0.002122:   Batch Loss = 0.137996, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4926244616508484, Accuracy = 0.9036437273025513
    Iter #2155008:  Learning rate = 0.002122:   Batch Loss = 0.137250, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4925016164779663, Accuracy = 0.9036437273025513
    Iter #2155520:  Learning rate = 0.002122:   Batch Loss = 0.137240, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4954618215560913, Accuracy = 0.901214599609375
    Iter #2156032:  Learning rate = 0.002122:   Batch Loss = 0.135867, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49429985880851746, Accuracy = 0.9032388925552368
    Iter #2156544:  Learning rate = 0.002122:   Batch Loss = 0.136021, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49235570430755615, Accuracy = 0.904453456401825
    Iter #2157056:  Learning rate = 0.002122:   Batch Loss = 0.134897, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49200597405433655, Accuracy = 0.9036437273025513
    Iter #2157568:  Learning rate = 0.002122:   Batch Loss = 0.134392, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49395936727523804, Accuracy = 0.904453456401825
    Iter #2158080:  Learning rate = 0.002122:   Batch Loss = 0.135258, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49616217613220215, Accuracy = 0.904453456401825
    Iter #2158592:  Learning rate = 0.002122:   Batch Loss = 0.135182, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4951125979423523, Accuracy = 0.9016194343566895
    Iter #2159104:  Learning rate = 0.002122:   Batch Loss = 0.133114, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4897061884403229, Accuracy = 0.9056680202484131
    Iter #2159616:  Learning rate = 0.002122:   Batch Loss = 0.133426, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4901728928089142, Accuracy = 0.9060728549957275
    Iter #2160128:  Learning rate = 0.002122:   Batch Loss = 0.133476, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5016283392906189, Accuracy = 0.901214599609375
    Iter #2160640:  Learning rate = 0.002122:   Batch Loss = 0.133503, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49920934438705444, Accuracy = 0.9036437273025513
    Iter #2161152:  Learning rate = 0.002122:   Batch Loss = 0.134073, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49059635400772095, Accuracy = 0.9040485620498657
    Iter #2161664:  Learning rate = 0.002122:   Batch Loss = 0.134181, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4889468550682068, Accuracy = 0.904453456401825
    Iter #2162176:  Learning rate = 0.002122:   Batch Loss = 0.132421, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4904566705226898, Accuracy = 0.9048582911491394
    Iter #2162688:  Learning rate = 0.002122:   Batch Loss = 0.132450, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49388933181762695, Accuracy = 0.9040485620498657
    Iter #2163200:  Learning rate = 0.002122:   Batch Loss = 0.132052, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49548906087875366, Accuracy = 0.9020242691040039
    Iter #2163712:  Learning rate = 0.002122:   Batch Loss = 0.135867, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4893532693386078, Accuracy = 0.9040485620498657
    Iter #2164224:  Learning rate = 0.002122:   Batch Loss = 0.132022, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4881407916545868, Accuracy = 0.9036437273025513
    Iter #2164736:  Learning rate = 0.002122:   Batch Loss = 0.130058, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48826730251312256, Accuracy = 0.904453456401825
    Iter #2165248:  Learning rate = 0.002122:   Batch Loss = 0.130864, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4914371967315674, Accuracy = 0.9036437273025513
    Iter #2165760:  Learning rate = 0.002122:   Batch Loss = 0.130234, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49598196148872375, Accuracy = 0.9016194343566895
    Iter #2166272:  Learning rate = 0.002122:   Batch Loss = 0.130893, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49625009298324585, Accuracy = 0.9028339982032776
    Iter #2166784:  Learning rate = 0.002122:   Batch Loss = 0.132131, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49473676085472107, Accuracy = 0.9032388925552368
    Iter #2167296:  Learning rate = 0.002122:   Batch Loss = 0.130552, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5014053583145142, Accuracy = 0.901214599609375
    Iter #2167808:  Learning rate = 0.002122:   Batch Loss = 0.130531, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4991369843482971, Accuracy = 0.9016194343566895
    Iter #2168320:  Learning rate = 0.002122:   Batch Loss = 0.131242, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4996759593486786, Accuracy = 0.9024291634559631
    Iter #2168832:  Learning rate = 0.002122:   Batch Loss = 0.127891, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.499765008687973, Accuracy = 0.8995951414108276
    Iter #2169344:  Learning rate = 0.002122:   Batch Loss = 0.130985, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5013364553451538, Accuracy = 0.9004048705101013
    Iter #2169856:  Learning rate = 0.002122:   Batch Loss = 0.129504, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5020483732223511, Accuracy = 0.9008097052574158
    Iter #2170368:  Learning rate = 0.002122:   Batch Loss = 0.126923, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.50384521484375, Accuracy = 0.8999999761581421
    Iter #2170880:  Learning rate = 0.002122:   Batch Loss = 0.131400, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4985421895980835, Accuracy = 0.9004048705101013
    Iter #2171392:  Learning rate = 0.002122:   Batch Loss = 0.129263, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4995369017124176, Accuracy = 0.9008097052574158
    Iter #2171904:  Learning rate = 0.002122:   Batch Loss = 0.130283, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49961888790130615, Accuracy = 0.8995951414108276
    Iter #2172416:  Learning rate = 0.002122:   Batch Loss = 0.128652, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4998006820678711, Accuracy = 0.901214599609375
    Iter #2172928:  Learning rate = 0.002122:   Batch Loss = 0.127370, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5042027831077576, Accuracy = 0.8975708484649658
    Iter #2173440:  Learning rate = 0.002122:   Batch Loss = 0.128109, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.500908374786377, Accuracy = 0.8995951414108276
    Iter #2173952:  Learning rate = 0.002122:   Batch Loss = 0.127666, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5021234154701233, Accuracy = 0.8967611193656921
    Iter #2174464:  Learning rate = 0.002122:   Batch Loss = 0.127333, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5034826993942261, Accuracy = 0.898785412311554
    Iter #2174976:  Learning rate = 0.002122:   Batch Loss = 0.126019, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5022886991500854, Accuracy = 0.8991903066635132
    Iter #2175488:  Learning rate = 0.002122:   Batch Loss = 0.126836, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5006122589111328, Accuracy = 0.8991903066635132
    Iter #2176000:  Learning rate = 0.002122:   Batch Loss = 0.128253, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5021060705184937, Accuracy = 0.8983805775642395
    Iter #2176512:  Learning rate = 0.002122:   Batch Loss = 0.125022, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5056982040405273, Accuracy = 0.8991903066635132
    Iter #2177024:  Learning rate = 0.002122:   Batch Loss = 0.126842, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.50096595287323, Accuracy = 0.9004048705101013
    Iter #2177536:  Learning rate = 0.002122:   Batch Loss = 0.127389, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5006584525108337, Accuracy = 0.898785412311554
    Iter #2178048:  Learning rate = 0.002122:   Batch Loss = 0.126806, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5021162033081055, Accuracy = 0.9008097052574158
    Iter #2178560:  Learning rate = 0.002122:   Batch Loss = 0.125713, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5035715103149414, Accuracy = 0.8983805775642395
    Iter #2179072:  Learning rate = 0.002122:   Batch Loss = 0.125069, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.513351321220398, Accuracy = 0.8963562846183777
    Iter #2179584:  Learning rate = 0.002122:   Batch Loss = 0.126995, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5016530156135559, Accuracy = 0.898785412311554
    Iter #2180096:  Learning rate = 0.002122:   Batch Loss = 0.126733, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5064301490783691, Accuracy = 0.8999999761581421
    Iter #2180608:  Learning rate = 0.002122:   Batch Loss = 0.128016, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5011981129646301, Accuracy = 0.8979756832122803
    Iter #2181120:  Learning rate = 0.002122:   Batch Loss = 0.124040, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4972221851348877, Accuracy = 0.9008097052574158
    Iter #2181632:  Learning rate = 0.002122:   Batch Loss = 0.125141, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5052497982978821, Accuracy = 0.898785412311554
    Iter #2182144:  Learning rate = 0.002122:   Batch Loss = 0.124705, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5026808977127075, Accuracy = 0.8975708484649658
    Iter #2182656:  Learning rate = 0.002122:   Batch Loss = 0.124681, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5054909586906433, Accuracy = 0.8967611193656921
    Iter #2183168:  Learning rate = 0.002122:   Batch Loss = 0.124383, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5012648701667786, Accuracy = 0.9004048705101013
    Iter #2183680:  Learning rate = 0.002122:   Batch Loss = 0.123129, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49934113025665283, Accuracy = 0.8975708484649658
    Iter #2184192:  Learning rate = 0.002122:   Batch Loss = 0.125668, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49771156907081604, Accuracy = 0.8999999761581421
    Iter #2184704:  Learning rate = 0.002122:   Batch Loss = 0.124566, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5008793473243713, Accuracy = 0.9004048705101013
    Iter #2185216:  Learning rate = 0.002122:   Batch Loss = 0.120985, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5080648064613342, Accuracy = 0.8979756832122803
    Iter #2185728:  Learning rate = 0.002122:   Batch Loss = 0.124390, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5041826963424683, Accuracy = 0.8983805775642395
    Iter #2186240:  Learning rate = 0.002122:   Batch Loss = 0.122491, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5064550638198853, Accuracy = 0.8939270973205566
    Iter #2186752:  Learning rate = 0.002122:   Batch Loss = 0.124716, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5059117078781128, Accuracy = 0.8975708484649658
    Iter #2187264:  Learning rate = 0.002122:   Batch Loss = 0.122875, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5086351633071899, Accuracy = 0.898785412311554
    Iter #2187776:  Learning rate = 0.002122:   Batch Loss = 0.125284, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5060325264930725, Accuracy = 0.8959513902664185
    Iter #2188288:  Learning rate = 0.002122:   Batch Loss = 0.124036, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5032320022583008, Accuracy = 0.8979756832122803
    Iter #2188800:  Learning rate = 0.002122:   Batch Loss = 0.122218, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5016176104545593, Accuracy = 0.8959513902664185
    Iter #2189312:  Learning rate = 0.002122:   Batch Loss = 0.123227, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5082196593284607, Accuracy = 0.8971660137176514
    Iter #2189824:  Learning rate = 0.002122:   Batch Loss = 0.122922, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5088865160942078, Accuracy = 0.8951417207717896
    Iter #2190336:  Learning rate = 0.002122:   Batch Loss = 0.122376, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5040581226348877, Accuracy = 0.8963562846183777
    Iter #2190848:  Learning rate = 0.002122:   Batch Loss = 0.121930, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5059868097305298, Accuracy = 0.8959513902664185
    Iter #2191360:  Learning rate = 0.002122:   Batch Loss = 0.120785, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5066850185394287, Accuracy = 0.8959513902664185
    Iter #2191872:  Learning rate = 0.002122:   Batch Loss = 0.122809, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5111561417579651, Accuracy = 0.8963562846183777
    Iter #2192384:  Learning rate = 0.002122:   Batch Loss = 0.121136, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5079651474952698, Accuracy = 0.8991903066635132
    Iter #2192896:  Learning rate = 0.002122:   Batch Loss = 0.122187, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5161089897155762, Accuracy = 0.8931174278259277
    Iter #2193408:  Learning rate = 0.002122:   Batch Loss = 0.123768, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5167164206504822, Accuracy = 0.892307698726654
    Iter #2193920:  Learning rate = 0.002122:   Batch Loss = 0.121316, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5208829045295715, Accuracy = 0.8898785710334778
    Iter #2194432:  Learning rate = 0.002122:   Batch Loss = 0.122298, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5225778818130493, Accuracy = 0.8902834057807922
    Iter #2194944:  Learning rate = 0.002122:   Batch Loss = 0.121801, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5139783024787903, Accuracy = 0.8947368264198303
    Iter #2195456:  Learning rate = 0.002122:   Batch Loss = 0.123124, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.514511227607727, Accuracy = 0.8919028043746948
    Iter #2195968:  Learning rate = 0.002122:   Batch Loss = 0.121488, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5134766697883606, Accuracy = 0.8910931348800659
    Iter #2196480:  Learning rate = 0.002122:   Batch Loss = 0.119582, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.514570415019989, Accuracy = 0.8927125334739685
    Iter #2196992:  Learning rate = 0.002122:   Batch Loss = 0.120230, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5174008011817932, Accuracy = 0.892307698726654
    Iter #2197504:  Learning rate = 0.002122:   Batch Loss = 0.118796, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5127533674240112, Accuracy = 0.895546555519104
    Iter #2198016:  Learning rate = 0.002122:   Batch Loss = 0.120257, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5099283456802368, Accuracy = 0.8943319916725159
    Iter #2198528:  Learning rate = 0.002122:   Batch Loss = 0.120316, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5134173035621643, Accuracy = 0.8919028043746948
    Iter #2199040:  Learning rate = 0.002122:   Batch Loss = 0.117756, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5129555463790894, Accuracy = 0.8943319916725159
    Iter #2199552:  Learning rate = 0.002122:   Batch Loss = 0.117502, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5100668668746948, Accuracy = 0.8963562846183777
    Iter #2200064:  Learning rate = 0.002037:   Batch Loss = 0.117986, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.514582633972168, Accuracy = 0.8939270973205566
    Iter #2200576:  Learning rate = 0.002037:   Batch Loss = 0.120861, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5103281736373901, Accuracy = 0.8931174278259277
    Iter #2201088:  Learning rate = 0.002037:   Batch Loss = 0.119186, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5119082927703857, Accuracy = 0.8935222625732422
    Iter #2201600:  Learning rate = 0.002037:   Batch Loss = 0.121046, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5180187225341797, Accuracy = 0.8902834057807922
    Iter #2202112:  Learning rate = 0.002037:   Batch Loss = 0.118860, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5118529796600342, Accuracy = 0.8927125334739685
    Iter #2202624:  Learning rate = 0.002037:   Batch Loss = 0.120006, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5169367790222168, Accuracy = 0.892307698726654
    Iter #2203136:  Learning rate = 0.002037:   Batch Loss = 0.119071, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5260602235794067, Accuracy = 0.8898785710334778
    Iter #2203648:  Learning rate = 0.002037:   Batch Loss = 0.121396, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5239948034286499, Accuracy = 0.8919028043746948
    Iter #2204160:  Learning rate = 0.002037:   Batch Loss = 0.116607, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5178415775299072, Accuracy = 0.8902834057807922
    Iter #2204672:  Learning rate = 0.002037:   Batch Loss = 0.117915, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5161306262016296, Accuracy = 0.8898785710334778
    Iter #2205184:  Learning rate = 0.002037:   Batch Loss = 0.118006, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5168554186820984, Accuracy = 0.8914979696273804
    Iter #2205696:  Learning rate = 0.002037:   Batch Loss = 0.118917, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5165235996246338, Accuracy = 0.8914979696273804
    Iter #2206208:  Learning rate = 0.002037:   Batch Loss = 0.119714, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5159167051315308, Accuracy = 0.8902834057807922
    Iter #2206720:  Learning rate = 0.002037:   Batch Loss = 0.118548, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5241553783416748, Accuracy = 0.8919028043746948
    Iter #2207232:  Learning rate = 0.002037:   Batch Loss = 0.117189, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5249577760696411, Accuracy = 0.8919028043746948
    Iter #2207744:  Learning rate = 0.002037:   Batch Loss = 0.118946, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.519059956073761, Accuracy = 0.8927125334739685
    Iter #2208256:  Learning rate = 0.002037:   Batch Loss = 0.118532, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5378351807594299, Accuracy = 0.8870445489883423
    Iter #2208768:  Learning rate = 0.002037:   Batch Loss = 0.120763, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5275181531906128, Accuracy = 0.8935222625732422
    Iter #2209280:  Learning rate = 0.002037:   Batch Loss = 0.119233, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5199160575866699, Accuracy = 0.8898785710334778
    Iter #2209792:  Learning rate = 0.002037:   Batch Loss = 0.116955, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5274385213851929, Accuracy = 0.8886639475822449
    Iter #2210304:  Learning rate = 0.002037:   Batch Loss = 0.115270, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.517606794834137, Accuracy = 0.8919028043746948
    Iter #2210816:  Learning rate = 0.002037:   Batch Loss = 0.115813, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5217839479446411, Accuracy = 0.8919028043746948
    Iter #2211328:  Learning rate = 0.002037:   Batch Loss = 0.118210, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5165742635726929, Accuracy = 0.8947368264198303
    Iter #2211840:  Learning rate = 0.002037:   Batch Loss = 0.116336, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5195257067680359, Accuracy = 0.8910931348800659
    Iter #2212352:  Learning rate = 0.002037:   Batch Loss = 0.118403, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5214135646820068, Accuracy = 0.8919028043746948
    Iter #2212864:  Learning rate = 0.002037:   Batch Loss = 0.119013, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5233702659606934, Accuracy = 0.8943319916725159
    Iter #2213376:  Learning rate = 0.002037:   Batch Loss = 0.117235, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.518190324306488, Accuracy = 0.8935222625732422
    Iter #2213888:  Learning rate = 0.002037:   Batch Loss = 0.114414, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5179921388626099, Accuracy = 0.8931174278259277
    Iter #2214400:  Learning rate = 0.002037:   Batch Loss = 0.117072, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.522344172000885, Accuracy = 0.8882591128349304
    Iter #2214912:  Learning rate = 0.002037:   Batch Loss = 0.114816, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5195437073707581, Accuracy = 0.8894736766815186
    Iter #2215424:  Learning rate = 0.002037:   Batch Loss = 0.118391, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5165838003158569, Accuracy = 0.8914979696273804
    Iter #2215936:  Learning rate = 0.002037:   Batch Loss = 0.116899, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5197462439537048, Accuracy = 0.8870445489883423
    Iter #2216448:  Learning rate = 0.002037:   Batch Loss = 0.114985, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5270615220069885, Accuracy = 0.8906882405281067
    Iter #2216960:  Learning rate = 0.002037:   Batch Loss = 0.119691, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5255520939826965, Accuracy = 0.8894736766815186
    Iter #2217472:  Learning rate = 0.002037:   Batch Loss = 0.116287, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5240967869758606, Accuracy = 0.8914979696273804
    Iter #2217984:  Learning rate = 0.002037:   Batch Loss = 0.116340, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5255404710769653, Accuracy = 0.8910931348800659
    Iter #2218496:  Learning rate = 0.002037:   Batch Loss = 0.116762, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5204349756240845, Accuracy = 0.8931174278259277
    Iter #2219008:  Learning rate = 0.002037:   Batch Loss = 0.371869, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.624565601348877, Accuracy = 0.8538461327552795
    Iter #2219520:  Learning rate = 0.002037:   Batch Loss = 0.954173, Accuracy = 0.765625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.972072958946228, Accuracy = 0.7441295385360718
    Iter #2220032:  Learning rate = 0.002037:   Batch Loss = 0.714509, Accuracy = 0.828125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9329243302345276, Accuracy = 0.7497975826263428
    Iter #2220544:  Learning rate = 0.002037:   Batch Loss = 0.449608, Accuracy = 0.875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.9302337169647217, Accuracy = 0.752226710319519
    Iter #2221056:  Learning rate = 0.002037:   Batch Loss = 0.478781, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8935710191726685, Accuracy = 0.7696356177330017
    Iter #2221568:  Learning rate = 0.002037:   Batch Loss = 0.567947, Accuracy = 0.84375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8071268796920776, Accuracy = 0.7975708246231079
    Iter #2222080:  Learning rate = 0.002037:   Batch Loss = 0.594636, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7956928014755249, Accuracy = 0.8178137540817261
    Iter #2222592:  Learning rate = 0.002037:   Batch Loss = 0.281758, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7142468094825745, Accuracy = 0.8445343971252441
    Iter #2223104:  Learning rate = 0.002037:   Batch Loss = 0.271868, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7100205421447754, Accuracy = 0.8453441262245178
    Iter #2223616:  Learning rate = 0.002037:   Batch Loss = 0.364946, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6491544246673584, Accuracy = 0.8562753200531006
    Iter #2224128:  Learning rate = 0.002037:   Batch Loss = 0.259429, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.687818169593811, Accuracy = 0.8481781482696533
    Iter #2224640:  Learning rate = 0.002037:   Batch Loss = 0.244029, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6481004357337952, Accuracy = 0.8546558618545532
    Iter #2225152:  Learning rate = 0.002037:   Batch Loss = 0.204635, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.659803032875061, Accuracy = 0.8578947186470032
    Iter #2225664:  Learning rate = 0.002037:   Batch Loss = 0.255067, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6650814414024353, Accuracy = 0.8554655909538269
    Iter #2226176:  Learning rate = 0.002037:   Batch Loss = 0.221224, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7179976105690002, Accuracy = 0.8352226614952087
    Iter #2226688:  Learning rate = 0.002037:   Batch Loss = 0.261183, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5959318280220032, Accuracy = 0.8736842274665833
    Iter #2227200:  Learning rate = 0.002037:   Batch Loss = 0.335451, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6968842148780823, Accuracy = 0.8493927121162415
    Iter #2227712:  Learning rate = 0.002037:   Batch Loss = 0.208795, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6470533609390259, Accuracy = 0.8595141768455505
    Iter #2228224:  Learning rate = 0.002037:   Batch Loss = 0.224333, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5947951674461365, Accuracy = 0.8773279190063477
    Iter #2228736:  Learning rate = 0.002037:   Batch Loss = 0.186295, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6039767265319824, Accuracy = 0.8769230842590332
    Iter #2229248:  Learning rate = 0.002037:   Batch Loss = 0.196707, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5599086284637451, Accuracy = 0.8898785710334778
    Iter #2229760:  Learning rate = 0.002037:   Batch Loss = 0.172339, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5913364291191101, Accuracy = 0.8825910687446594
    Iter #2230272:  Learning rate = 0.002037:   Batch Loss = 0.183532, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5675806999206543, Accuracy = 0.8906882405281067
    Iter #2230784:  Learning rate = 0.002037:   Batch Loss = 0.173407, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5510014891624451, Accuracy = 0.8939270973205566
    Iter #2231296:  Learning rate = 0.002037:   Batch Loss = 0.173800, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5491636991500854, Accuracy = 0.8943319916725159
    Iter #2231808:  Learning rate = 0.002037:   Batch Loss = 0.171545, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5523804426193237, Accuracy = 0.8906882405281067
    Iter #2232320:  Learning rate = 0.002037:   Batch Loss = 0.175908, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5475633144378662, Accuracy = 0.8939270973205566
    Iter #2232832:  Learning rate = 0.002037:   Batch Loss = 0.165339, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5316799283027649, Accuracy = 0.9004048705101013
    Iter #2233344:  Learning rate = 0.002037:   Batch Loss = 0.160573, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5208047032356262, Accuracy = 0.9028339982032776
    Iter #2233856:  Learning rate = 0.002037:   Batch Loss = 0.159190, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5398281812667847, Accuracy = 0.8959513902664185
    Iter #2234368:  Learning rate = 0.002037:   Batch Loss = 0.157721, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5330330729484558, Accuracy = 0.8999999761581421
    Iter #2234880:  Learning rate = 0.002037:   Batch Loss = 0.168871, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5201388597488403, Accuracy = 0.9036437273025513
    Iter #2235392:  Learning rate = 0.002037:   Batch Loss = 0.156007, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5174678564071655, Accuracy = 0.9028339982032776
    Iter #2235904:  Learning rate = 0.002037:   Batch Loss = 0.178574, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5214803218841553, Accuracy = 0.8999999761581421
    Iter #2236416:  Learning rate = 0.002037:   Batch Loss = 0.167892, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5371510982513428, Accuracy = 0.8995951414108276
    Iter #2236928:  Learning rate = 0.002037:   Batch Loss = 0.154516, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5085688233375549, Accuracy = 0.9028339982032776
    Iter #2237440:  Learning rate = 0.002037:   Batch Loss = 0.150422, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5052801966667175, Accuracy = 0.9064777493476868
    Iter #2237952:  Learning rate = 0.002037:   Batch Loss = 0.150062, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5136761665344238, Accuracy = 0.9028339982032776
    Iter #2238464:  Learning rate = 0.002037:   Batch Loss = 0.149701, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5067090392112732, Accuracy = 0.9052631855010986
    Iter #2238976:  Learning rate = 0.002037:   Batch Loss = 0.147171, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49539196491241455, Accuracy = 0.9072874784469604
    Iter #2239488:  Learning rate = 0.002037:   Batch Loss = 0.148036, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49398714303970337, Accuracy = 0.9076923131942749
    Iter #2240000:  Learning rate = 0.002037:   Batch Loss = 0.146463, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49560636281967163, Accuracy = 0.9064777493476868
    Iter #2240512:  Learning rate = 0.002037:   Batch Loss = 0.146685, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4919869899749756, Accuracy = 0.9064777493476868
    Iter #2241024:  Learning rate = 0.002037:   Batch Loss = 0.144240, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4923730790615082, Accuracy = 0.9080971479415894
    Iter #2241536:  Learning rate = 0.002037:   Batch Loss = 0.144654, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49082162976264954, Accuracy = 0.9085020422935486
    Iter #2242048:  Learning rate = 0.002037:   Batch Loss = 0.144838, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4854896664619446, Accuracy = 0.9080971479415894
    Iter #2242560:  Learning rate = 0.002037:   Batch Loss = 0.143733, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4826066493988037, Accuracy = 0.9097166061401367
    Iter #2243072:  Learning rate = 0.002037:   Batch Loss = 0.141830, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.480252742767334, Accuracy = 0.908906877040863
    Iter #2243584:  Learning rate = 0.002037:   Batch Loss = 0.143285, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48021718859672546, Accuracy = 0.9097166061401367
    Iter #2244096:  Learning rate = 0.002037:   Batch Loss = 0.141353, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4809321165084839, Accuracy = 0.9093117117881775
    Iter #2244608:  Learning rate = 0.002037:   Batch Loss = 0.141939, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4824804663658142, Accuracy = 0.9076923131942749
    Iter #2245120:  Learning rate = 0.002037:   Batch Loss = 0.139899, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4814874231815338, Accuracy = 0.9072874784469604
    Iter #2245632:  Learning rate = 0.002037:   Batch Loss = 0.142184, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48138946294784546, Accuracy = 0.908906877040863
    Iter #2246144:  Learning rate = 0.002037:   Batch Loss = 0.141083, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4811653196811676, Accuracy = 0.9101214408874512
    Iter #2246656:  Learning rate = 0.002037:   Batch Loss = 0.139997, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4811846911907196, Accuracy = 0.9085020422935486
    Iter #2247168:  Learning rate = 0.002037:   Batch Loss = 0.141036, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.480057954788208, Accuracy = 0.9085020422935486
    Iter #2247680:  Learning rate = 0.002037:   Batch Loss = 0.142254, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.47957855463027954, Accuracy = 0.9097166061401367
    Iter #2248192:  Learning rate = 0.002037:   Batch Loss = 0.138484, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.478346586227417, Accuracy = 0.9080971479415894
    Iter #2248704:  Learning rate = 0.002037:   Batch Loss = 0.138092, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.47717851400375366, Accuracy = 0.9105263352394104
    Iter #2249216:  Learning rate = 0.002037:   Batch Loss = 0.138704, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48094624280929565, Accuracy = 0.9080971479415894
    Iter #2249728:  Learning rate = 0.002037:   Batch Loss = 0.137011, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48394981026649475, Accuracy = 0.9068825840950012
    Iter #2250240:  Learning rate = 0.002037:   Batch Loss = 0.136236, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48222097754478455, Accuracy = 0.9085020422935486
    Iter #2250752:  Learning rate = 0.002037:   Batch Loss = 0.134965, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48021382093429565, Accuracy = 0.9097166061401367
    Iter #2251264:  Learning rate = 0.002037:   Batch Loss = 0.136684, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4795967638492584, Accuracy = 0.9097166061401367
    Iter #2251776:  Learning rate = 0.002037:   Batch Loss = 0.137246, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4787418842315674, Accuracy = 0.9076923131942749
    Iter #2252288:  Learning rate = 0.002037:   Batch Loss = 0.135663, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.47767898440361023, Accuracy = 0.9076923131942749
    Iter #2252800:  Learning rate = 0.002037:   Batch Loss = 0.135492, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.47827911376953125, Accuracy = 0.908906877040863
    Iter #2253312:  Learning rate = 0.002037:   Batch Loss = 0.136724, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4804934561252594, Accuracy = 0.9080971479415894
    Iter #2253824:  Learning rate = 0.002037:   Batch Loss = 0.134755, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48780161142349243, Accuracy = 0.9064777493476868
    Iter #2254336:  Learning rate = 0.002037:   Batch Loss = 0.134261, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4857424199581146, Accuracy = 0.9072874784469604
    Iter #2254848:  Learning rate = 0.002037:   Batch Loss = 0.132524, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4814486503601074, Accuracy = 0.9072874784469604
    Iter #2255360:  Learning rate = 0.002037:   Batch Loss = 0.134565, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4811704456806183, Accuracy = 0.9076923131942749
    Iter #2255872:  Learning rate = 0.002037:   Batch Loss = 0.132873, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48073306679725647, Accuracy = 0.9064777493476868
    Iter #2256384:  Learning rate = 0.002037:   Batch Loss = 0.134197, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48188936710357666, Accuracy = 0.9056680202484131
    Iter #2256896:  Learning rate = 0.002037:   Batch Loss = 0.131655, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4817940592765808, Accuracy = 0.9052631855010986
    Iter #2257408:  Learning rate = 0.002037:   Batch Loss = 0.132124, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4860043227672577, Accuracy = 0.9056680202484131
    Iter #2257920:  Learning rate = 0.002037:   Batch Loss = 0.130926, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4858552813529968, Accuracy = 0.9040485620498657
    Iter #2258432:  Learning rate = 0.002037:   Batch Loss = 0.132530, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48479026556015015, Accuracy = 0.9048582911491394
    Iter #2258944:  Learning rate = 0.002037:   Batch Loss = 0.133354, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48810744285583496, Accuracy = 0.904453456401825
    Iter #2259456:  Learning rate = 0.002037:   Batch Loss = 0.131384, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4861793518066406, Accuracy = 0.9048582911491394
    Iter #2259968:  Learning rate = 0.002037:   Batch Loss = 0.130811, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4893357455730438, Accuracy = 0.9040485620498657
    Iter #2260480:  Learning rate = 0.002037:   Batch Loss = 0.130150, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49347251653671265, Accuracy = 0.9028339982032776
    Iter #2260992:  Learning rate = 0.002037:   Batch Loss = 0.128420, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48973149061203003, Accuracy = 0.9024291634559631
    Iter #2261504:  Learning rate = 0.002037:   Batch Loss = 0.129734, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4839141070842743, Accuracy = 0.9016194343566895
    Iter #2262016:  Learning rate = 0.002037:   Batch Loss = 0.128403, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4815950393676758, Accuracy = 0.9048582911491394
    Iter #2262528:  Learning rate = 0.002037:   Batch Loss = 0.128766, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4835250675678253, Accuracy = 0.9052631855010986
    Iter #2263040:  Learning rate = 0.002037:   Batch Loss = 0.129210, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.486868292093277, Accuracy = 0.9052631855010986
    Iter #2263552:  Learning rate = 0.002037:   Batch Loss = 0.127425, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.485385537147522, Accuracy = 0.9048582911491394
    Iter #2264064:  Learning rate = 0.002037:   Batch Loss = 0.126542, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48625004291534424, Accuracy = 0.9056680202484131
    Iter #2264576:  Learning rate = 0.002037:   Batch Loss = 0.128301, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48870667815208435, Accuracy = 0.9028339982032776
    Iter #2265088:  Learning rate = 0.002037:   Batch Loss = 0.127451, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4830555319786072, Accuracy = 0.9040485620498657
    Iter #2265600:  Learning rate = 0.002037:   Batch Loss = 0.129223, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4869653582572937, Accuracy = 0.904453456401825
    Iter #2266112:  Learning rate = 0.002037:   Batch Loss = 0.125307, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48847338557243347, Accuracy = 0.9036437273025513
    Iter #2266624:  Learning rate = 0.002037:   Batch Loss = 0.126421, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4844534397125244, Accuracy = 0.9032388925552368
    Iter #2267136:  Learning rate = 0.002037:   Batch Loss = 0.125397, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4822119474411011, Accuracy = 0.9064777493476868
    Iter #2267648:  Learning rate = 0.002037:   Batch Loss = 0.126985, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48658105731010437, Accuracy = 0.9040485620498657
    Iter #2268160:  Learning rate = 0.002037:   Batch Loss = 0.126617, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4940785765647888, Accuracy = 0.9024291634559631
    Iter #2268672:  Learning rate = 0.002037:   Batch Loss = 0.124838, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4980047047138214, Accuracy = 0.901214599609375
    Iter #2269184:  Learning rate = 0.002037:   Batch Loss = 0.123283, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5032802820205688, Accuracy = 0.9004048705101013
    Iter #2269696:  Learning rate = 0.002037:   Batch Loss = 0.125648, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4995279312133789, Accuracy = 0.8999999761581421
    Iter #2270208:  Learning rate = 0.002037:   Batch Loss = 0.124457, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48802292346954346, Accuracy = 0.9032388925552368
    Iter #2270720:  Learning rate = 0.002037:   Batch Loss = 0.125991, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4890749454498291, Accuracy = 0.9008097052574158
    Iter #2271232:  Learning rate = 0.002037:   Batch Loss = 0.126300, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4994048476219177, Accuracy = 0.9008097052574158
    Iter #2271744:  Learning rate = 0.002037:   Batch Loss = 0.123208, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4980294704437256, Accuracy = 0.901214599609375
    Iter #2272256:  Learning rate = 0.002037:   Batch Loss = 0.125079, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4916839599609375, Accuracy = 0.9008097052574158
    Iter #2272768:  Learning rate = 0.002037:   Batch Loss = 0.125628, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4917411506175995, Accuracy = 0.9024291634559631
    Iter #2273280:  Learning rate = 0.002037:   Batch Loss = 0.125712, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4931619167327881, Accuracy = 0.9024291634559631
    Iter #2273792:  Learning rate = 0.002037:   Batch Loss = 0.122303, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4939127266407013, Accuracy = 0.9032388925552368
    Iter #2274304:  Learning rate = 0.002037:   Batch Loss = 0.123409, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4933108687400818, Accuracy = 0.9028339982032776
    Iter #2274816:  Learning rate = 0.002037:   Batch Loss = 0.123759, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4951038956642151, Accuracy = 0.9008097052574158
    Iter #2275328:  Learning rate = 0.002037:   Batch Loss = 0.122457, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4974052309989929, Accuracy = 0.8991903066635132
    Iter #2275840:  Learning rate = 0.002037:   Batch Loss = 0.125294, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4925658702850342, Accuracy = 0.901214599609375
    Iter #2276352:  Learning rate = 0.002037:   Batch Loss = 0.121572, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48895689845085144, Accuracy = 0.9024291634559631
    Iter #2276864:  Learning rate = 0.002037:   Batch Loss = 0.124496, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4923948645591736, Accuracy = 0.9036437273025513
    Iter #2277376:  Learning rate = 0.002037:   Batch Loss = 0.121972, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4938732087612152, Accuracy = 0.901214599609375
    Iter #2277888:  Learning rate = 0.002037:   Batch Loss = 0.122986, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4965493381023407, Accuracy = 0.9008097052574158
    Iter #2278400:  Learning rate = 0.002037:   Batch Loss = 0.122694, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4951212704181671, Accuracy = 0.9032388925552368
    Iter #2278912:  Learning rate = 0.002037:   Batch Loss = 0.121485, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.494878888130188, Accuracy = 0.9008097052574158
    Iter #2279424:  Learning rate = 0.002037:   Batch Loss = 0.122617, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49337881803512573, Accuracy = 0.9008097052574158
    Iter #2279936:  Learning rate = 0.002037:   Batch Loss = 0.122840, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4938153028488159, Accuracy = 0.8991903066635132
    Iter #2280448:  Learning rate = 0.002037:   Batch Loss = 0.123086, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49460670351982117, Accuracy = 0.9016194343566895
    Iter #2280960:  Learning rate = 0.002037:   Batch Loss = 0.120468, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49233105778694153, Accuracy = 0.9032388925552368
    Iter #2281472:  Learning rate = 0.002037:   Batch Loss = 0.123532, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4910237193107605, Accuracy = 0.9036437273025513
    Iter #2281984:  Learning rate = 0.002037:   Batch Loss = 0.120284, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4922036826610565, Accuracy = 0.9028339982032776
    Iter #2282496:  Learning rate = 0.002037:   Batch Loss = 0.121621, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49532485008239746, Accuracy = 0.9008097052574158
    Iter #2283008:  Learning rate = 0.002037:   Batch Loss = 0.122730, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49910080432891846, Accuracy = 0.8979756832122803
    Iter #2283520:  Learning rate = 0.002037:   Batch Loss = 0.121085, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49495962262153625, Accuracy = 0.8999999761581421
    Iter #2284032:  Learning rate = 0.002037:   Batch Loss = 0.120419, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49545735120773315, Accuracy = 0.9020242691040039
    Iter #2284544:  Learning rate = 0.002037:   Batch Loss = 0.120161, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49432802200317383, Accuracy = 0.901214599609375
    Iter #2285056:  Learning rate = 0.002037:   Batch Loss = 0.118298, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4974600672721863, Accuracy = 0.9008097052574158
    Iter #2285568:  Learning rate = 0.002037:   Batch Loss = 0.118945, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49459320306777954, Accuracy = 0.9024291634559631
    Iter #2286080:  Learning rate = 0.002037:   Batch Loss = 0.119219, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4942440986633301, Accuracy = 0.9028339982032776
    Iter #2286592:  Learning rate = 0.002037:   Batch Loss = 0.121592, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4976354241371155, Accuracy = 0.9020242691040039
    Iter #2287104:  Learning rate = 0.002037:   Batch Loss = 0.119014, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49969902634620667, Accuracy = 0.898785412311554
    Iter #2287616:  Learning rate = 0.002037:   Batch Loss = 0.119633, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5017621517181396, Accuracy = 0.8999999761581421
    Iter #2288128:  Learning rate = 0.002037:   Batch Loss = 0.118760, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5003347396850586, Accuracy = 0.901214599609375
    Iter #2288640:  Learning rate = 0.002037:   Batch Loss = 0.119137, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5091698169708252, Accuracy = 0.8983805775642395
    Iter #2289152:  Learning rate = 0.002037:   Batch Loss = 0.122118, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5096780061721802, Accuracy = 0.8947368264198303
    Iter #2289664:  Learning rate = 0.002037:   Batch Loss = 0.118161, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5031394362449646, Accuracy = 0.8979756832122803
    Iter #2290176:  Learning rate = 0.002037:   Batch Loss = 0.119170, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49950486421585083, Accuracy = 0.8983805775642395
    Iter #2290688:  Learning rate = 0.002037:   Batch Loss = 0.119472, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49975669384002686, Accuracy = 0.8975708484649658
    Iter #2291200:  Learning rate = 0.002037:   Batch Loss = 0.119338, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5026237964630127, Accuracy = 0.8991903066635132
    Iter #2291712:  Learning rate = 0.002037:   Batch Loss = 0.119875, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5005133152008057, Accuracy = 0.8991903066635132
    Iter #2292224:  Learning rate = 0.002037:   Batch Loss = 0.118849, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5061205625534058, Accuracy = 0.8979756832122803
    Iter #2292736:  Learning rate = 0.002037:   Batch Loss = 0.117506, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49936872720718384, Accuracy = 0.898785412311554
    Iter #2293248:  Learning rate = 0.002037:   Batch Loss = 0.118990, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4971867799758911, Accuracy = 0.9016194343566895
    Iter #2293760:  Learning rate = 0.002037:   Batch Loss = 0.118283, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5019140243530273, Accuracy = 0.8995951414108276
    Iter #2294272:  Learning rate = 0.002037:   Batch Loss = 0.118812, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4982401728630066, Accuracy = 0.8991903066635132
    Iter #2294784:  Learning rate = 0.002037:   Batch Loss = 0.116554, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49924805760383606, Accuracy = 0.8995951414108276
    Iter #2295296:  Learning rate = 0.002037:   Batch Loss = 0.116144, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4978354871273041, Accuracy = 0.8999999761581421
    Iter #2295808:  Learning rate = 0.002037:   Batch Loss = 0.116353, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4963867664337158, Accuracy = 0.901214599609375
    Iter #2296320:  Learning rate = 0.002037:   Batch Loss = 0.119671, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5032625198364258, Accuracy = 0.8999999761581421
    Iter #2296832:  Learning rate = 0.002037:   Batch Loss = 0.120657, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5112998485565186, Accuracy = 0.8991903066635132
    Iter #2297344:  Learning rate = 0.002037:   Batch Loss = 0.119324, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5099797248840332, Accuracy = 0.8947368264198303
    Iter #2297856:  Learning rate = 0.002037:   Batch Loss = 0.117141, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5036180019378662, Accuracy = 0.8943319916725159
    Iter #2298368:  Learning rate = 0.002037:   Batch Loss = 0.117044, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.498121976852417, Accuracy = 0.8991903066635132
    Iter #2298880:  Learning rate = 0.002037:   Batch Loss = 0.116647, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5051865577697754, Accuracy = 0.9008097052574158
    Iter #2299392:  Learning rate = 0.002037:   Batch Loss = 0.117573, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5184465050697327, Accuracy = 0.8983805775642395
    Iter #2299904:  Learning rate = 0.002037:   Batch Loss = 0.118154, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.517467737197876, Accuracy = 0.8975708484649658
    Iter #2300416:  Learning rate = 0.001955:   Batch Loss = 0.118412, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5101675987243652, Accuracy = 0.895546555519104
    Iter #2300928:  Learning rate = 0.001955:   Batch Loss = 0.118502, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49763739109039307, Accuracy = 0.9036437273025513
    Iter #2301440:  Learning rate = 0.001955:   Batch Loss = 0.115695, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5121577978134155, Accuracy = 0.8963562846183777
    Iter #2301952:  Learning rate = 0.001955:   Batch Loss = 0.115144, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5097388625144958, Accuracy = 0.8971660137176514
    Iter #2302464:  Learning rate = 0.001955:   Batch Loss = 0.115218, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5126929879188538, Accuracy = 0.8971660137176514
    Iter #2302976:  Learning rate = 0.001955:   Batch Loss = 0.116207, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5085980296134949, Accuracy = 0.8959513902664185
    Iter #2303488:  Learning rate = 0.001955:   Batch Loss = 0.114745, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5068948268890381, Accuracy = 0.8967611193656921
    Iter #2304000:  Learning rate = 0.001955:   Batch Loss = 0.115010, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5040701627731323, Accuracy = 0.9016194343566895
    Iter #2304512:  Learning rate = 0.001955:   Batch Loss = 0.115075, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5104127526283264, Accuracy = 0.8979756832122803
    Iter #2305024:  Learning rate = 0.001955:   Batch Loss = 0.114093, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5155718922615051, Accuracy = 0.8939270973205566
    Iter #2305536:  Learning rate = 0.001955:   Batch Loss = 0.115221, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.514582097530365, Accuracy = 0.8931174278259277
    Iter #2306048:  Learning rate = 0.001955:   Batch Loss = 0.113727, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.508943498134613, Accuracy = 0.8967611193656921
    Iter #2306560:  Learning rate = 0.001955:   Batch Loss = 0.114489, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5175841450691223, Accuracy = 0.8919028043746948
    Iter #2307072:  Learning rate = 0.001955:   Batch Loss = 0.117271, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5199770331382751, Accuracy = 0.8979756832122803
    Iter #2307584:  Learning rate = 0.001955:   Batch Loss = 0.120093, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5125324130058289, Accuracy = 0.8983805775642395
    Iter #2308096:  Learning rate = 0.001955:   Batch Loss = 0.128364, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5401459336280823, Accuracy = 0.8902834057807922
    Iter #2308608:  Learning rate = 0.001955:   Batch Loss = 0.128787, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5729830861091614, Accuracy = 0.8777328133583069
    Iter #2309120:  Learning rate = 0.001955:   Batch Loss = 0.125473, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5398878455162048, Accuracy = 0.8894736766815186
    Iter #2309632:  Learning rate = 0.001955:   Batch Loss = 0.251434, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8255708813667297, Accuracy = 0.8218623399734497
    Iter #2310144:  Learning rate = 0.001955:   Batch Loss = 0.443576, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.856321394443512, Accuracy = 0.7975708246231079
    Iter #2310656:  Learning rate = 0.001955:   Batch Loss = 0.410390, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.8242758512496948, Accuracy = 0.807692289352417
    Iter #2311168:  Learning rate = 0.001955:   Batch Loss = 0.529498, Accuracy = 0.90625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7923170328140259, Accuracy = 0.8149797320365906
    Iter #2311680:  Learning rate = 0.001955:   Batch Loss = 0.371614, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7749534845352173, Accuracy = 0.8137651681900024
    Iter #2312192:  Learning rate = 0.001955:   Batch Loss = 0.389245, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6985023021697998, Accuracy = 0.8348178267478943
    Iter #2312704:  Learning rate = 0.001955:   Batch Loss = 0.256149, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6761311292648315, Accuracy = 0.8558704257011414
    Iter #2313216:  Learning rate = 0.001955:   Batch Loss = 0.232111, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7363612055778503, Accuracy = 0.8291497826576233
    Iter #2313728:  Learning rate = 0.001955:   Batch Loss = 0.298017, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6637535691261292, Accuracy = 0.852226734161377
    Iter #2314240:  Learning rate = 0.001955:   Batch Loss = 0.288807, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7298513650894165, Accuracy = 0.8384615182876587
    Iter #2314752:  Learning rate = 0.001955:   Batch Loss = 0.272019, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7372530102729797, Accuracy = 0.8303643465042114
    Iter #2315264:  Learning rate = 0.001955:   Batch Loss = 0.255450, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6674165725708008, Accuracy = 0.8526315689086914
    Iter #2315776:  Learning rate = 0.001955:   Batch Loss = 0.321871, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7241215109825134, Accuracy = 0.8392712473869324
    Iter #2316288:  Learning rate = 0.001955:   Batch Loss = 0.389745, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7825808525085449, Accuracy = 0.8194332122802734
    Iter #2316800:  Learning rate = 0.001955:   Batch Loss = 0.332307, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6926785707473755, Accuracy = 0.8497975468635559
    Iter #2317312:  Learning rate = 0.001955:   Batch Loss = 0.301078, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6700779795646667, Accuracy = 0.8587044477462769
    Iter #2317824:  Learning rate = 0.001955:   Batch Loss = 0.225325, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6499133706092834, Accuracy = 0.8668016195297241
    Iter #2318336:  Learning rate = 0.001955:   Batch Loss = 0.219405, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6692191362380981, Accuracy = 0.8603239059448242
    Iter #2318848:  Learning rate = 0.001955:   Batch Loss = 0.200828, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6398583650588989, Accuracy = 0.8647773265838623
    Iter #2319360:  Learning rate = 0.001955:   Batch Loss = 0.254596, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5814957022666931, Accuracy = 0.8846153616905212
    Iter #2319872:  Learning rate = 0.001955:   Batch Loss = 0.194930, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6309899687767029, Accuracy = 0.8728744983673096
    Iter #2320384:  Learning rate = 0.001955:   Batch Loss = 0.186432, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5683233141899109, Accuracy = 0.8931174278259277
    Iter #2320896:  Learning rate = 0.001955:   Batch Loss = 0.199735, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6251481175422668, Accuracy = 0.8708502054214478
    Iter #2321408:  Learning rate = 0.001955:   Batch Loss = 0.181146, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6213779449462891, Accuracy = 0.8748987913131714
    Iter #2321920:  Learning rate = 0.001955:   Batch Loss = 0.176862, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6122736930847168, Accuracy = 0.8829959630966187
    Iter #2322432:  Learning rate = 0.001955:   Batch Loss = 0.275299, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6318360567092896, Accuracy = 0.8748987913131714
    Iter #2322944:  Learning rate = 0.001955:   Batch Loss = 0.191018, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5875033140182495, Accuracy = 0.8838056921958923
    Iter #2323456:  Learning rate = 0.001955:   Batch Loss = 0.184408, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.578871488571167, Accuracy = 0.8846153616905212
    Iter #2323968:  Learning rate = 0.001955:   Batch Loss = 0.178182, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5570852756500244, Accuracy = 0.8975708484649658
    Iter #2324480:  Learning rate = 0.001955:   Batch Loss = 0.197291, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.553554117679596, Accuracy = 0.9004048705101013
    Iter #2324992:  Learning rate = 0.001955:   Batch Loss = 0.165474, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5601625442504883, Accuracy = 0.892307698726654
    Iter #2325504:  Learning rate = 0.001955:   Batch Loss = 0.161237, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5347431898117065, Accuracy = 0.9028339982032776
    Iter #2326016:  Learning rate = 0.001955:   Batch Loss = 0.189103, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5259118676185608, Accuracy = 0.9052631855010986
    Iter #2326528:  Learning rate = 0.001955:   Batch Loss = 0.181732, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5391116142272949, Accuracy = 0.8963562846183777
    Iter #2327040:  Learning rate = 0.001955:   Batch Loss = 0.185717, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5365074872970581, Accuracy = 0.8991903066635132
    Iter #2327552:  Learning rate = 0.001955:   Batch Loss = 0.172365, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5615339279174805, Accuracy = 0.8914979696273804
    Iter #2328064:  Learning rate = 0.001955:   Batch Loss = 0.211984, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5490118861198425, Accuracy = 0.8983805775642395
    Iter #2328576:  Learning rate = 0.001955:   Batch Loss = 0.156382, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5613205432891846, Accuracy = 0.8902834057807922
    Iter #2329088:  Learning rate = 0.001955:   Batch Loss = 0.158180, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5454517006874084, Accuracy = 0.8967611193656921
    Iter #2329600:  Learning rate = 0.001955:   Batch Loss = 0.163022, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5413748025894165, Accuracy = 0.9028339982032776
    Iter #2330112:  Learning rate = 0.001955:   Batch Loss = 0.172914, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5233235955238342, Accuracy = 0.9080971479415894
    Iter #2330624:  Learning rate = 0.001955:   Batch Loss = 0.154520, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5200506448745728, Accuracy = 0.9068825840950012
    Iter #2331136:  Learning rate = 0.001955:   Batch Loss = 0.150323, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5178704261779785, Accuracy = 0.9072874784469604
    Iter #2331648:  Learning rate = 0.001955:   Batch Loss = 0.151285, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5096033215522766, Accuracy = 0.9080971479415894
    Iter #2332160:  Learning rate = 0.001955:   Batch Loss = 0.149625, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5097887516021729, Accuracy = 0.9060728549957275
    Iter #2332672:  Learning rate = 0.001955:   Batch Loss = 0.146215, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49977320432662964, Accuracy = 0.9072874784469604
    Iter #2333184:  Learning rate = 0.001955:   Batch Loss = 0.148940, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49661415815353394, Accuracy = 0.9076923131942749
    Iter #2333696:  Learning rate = 0.001955:   Batch Loss = 0.145532, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4944157600402832, Accuracy = 0.9093117117881775
    Iter #2334208:  Learning rate = 0.001955:   Batch Loss = 0.147202, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4929810166358948, Accuracy = 0.9105263352394104
    Iter #2334720:  Learning rate = 0.001955:   Batch Loss = 0.144961, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4901222586631775, Accuracy = 0.9117408990859985
    Iter #2335232:  Learning rate = 0.001955:   Batch Loss = 0.147481, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4906240999698639, Accuracy = 0.9109311699867249
    Iter #2335744:  Learning rate = 0.001955:   Batch Loss = 0.145284, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4900747537612915, Accuracy = 0.912145733833313
    Iter #2336256:  Learning rate = 0.001955:   Batch Loss = 0.143285, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48937860131263733, Accuracy = 0.9085020422935486
    Iter #2336768:  Learning rate = 0.001955:   Batch Loss = 0.140339, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4896327257156372, Accuracy = 0.9076923131942749
    Iter #2337280:  Learning rate = 0.001955:   Batch Loss = 0.140683, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48804259300231934, Accuracy = 0.9097166061401367
    Iter #2337792:  Learning rate = 0.001955:   Batch Loss = 0.138580, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.489998459815979, Accuracy = 0.9109311699867249
    Iter #2338304:  Learning rate = 0.001955:   Batch Loss = 0.138111, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4919792413711548, Accuracy = 0.9101214408874512
    Iter #2338816:  Learning rate = 0.001955:   Batch Loss = 0.139255, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4922260046005249, Accuracy = 0.908906877040863
    Iter #2339328:  Learning rate = 0.001955:   Batch Loss = 0.137910, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4881558120250702, Accuracy = 0.908906877040863
    Iter #2339840:  Learning rate = 0.001955:   Batch Loss = 0.138772, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4857203960418701, Accuracy = 0.9068825840950012
    Iter #2340352:  Learning rate = 0.001955:   Batch Loss = 0.135105, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48392337560653687, Accuracy = 0.9097166061401367
    Iter #2340864:  Learning rate = 0.001955:   Batch Loss = 0.139950, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48598724603652954, Accuracy = 0.9093117117881775
    Iter #2341376:  Learning rate = 0.001955:   Batch Loss = 0.136204, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4838784337043762, Accuracy = 0.9097166061401367
    Iter #2341888:  Learning rate = 0.001955:   Batch Loss = 0.137977, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48116427659988403, Accuracy = 0.9093117117881775
    Iter #2342400:  Learning rate = 0.001955:   Batch Loss = 0.135325, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.47805890440940857, Accuracy = 0.9080971479415894
    Iter #2342912:  Learning rate = 0.001955:   Batch Loss = 0.137194, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4772038459777832, Accuracy = 0.9080971479415894
    Iter #2343424:  Learning rate = 0.001955:   Batch Loss = 0.135953, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48183414340019226, Accuracy = 0.9076923131942749
    Iter #2343936:  Learning rate = 0.001955:   Batch Loss = 0.137834, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4829341769218445, Accuracy = 0.9068825840950012
    Iter #2344448:  Learning rate = 0.001955:   Batch Loss = 0.135176, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48323845863342285, Accuracy = 0.9052631855010986
    Iter #2344960:  Learning rate = 0.001955:   Batch Loss = 0.133068, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4814966917037964, Accuracy = 0.9064777493476868
    Iter #2345472:  Learning rate = 0.001955:   Batch Loss = 0.136552, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4812156856060028, Accuracy = 0.9076923131942749
    Iter #2345984:  Learning rate = 0.001955:   Batch Loss = 0.132063, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4817980229854584, Accuracy = 0.9068825840950012
    Iter #2346496:  Learning rate = 0.001955:   Batch Loss = 0.133498, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4825744926929474, Accuracy = 0.9085020422935486
    Iter #2347008:  Learning rate = 0.001955:   Batch Loss = 0.133523, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48396217823028564, Accuracy = 0.9085020422935486
    Iter #2347520:  Learning rate = 0.001955:   Batch Loss = 0.131363, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4800264239311218, Accuracy = 0.9064777493476868
    Iter #2348032:  Learning rate = 0.001955:   Batch Loss = 0.131257, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.47767481207847595, Accuracy = 0.9064777493476868
    Iter #2348544:  Learning rate = 0.001955:   Batch Loss = 0.133040, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.476350337266922, Accuracy = 0.9064777493476868
    Iter #2349056:  Learning rate = 0.001955:   Batch Loss = 0.131926, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4802180230617523, Accuracy = 0.908906877040863
    Iter #2349568:  Learning rate = 0.001955:   Batch Loss = 0.132091, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.47916561365127563, Accuracy = 0.9113360047340393
    Iter #2350080:  Learning rate = 0.001955:   Batch Loss = 0.130590, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.480150043964386, Accuracy = 0.9085020422935486
    Iter #2350592:  Learning rate = 0.001955:   Batch Loss = 0.130068, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48050156235694885, Accuracy = 0.9036437273025513
    Iter #2351104:  Learning rate = 0.001955:   Batch Loss = 0.130948, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48195964097976685, Accuracy = 0.904453456401825
    Iter #2351616:  Learning rate = 0.001955:   Batch Loss = 0.133048, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.47955644130706787, Accuracy = 0.9056680202484131
    Iter #2352128:  Learning rate = 0.001955:   Batch Loss = 0.130234, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48068901896476746, Accuracy = 0.9056680202484131
    Iter #2352640:  Learning rate = 0.001955:   Batch Loss = 0.129657, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.47832226753234863, Accuracy = 0.9060728549957275
    Iter #2353152:  Learning rate = 0.001955:   Batch Loss = 0.130834, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4774160087108612, Accuracy = 0.9056680202484131
    Iter #2353664:  Learning rate = 0.001955:   Batch Loss = 0.129628, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.47722959518432617, Accuracy = 0.9036437273025513
    Iter #2354176:  Learning rate = 0.001955:   Batch Loss = 0.129143, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4762629568576813, Accuracy = 0.9040485620498657
    Iter #2354688:  Learning rate = 0.001955:   Batch Loss = 0.128353, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4766635298728943, Accuracy = 0.9048582911491394
    Iter #2355200:  Learning rate = 0.001955:   Batch Loss = 0.128209, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4777460992336273, Accuracy = 0.9052631855010986
    Iter #2355712:  Learning rate = 0.001955:   Batch Loss = 0.130460, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.477728009223938, Accuracy = 0.9032388925552368
    Iter #2356224:  Learning rate = 0.001955:   Batch Loss = 0.126821, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4785574674606323, Accuracy = 0.9052631855010986
    Iter #2356736:  Learning rate = 0.001955:   Batch Loss = 0.127839, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4813529849052429, Accuracy = 0.901214599609375
    Iter #2357248:  Learning rate = 0.001955:   Batch Loss = 0.129527, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4801740050315857, Accuracy = 0.9040485620498657
    Iter #2357760:  Learning rate = 0.001955:   Batch Loss = 0.129291, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4779415726661682, Accuracy = 0.9076923131942749
    Iter #2358272:  Learning rate = 0.001955:   Batch Loss = 0.128414, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4772922992706299, Accuracy = 0.9052631855010986
    Iter #2358784:  Learning rate = 0.001955:   Batch Loss = 0.128831, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4793863594532013, Accuracy = 0.9064777493476868
    Iter #2359296:  Learning rate = 0.001955:   Batch Loss = 0.126755, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.476911723613739, Accuracy = 0.9052631855010986
    Iter #2359808:  Learning rate = 0.001955:   Batch Loss = 0.127301, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.47584474086761475, Accuracy = 0.9056680202484131
    Iter #2360320:  Learning rate = 0.001955:   Batch Loss = 0.126627, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4730588495731354, Accuracy = 0.9068825840950012
    Iter #2360832:  Learning rate = 0.001955:   Batch Loss = 0.127073, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4752005636692047, Accuracy = 0.9048582911491394
    Iter #2361344:  Learning rate = 0.001955:   Batch Loss = 0.125035, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.47885215282440186, Accuracy = 0.9036437273025513
    Iter #2361856:  Learning rate = 0.001955:   Batch Loss = 0.127662, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4810040593147278, Accuracy = 0.9028339982032776
    Iter #2362368:  Learning rate = 0.001955:   Batch Loss = 0.125061, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4807738959789276, Accuracy = 0.9060728549957275
    Iter #2362880:  Learning rate = 0.001955:   Batch Loss = 0.124740, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4767533838748932, Accuracy = 0.9052631855010986
    Iter #2363392:  Learning rate = 0.001955:   Batch Loss = 0.126168, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4810693860054016, Accuracy = 0.9040485620498657
    Iter #2363904:  Learning rate = 0.001955:   Batch Loss = 0.124316, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.481310099363327, Accuracy = 0.9032388925552368
    Iter #2364416:  Learning rate = 0.001955:   Batch Loss = 0.123181, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.476818323135376, Accuracy = 0.9068825840950012
    Iter #2364928:  Learning rate = 0.001955:   Batch Loss = 0.123737, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4810712933540344, Accuracy = 0.9036437273025513
    Iter #2365440:  Learning rate = 0.001955:   Batch Loss = 0.127215, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4834200143814087, Accuracy = 0.9020242691040039
    Iter #2365952:  Learning rate = 0.001955:   Batch Loss = 0.125810, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4789799153804779, Accuracy = 0.9056680202484131
    Iter #2366464:  Learning rate = 0.001955:   Batch Loss = 0.122671, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4805206060409546, Accuracy = 0.9032388925552368
    Iter #2366976:  Learning rate = 0.001955:   Batch Loss = 0.122982, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.481631338596344, Accuracy = 0.9020242691040039
    Iter #2367488:  Learning rate = 0.001955:   Batch Loss = 0.123713, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.47817328572273254, Accuracy = 0.9056680202484131
    Iter #2368000:  Learning rate = 0.001955:   Batch Loss = 0.122877, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4784270226955414, Accuracy = 0.9048582911491394
    Iter #2368512:  Learning rate = 0.001955:   Batch Loss = 0.121213, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4800071120262146, Accuracy = 0.9040485620498657
    Iter #2369024:  Learning rate = 0.001955:   Batch Loss = 0.121191, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4832037389278412, Accuracy = 0.901214599609375
    Iter #2369536:  Learning rate = 0.001955:   Batch Loss = 0.123780, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.480282723903656, Accuracy = 0.9028339982032776
    Iter #2370048:  Learning rate = 0.001955:   Batch Loss = 0.124728, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4744442105293274, Accuracy = 0.9068825840950012
    Iter #2370560:  Learning rate = 0.001955:   Batch Loss = 0.122949, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4756636619567871, Accuracy = 0.9040485620498657
    Iter #2371072:  Learning rate = 0.001955:   Batch Loss = 0.121705, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48445233702659607, Accuracy = 0.9016194343566895
    Iter #2371584:  Learning rate = 0.001955:   Batch Loss = 0.122092, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48409557342529297, Accuracy = 0.9032388925552368
    Iter #2372096:  Learning rate = 0.001955:   Batch Loss = 0.121134, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4758778214454651, Accuracy = 0.9048582911491394
    Iter #2372608:  Learning rate = 0.001955:   Batch Loss = 0.121346, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.47512105107307434, Accuracy = 0.9068825840950012
    Iter #2373120:  Learning rate = 0.001955:   Batch Loss = 0.121962, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4835316836833954, Accuracy = 0.9020242691040039
    Iter #2373632:  Learning rate = 0.001955:   Batch Loss = 0.119970, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4826083481311798, Accuracy = 0.9036437273025513
    Iter #2374144:  Learning rate = 0.001955:   Batch Loss = 0.120372, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4895325303077698, Accuracy = 0.9008097052574158
    Iter #2374656:  Learning rate = 0.001955:   Batch Loss = 0.121601, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4898841679096222, Accuracy = 0.8983805775642395
    Iter #2375168:  Learning rate = 0.001955:   Batch Loss = 0.120540, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4820106625556946, Accuracy = 0.9028339982032776
    Iter #2375680:  Learning rate = 0.001955:   Batch Loss = 0.121357, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4809347093105316, Accuracy = 0.9016194343566895
    Iter #2376192:  Learning rate = 0.001955:   Batch Loss = 0.120856, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4803895652294159, Accuracy = 0.9028339982032776
    Iter #2376704:  Learning rate = 0.001955:   Batch Loss = 0.119753, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4735042452812195, Accuracy = 0.9040485620498657
    Iter #2377216:  Learning rate = 0.001955:   Batch Loss = 0.118551, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4752519130706787, Accuracy = 0.9040485620498657
    Iter #2377728:  Learning rate = 0.001955:   Batch Loss = 0.119113, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48349347710609436, Accuracy = 0.9024291634559631
    Iter #2378240:  Learning rate = 0.001955:   Batch Loss = 0.119136, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49274423718452454, Accuracy = 0.8979756832122803
    Iter #2378752:  Learning rate = 0.001955:   Batch Loss = 0.119298, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4819222092628479, Accuracy = 0.9040485620498657
    Iter #2379264:  Learning rate = 0.001955:   Batch Loss = 0.120049, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4785711467266083, Accuracy = 0.9040485620498657
    Iter #2379776:  Learning rate = 0.001955:   Batch Loss = 0.121394, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48097556829452515, Accuracy = 0.9040485620498657
    Iter #2380288:  Learning rate = 0.001955:   Batch Loss = 0.121140, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48193833231925964, Accuracy = 0.9036437273025513
    Iter #2380800:  Learning rate = 0.001955:   Batch Loss = 0.117942, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.47999173402786255, Accuracy = 0.9036437273025513
    Iter #2381312:  Learning rate = 0.001955:   Batch Loss = 0.118170, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.47977662086486816, Accuracy = 0.9032388925552368
    Iter #2381824:  Learning rate = 0.001955:   Batch Loss = 0.119028, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48092299699783325, Accuracy = 0.9008097052574158
    Iter #2382336:  Learning rate = 0.001955:   Batch Loss = 0.119062, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4850669205188751, Accuracy = 0.8995951414108276
    Iter #2382848:  Learning rate = 0.001955:   Batch Loss = 0.118341, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4784262776374817, Accuracy = 0.9020242691040039
    Iter #2383360:  Learning rate = 0.001955:   Batch Loss = 0.118170, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.47568079829216003, Accuracy = 0.9028339982032776
    Iter #2383872:  Learning rate = 0.001955:   Batch Loss = 0.118186, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.47727423906326294, Accuracy = 0.9028339982032776
    Iter #2384384:  Learning rate = 0.001955:   Batch Loss = 0.117684, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48005980253219604, Accuracy = 0.9040485620498657
    Iter #2384896:  Learning rate = 0.001955:   Batch Loss = 0.116924, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4781610369682312, Accuracy = 0.9008097052574158
    Iter #2385408:  Learning rate = 0.001955:   Batch Loss = 0.116970, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4731970429420471, Accuracy = 0.904453456401825
    Iter #2385920:  Learning rate = 0.001955:   Batch Loss = 0.117816, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.47585341334342957, Accuracy = 0.9028339982032776
    Iter #2386432:  Learning rate = 0.001955:   Batch Loss = 0.117203, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48825711011886597, Accuracy = 0.8995951414108276
    Iter #2386944:  Learning rate = 0.001955:   Batch Loss = 0.116827, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48508521914482117, Accuracy = 0.9016194343566895
    Iter #2387456:  Learning rate = 0.001955:   Batch Loss = 0.117002, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48138800263404846, Accuracy = 0.9016194343566895
    Iter #2387968:  Learning rate = 0.001955:   Batch Loss = 0.116722, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.47951585054397583, Accuracy = 0.8975708484649658
    Iter #2388480:  Learning rate = 0.001955:   Batch Loss = 0.115318, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.47755467891693115, Accuracy = 0.9008097052574158
    Iter #2388992:  Learning rate = 0.001955:   Batch Loss = 0.116372, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.47733038663864136, Accuracy = 0.9016194343566895
    Iter #2389504:  Learning rate = 0.001955:   Batch Loss = 0.115281, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.47842609882354736, Accuracy = 0.9020242691040039
    Iter #2390016:  Learning rate = 0.001955:   Batch Loss = 0.116016, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48086124658584595, Accuracy = 0.9004048705101013
    Iter #2390528:  Learning rate = 0.001955:   Batch Loss = 0.115285, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4819483160972595, Accuracy = 0.898785412311554
    Iter #2391040:  Learning rate = 0.001955:   Batch Loss = 0.115635, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48263081908226013, Accuracy = 0.8995951414108276
    Iter #2391552:  Learning rate = 0.001955:   Batch Loss = 0.114794, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4859335422515869, Accuracy = 0.8975708484649658
    Iter #2392064:  Learning rate = 0.001955:   Batch Loss = 0.114194, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4845350384712219, Accuracy = 0.898785412311554
    Iter #2392576:  Learning rate = 0.001955:   Batch Loss = 0.118276, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4795111119747162, Accuracy = 0.8999999761581421
    Iter #2393088:  Learning rate = 0.001955:   Batch Loss = 0.118173, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.484505832195282, Accuracy = 0.8979756832122803
    Iter #2393600:  Learning rate = 0.001955:   Batch Loss = 0.114914, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48942774534225464, Accuracy = 0.8971660137176514
    Iter #2394112:  Learning rate = 0.001955:   Batch Loss = 0.116806, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48879510164260864, Accuracy = 0.895546555519104
    Iter #2394624:  Learning rate = 0.001955:   Batch Loss = 0.115006, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48158663511276245, Accuracy = 0.9004048705101013
    Iter #2395136:  Learning rate = 0.001955:   Batch Loss = 0.115162, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4844735860824585, Accuracy = 0.8975708484649658
    Iter #2395648:  Learning rate = 0.001955:   Batch Loss = 0.116700, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4827467203140259, Accuracy = 0.8999999761581421
    Iter #2396160:  Learning rate = 0.001955:   Batch Loss = 0.114976, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48260658979415894, Accuracy = 0.8971660137176514
    Iter #2396672:  Learning rate = 0.001955:   Batch Loss = 0.114615, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4783931374549866, Accuracy = 0.8995951414108276
    Iter #2397184:  Learning rate = 0.001955:   Batch Loss = 0.114581, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4824082851409912, Accuracy = 0.8967611193656921
    Iter #2397696:  Learning rate = 0.001955:   Batch Loss = 0.114225, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4822503924369812, Accuracy = 0.8975708484649658
    Iter #2398208:  Learning rate = 0.001955:   Batch Loss = 0.114221, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4819899797439575, Accuracy = 0.8967611193656921
    Iter #2398720:  Learning rate = 0.001955:   Batch Loss = 0.115915, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4806368052959442, Accuracy = 0.8991903066635132
    Iter #2399232:  Learning rate = 0.001955:   Batch Loss = 0.113839, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.486160010099411, Accuracy = 0.8975708484649658
    Iter #2399744:  Learning rate = 0.001955:   Batch Loss = 0.113673, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4913543164730072, Accuracy = 0.8947368264198303
    Iter #2400256:  Learning rate = 0.001877:   Batch Loss = 0.115712, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4913290739059448, Accuracy = 0.8951417207717896
    Iter #2400768:  Learning rate = 0.001877:   Batch Loss = 0.114292, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4891451299190521, Accuracy = 0.8963562846183777
    Iter #2401280:  Learning rate = 0.001877:   Batch Loss = 0.113462, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5034551620483398, Accuracy = 0.8943319916725159
    Iter #2401792:  Learning rate = 0.001877:   Batch Loss = 0.114332, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48908209800720215, Accuracy = 0.895546555519104
    Iter #2402304:  Learning rate = 0.001877:   Batch Loss = 0.115747, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4985349774360657, Accuracy = 0.8939270973205566
    Iter #2402816:  Learning rate = 0.001877:   Batch Loss = 0.112613, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4813355803489685, Accuracy = 0.8995951414108276
    Iter #2403328:  Learning rate = 0.001877:   Batch Loss = 0.111457, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49424663186073303, Accuracy = 0.8931174278259277
    Iter #2403840:  Learning rate = 0.001877:   Batch Loss = 0.115707, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48786336183547974, Accuracy = 0.8943319916725159
    Iter #2404352:  Learning rate = 0.001877:   Batch Loss = 0.112633, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49682551622390747, Accuracy = 0.8959513902664185
    Iter #2404864:  Learning rate = 0.001877:   Batch Loss = 0.112612, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4906633496284485, Accuracy = 0.8939270973205566
    Iter #2405376:  Learning rate = 0.001877:   Batch Loss = 0.114704, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4932314157485962, Accuracy = 0.898785412311554
    Iter #2405888:  Learning rate = 0.001877:   Batch Loss = 0.114206, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.497499942779541, Accuracy = 0.8914979696273804
    Iter #2406400:  Learning rate = 0.001877:   Batch Loss = 0.112746, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49434179067611694, Accuracy = 0.8959513902664185
    Iter #2406912:  Learning rate = 0.001877:   Batch Loss = 0.237728, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7091599106788635, Accuracy = 0.8477732539176941
    Iter #2407424:  Learning rate = 0.001877:   Batch Loss = 0.361934, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7344135642051697, Accuracy = 0.8242915272712708
    Iter #2407936:  Learning rate = 0.001877:   Batch Loss = 0.503434, Accuracy = 0.890625
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7306275963783264, Accuracy = 0.8222672343254089
    Iter #2408448:  Learning rate = 0.001877:   Batch Loss = 0.527740, Accuracy = 0.875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.717353105545044, Accuracy = 0.829959511756897
    Iter #2408960:  Learning rate = 0.001877:   Batch Loss = 0.320457, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7243804335594177, Accuracy = 0.8295546770095825
    Iter #2409472:  Learning rate = 0.001877:   Batch Loss = 0.362087, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7509598135948181, Accuracy = 0.821052610874176
    Iter #2409984:  Learning rate = 0.001877:   Batch Loss = 0.364331, Accuracy = 0.921875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.7397449016571045, Accuracy = 0.8283400535583496
    Iter #2410496:  Learning rate = 0.001877:   Batch Loss = 0.349032, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6359504461288452, Accuracy = 0.8587044477462769
    Iter #2411008:  Learning rate = 0.001877:   Batch Loss = 0.309205, Accuracy = 0.96875
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6334444284439087, Accuracy = 0.8672064542770386
    Iter #2411520:  Learning rate = 0.001877:   Batch Loss = 0.340165, Accuracy = 0.9375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6645464897155762, Accuracy = 0.8554655909538269
    Iter #2412032:  Learning rate = 0.001877:   Batch Loss = 0.307842, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6158589124679565, Accuracy = 0.8631578683853149
    Iter #2412544:  Learning rate = 0.001877:   Batch Loss = 0.327775, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6829296350479126, Accuracy = 0.843319833278656
    Iter #2413056:  Learning rate = 0.001877:   Batch Loss = 0.317982, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6847798824310303, Accuracy = 0.8530364632606506
    Iter #2413568:  Learning rate = 0.001877:   Batch Loss = 0.249051, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6857447028160095, Accuracy = 0.8506072759628296
    Iter #2414080:  Learning rate = 0.001877:   Batch Loss = 0.312169, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6516175866127014, Accuracy = 0.8603239059448242
    Iter #2414592:  Learning rate = 0.001877:   Batch Loss = 0.215701, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6309342384338379, Accuracy = 0.8688259124755859
    Iter #2415104:  Learning rate = 0.001877:   Batch Loss = 0.212236, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6648795008659363, Accuracy = 0.8692307472229004
    Iter #2415616:  Learning rate = 0.001877:   Batch Loss = 0.197981, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6484555006027222, Accuracy = 0.8587044477462769
    Iter #2416128:  Learning rate = 0.001877:   Batch Loss = 0.190880, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5888261198997498, Accuracy = 0.8744939565658569
    Iter #2416640:  Learning rate = 0.001877:   Batch Loss = 0.159086, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6040881872177124, Accuracy = 0.8809716701507568
    Iter #2417152:  Learning rate = 0.001877:   Batch Loss = 0.207864, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5730729103088379, Accuracy = 0.8838056921958923
    Iter #2417664:  Learning rate = 0.001877:   Batch Loss = 0.162954, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.6047241687774658, Accuracy = 0.8736842274665833
    Iter #2418176:  Learning rate = 0.001877:   Batch Loss = 0.167720, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5886029601097107, Accuracy = 0.8838056921958923
    Iter #2418688:  Learning rate = 0.001877:   Batch Loss = 0.160093, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5785871148109436, Accuracy = 0.8846153616905212
    Iter #2419200:  Learning rate = 0.001877:   Batch Loss = 0.162255, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5666640996932983, Accuracy = 0.8866396546363831
    Iter #2419712:  Learning rate = 0.001877:   Batch Loss = 0.163265, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5480965971946716, Accuracy = 0.8935222625732422
    Iter #2420224:  Learning rate = 0.001877:   Batch Loss = 0.283757, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.581510603427887, Accuracy = 0.8894736766815186
    Iter #2420736:  Learning rate = 0.001877:   Batch Loss = 0.177333, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5734069347381592, Accuracy = 0.8870445489883423
    Iter #2421248:  Learning rate = 0.001877:   Batch Loss = 0.157851, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5524405837059021, Accuracy = 0.8914979696273804
    Iter #2421760:  Learning rate = 0.001877:   Batch Loss = 0.152678, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5405606627464294, Accuracy = 0.8967611193656921
    Iter #2422272:  Learning rate = 0.001877:   Batch Loss = 0.152944, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5234419107437134, Accuracy = 0.898785412311554
    Iter #2422784:  Learning rate = 0.001877:   Batch Loss = 0.162676, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5565066337585449, Accuracy = 0.8906882405281067
    Iter #2423296:  Learning rate = 0.001877:   Batch Loss = 0.156876, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5145567655563354, Accuracy = 0.9056680202484131
    Iter #2423808:  Learning rate = 0.001877:   Batch Loss = 0.145999, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5119544267654419, Accuracy = 0.9072874784469604
    Iter #2424320:  Learning rate = 0.001877:   Batch Loss = 0.201356, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5449130535125732, Accuracy = 0.8971660137176514
    Iter #2424832:  Learning rate = 0.001877:   Batch Loss = 0.204409, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.689893901348114, Accuracy = 0.8542510271072388
    Iter #2425344:  Learning rate = 0.001877:   Batch Loss = 0.308258, Accuracy = 0.953125
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5891846418380737, Accuracy = 0.8781376481056213
    Iter #2425856:  Learning rate = 0.001877:   Batch Loss = 0.192881, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5724631547927856, Accuracy = 0.8769230842590332
    Iter #2426368:  Learning rate = 0.001877:   Batch Loss = 0.169530, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5771973133087158, Accuracy = 0.8834007978439331
    Iter #2426880:  Learning rate = 0.001877:   Batch Loss = 0.156039, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5713655352592468, Accuracy = 0.8890688419342041
    Iter #2427392:  Learning rate = 0.001877:   Batch Loss = 0.155969, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5704303979873657, Accuracy = 0.8838056921958923
    Iter #2427904:  Learning rate = 0.001877:   Batch Loss = 0.153754, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.539402961730957, Accuracy = 0.8935222625732422
    Iter #2428416:  Learning rate = 0.001877:   Batch Loss = 0.164345, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5310525894165039, Accuracy = 0.8935222625732422
    Iter #2428928:  Learning rate = 0.001877:   Batch Loss = 0.144730, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.544560432434082, Accuracy = 0.8910931348800659
    Iter #2429440:  Learning rate = 0.001877:   Batch Loss = 0.151513, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5433262586593628, Accuracy = 0.8854250907897949
    Iter #2429952:  Learning rate = 0.001877:   Batch Loss = 0.143451, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5328372716903687, Accuracy = 0.8939270973205566
    Iter #2430464:  Learning rate = 0.001877:   Batch Loss = 0.142313, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.524931013584137, Accuracy = 0.8999999761581421
    Iter #2430976:  Learning rate = 0.001877:   Batch Loss = 0.144791, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5136017799377441, Accuracy = 0.8999999761581421
    Iter #2431488:  Learning rate = 0.001877:   Batch Loss = 0.142473, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5168321132659912, Accuracy = 0.8959513902664185
    Iter #2432000:  Learning rate = 0.001877:   Batch Loss = 0.139934, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5159099698066711, Accuracy = 0.8971660137176514
    Iter #2432512:  Learning rate = 0.001877:   Batch Loss = 0.136845, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5138355493545532, Accuracy = 0.9028339982032776
    Iter #2433024:  Learning rate = 0.001877:   Batch Loss = 0.140852, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5051620006561279, Accuracy = 0.9036437273025513
    Iter #2433536:  Learning rate = 0.001877:   Batch Loss = 0.139416, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49819913506507874, Accuracy = 0.9048582911491394
    Iter #2434048:  Learning rate = 0.001877:   Batch Loss = 0.137216, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4952549338340759, Accuracy = 0.9056680202484131
    Iter #2434560:  Learning rate = 0.001877:   Batch Loss = 0.140751, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5194339752197266, Accuracy = 0.8983805775642395
    Iter #2435072:  Learning rate = 0.001877:   Batch Loss = 0.151853, Accuracy = 0.984375
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49638813734054565, Accuracy = 0.904453456401825
    Iter #2435584:  Learning rate = 0.001877:   Batch Loss = 0.146115, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5087330937385559, Accuracy = 0.8999999761581421
    Iter #2436096:  Learning rate = 0.001877:   Batch Loss = 0.139746, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5047241449356079, Accuracy = 0.9008097052574158
    Iter #2436608:  Learning rate = 0.001877:   Batch Loss = 0.135433, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.5011345744132996, Accuracy = 0.9024291634559631
    Iter #2437120:  Learning rate = 0.001877:   Batch Loss = 0.134690, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49465227127075195, Accuracy = 0.904453456401825
    Iter #2437632:  Learning rate = 0.001877:   Batch Loss = 0.135143, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4916830360889435, Accuracy = 0.9052631855010986
    Iter #2438144:  Learning rate = 0.001877:   Batch Loss = 0.135752, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4903761148452759, Accuracy = 0.9068825840950012
    Iter #2438656:  Learning rate = 0.001877:   Batch Loss = 0.133777, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48945263028144836, Accuracy = 0.9032388925552368
    Iter #2439168:  Learning rate = 0.001877:   Batch Loss = 0.132314, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4874570667743683, Accuracy = 0.9040485620498657
    Iter #2439680:  Learning rate = 0.001877:   Batch Loss = 0.132873, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4865158796310425, Accuracy = 0.904453456401825
    Iter #2440192:  Learning rate = 0.001877:   Batch Loss = 0.132781, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4875546395778656, Accuracy = 0.9028339982032776
    Iter #2440704:  Learning rate = 0.001877:   Batch Loss = 0.134205, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48854008316993713, Accuracy = 0.9036437273025513
    Iter #2441216:  Learning rate = 0.001877:   Batch Loss = 0.132511, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4890504777431488, Accuracy = 0.9036437273025513
    Iter #2441728:  Learning rate = 0.001877:   Batch Loss = 0.132711, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48692572116851807, Accuracy = 0.9032388925552368
    Iter #2442240:  Learning rate = 0.001877:   Batch Loss = 0.130839, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4834791123867035, Accuracy = 0.9048582911491394
    Iter #2442752:  Learning rate = 0.001877:   Batch Loss = 0.132847, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4821193218231201, Accuracy = 0.9048582911491394
    Iter #2443264:  Learning rate = 0.001877:   Batch Loss = 0.130962, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4809584617614746, Accuracy = 0.9056680202484131
    Iter #2443776:  Learning rate = 0.001877:   Batch Loss = 0.130920, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48052501678466797, Accuracy = 0.9085020422935486
    Iter #2444288:  Learning rate = 0.001877:   Batch Loss = 0.129511, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48142024874687195, Accuracy = 0.9060728549957275
    Iter #2444800:  Learning rate = 0.001877:   Batch Loss = 0.130879, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4828273057937622, Accuracy = 0.9076923131942749
    Iter #2445312:  Learning rate = 0.001877:   Batch Loss = 0.129225, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4835449755191803, Accuracy = 0.9056680202484131
    Iter #2445824:  Learning rate = 0.001877:   Batch Loss = 0.129213, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4835091531276703, Accuracy = 0.9028339982032776
    Iter #2446336:  Learning rate = 0.001877:   Batch Loss = 0.130620, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48513761162757874, Accuracy = 0.9028339982032776
    Iter #2446848:  Learning rate = 0.001877:   Batch Loss = 0.127487, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48916834592819214, Accuracy = 0.901214599609375
    Iter #2447360:  Learning rate = 0.001877:   Batch Loss = 0.127819, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4884152412414551, Accuracy = 0.9008097052574158
    Iter #2447872:  Learning rate = 0.001877:   Batch Loss = 0.131520, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48257380723953247, Accuracy = 0.9040485620498657
    Iter #2448384:  Learning rate = 0.001877:   Batch Loss = 0.128410, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.47982123494148254, Accuracy = 0.9040485620498657
    Iter #2448896:  Learning rate = 0.001877:   Batch Loss = 0.127338, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.47894757986068726, Accuracy = 0.9036437273025513
    Iter #2449408:  Learning rate = 0.001877:   Batch Loss = 0.127516, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.47849974036216736, Accuracy = 0.9056680202484131
    Iter #2449920:  Learning rate = 0.001877:   Batch Loss = 0.128801, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48089882731437683, Accuracy = 0.9048582911491394
    Iter #2450432:  Learning rate = 0.001877:   Batch Loss = 0.128919, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48662227392196655, Accuracy = 0.9020242691040039
    Iter #2450944:  Learning rate = 0.001877:   Batch Loss = 0.125269, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48790067434310913, Accuracy = 0.9016194343566895
    Iter #2451456:  Learning rate = 0.001877:   Batch Loss = 0.124632, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4849352240562439, Accuracy = 0.9056680202484131
    Iter #2451968:  Learning rate = 0.001877:   Batch Loss = 0.124735, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4834352731704712, Accuracy = 0.9036437273025513
    Iter #2452480:  Learning rate = 0.001877:   Batch Loss = 0.126198, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4871038496494293, Accuracy = 0.9040485620498657
    Iter #2452992:  Learning rate = 0.001877:   Batch Loss = 0.124955, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49074482917785645, Accuracy = 0.9032388925552368
    Iter #2453504:  Learning rate = 0.001877:   Batch Loss = 0.127213, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4880526661872864, Accuracy = 0.9032388925552368
    Iter #2454016:  Learning rate = 0.001877:   Batch Loss = 0.123764, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4832439720630646, Accuracy = 0.9032388925552368
    Iter #2454528:  Learning rate = 0.001877:   Batch Loss = 0.126361, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48210638761520386, Accuracy = 0.9020242691040039
    Iter #2455040:  Learning rate = 0.001877:   Batch Loss = 0.125480, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4845232665538788, Accuracy = 0.9024291634559631
    Iter #2455552:  Learning rate = 0.001877:   Batch Loss = 0.122987, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48495161533355713, Accuracy = 0.9020242691040039
    Iter #2456064:  Learning rate = 0.001877:   Batch Loss = 0.123170, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48305872082710266, Accuracy = 0.9040485620498657
    Iter #2456576:  Learning rate = 0.001877:   Batch Loss = 0.127324, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4835718274116516, Accuracy = 0.9024291634559631
    Iter #2457088:  Learning rate = 0.001877:   Batch Loss = 0.123338, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48474156856536865, Accuracy = 0.9016194343566895
    Iter #2457600:  Learning rate = 0.001877:   Batch Loss = 0.124893, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48721277713775635, Accuracy = 0.8999999761581421
    Iter #2458112:  Learning rate = 0.001877:   Batch Loss = 0.122545, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4869239032268524, Accuracy = 0.8999999761581421
    Iter #2458624:  Learning rate = 0.001877:   Batch Loss = 0.124172, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48510444164276123, Accuracy = 0.904453456401825
    Iter #2459136:  Learning rate = 0.001877:   Batch Loss = 0.124134, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.484699010848999, Accuracy = 0.901214599609375
    Iter #2459648:  Learning rate = 0.001877:   Batch Loss = 0.120553, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.486512154340744, Accuracy = 0.8995951414108276
    Iter #2460160:  Learning rate = 0.001877:   Batch Loss = 0.122086, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.49023303389549255, Accuracy = 0.8971660137176514
    Iter #2460672:  Learning rate = 0.001877:   Batch Loss = 0.122937, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4916374087333679, Accuracy = 0.9004048705101013
    Iter #2461184:  Learning rate = 0.001877:   Batch Loss = 0.120538, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4889776408672333, Accuracy = 0.9004048705101013
    Iter #2461696:  Learning rate = 0.001877:   Batch Loss = 0.120587, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4879259169101715, Accuracy = 0.8979756832122803
    Iter #2462208:  Learning rate = 0.001877:   Batch Loss = 0.120697, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.485079824924469, Accuracy = 0.9024291634559631
    Iter #2462720:  Learning rate = 0.001877:   Batch Loss = 0.124193, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48356759548187256, Accuracy = 0.904453456401825
    Iter #2463232:  Learning rate = 0.001877:   Batch Loss = 0.120607, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48303693532943726, Accuracy = 0.9016194343566895
    Iter #2463744:  Learning rate = 0.001877:   Batch Loss = 0.123089, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.48314374685287476, Accuracy = 0.9008097052574158
    Iter #2464256:  Learning rate = 0.001877:   Batch Loss = 0.121594, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4823577404022217, Accuracy = 0.9032388925552368
    Iter #2464768:  Learning rate = 0.001877:   Batch Loss = 0.121252, Accuracy = 1.0
    PERFORMANCE ON TEST SET:             Batch Loss = 0.4877227246761322, Accuracy = 0.9004048705101013
    


```python
##### Check if you want to save your current model
if update:
    save_path = saver.save(sess, DATASET_PATH + "model.ckpt")
    print("Model saved in file: %s" % save_path)
```


```python
##### Inferencing

# X_infer_path = "utilities/something/something.txt"
X_infer_path = DATASET_PATH + "X_test.txt"

X_val = load_X(X_infer_path)

preds = sess.run(
    [pred],
    feed_dict={
        x: X_val
   }
)

print(preds)
```

## Results:




```python
# (Inline plots: )
%matplotlib inline

font = {
    'family' : 'Bitstream Vera Sans',
    'weight' : 'bold',
    'size'   : 18
}
matplotlib.rc('font', **font)

width = 12
height = 12
plt.figure(figsize=(width, height))

indep_train_axis = np.array(range(batch_size, (len(train_losses)+1)*batch_size, batch_size))
#plt.plot(indep_train_axis, np.array(train_losses),     "b--", label="Train losses")
plt.plot(indep_train_axis, np.array(train_accuracies), "g--", label="Train accuracies")

indep_test_axis = np.append(
    np.array(range(batch_size, len(test_losses)*display_iter, display_iter)[:-1]),
    [training_iters]
)
#plt.plot(indep_test_axis, np.array(test_losses), "b-", linewidth=2.0, label="Test losses")
plt.plot(indep_test_axis, np.array(test_accuracies), "b-", linewidth=2.0, label="Test accuracies")
print(len(test_accuracies))
print(len(train_accuracies))

plt.title("Training session's Accuracy over Iterations")
plt.legend(loc='lower right', shadow=True)
plt.ylabel('Training Accuracy')
plt.xlabel('Training Iteration')

plt.show()

# Results

predictions = one_hot_predictions.argmax(1)

print("Testing Accuracy: {}%".format(100*accuracy_fin))

print("")
print("Precision: {}%".format(100*metrics.precision_score(y_test, predictions, average="weighted")))
print("Recall: {}%".format(100*metrics.recall_score(y_test, predictions, average="weighted")))
print("f1_score: {}%".format(100*metrics.f1_score(y_test, predictions, average="weighted")))

print("")
print("Confusion Matrix:")
print("Created using test set of {} datapoints, normalised to % of each class in the test dataset".format(len(y_test)))
confusion_matrix = metrics.confusion_matrix(y_test, predictions)


#print(confusion_matrix)
normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100


# Plot Results: 
width = 12
height = 12
plt.figure(figsize=(width, height))
plt.imshow(
    normalised_confusion_matrix, 
    interpolation='nearest', 
    cmap=plt.cm.Blues
)
plt.title("Confusion matrix \n(normalised to % of total test data)")
plt.colorbar()
tick_marks = np.arange(n_classes)
plt.xticks(tick_marks, LABELS, rotation=90)
plt.yticks(tick_marks, LABELS)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

```


```python


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
```


```python
#sess.close()
print(test_accuracies)
```

## Conclusion

Final accuracy of >90% is pretty good, considering that training takes about 7 minutes.

Noticeable confusion between activities of Clapping Hands and Boxing, and between Jumping Jacks and Waving Two Hands which is understandable.

In terms of the applicability of this to a wider dataset, I would imagine that it would be able to work for any activities in which the training included a views from all angles to be tested on. It would be interesting to see it's applicability to camera angles in between the 4 used in this dataset, without training on them specifically.

 Overall, this experiment validates the idea that 2D pose can be used for at least human activity recognition, and provides verification to continue onto use of 2D pose for behaviour estimation in both people and animals
 

 ### With regards to Using LSTM-RNNs
 - Batch sampling
     - It is neccessary to ensure you are not just sampling classes one at a time! (ie y_train is ordered by class and batch chosen in order)The use of random sampling of batches without replacement from the training data resolves this.    
 
 - Architecture
     - Testing has been run using a variety of hidden units per LSTM cell, with results showing that testing accuracy achieves a higher score when using a number of hidden cells approximately equal to that of the input, ie 34. The following figure displays the final accuracy achieved on the testing dataset for a variety of hidden units, all using a batch size of 4096 and 300 epochs (a total of 1657 iterations, with testing performed every 8th iteration).
   
 
 

## Future Works

Inclusion of :

 - A pipeline for qualitative results
 - A validation dataset
 - Momentum     
 - Normalise input data (each point with respect to distribution of itself only)
 - Dropout
 - Comparison of effect of changing batch size
 

Further research will be made into the use on more subtle activity classes, such as walking versus running, agitated movement versus calm movement, and perhaps normal versus abnormal behaviour, based on a baseline of normal motion.


## References

The dataset can be found at http://tele-immersion.citris-uc.org/berkeley_mhad released under the BSD-2 license
>Copyright (c) 2013, Regents of the University of California All rights reserved.

The network used in this experiment is based on the following, available under the [MIT License](https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition/blob/master/LICENSE). :
> Guillaume Chevalier, LSTMs for Human Activity Recognition, 2016
> https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition




```python
# Let's convert this notebook to a README for the GitHub project's title page:
!jupyter nbconvert --to markdown LSTM.ipynb
!mv LSTM.md README.md
```

## 


```python
#### Camera

# import argparse
# import logging
# import time

# import cv2
# import numpy as np

# from tf_pose.estimator import TfPoseEstimator
# from tf_pose.networks import get_graph_path, model_wh

# logger = logging.getLogger('TfPoseEstimator-WebCam')
# logger.setLevel(logging.DEBUG)
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
# formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
# ch.setFormatter(formatter)
# logger.addHandler(ch)

# class openpose:
#     def __init__(self, camera=0,resize='0x0',resize_out_ratio=4.0,model='mobilenet_thin',show_process=False):

#         logger.debug('initialization %s : %s' % (model, get_graph_path(model)))
#         w, h = model_wh(resize)
#         if w > 0 and h > 0:
#             e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))
#         else:
#             e = TfPoseEstimator(get_graph_path(model), target_size=(432, 368))
#         logger.debug('cam read+')
#         cam = cv2.VideoCapture(camera)
#         ret_val, image = cam.read()
#         logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
        
#         fps_time = 0

#         while True:
#             ret_val, image = cam.read()

#             logger.debug('image process+')
#             humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)

#             logger.debug('postprocess+')
#             image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

#             logger.debug('show+')
#             cv2.putText(image,
#                         "FPS: %f" % (1.0 / (time.time() - fps_time)),
#                         (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                         (0, 255, 0), 2)
#             cv2.imshow('tf-pose-estimation result', image)
#             fps_time = time.time()
#             if cv2.waitKey(1) == 27:
#                 break
#             logger.debug('finished+')

#         cv2.destroyAllWindows()

# if __name__ == '__main__':
#     openpose()

```
