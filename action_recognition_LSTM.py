import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.image as mpimg
import random
import pickle
import scipy.io as sio
from numpy.random import seed
from numpy.random import randint
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
seed(1)
# create the label vector 
def smooth(y,box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y,box,mode='same')
    return y_smooth

#loading data and label 
data_file = open("youtube_action_train_data_part1.pkl","rb")
part1_data,part1_labels = pickle.load(data_file,encoding='latin1')
data_file.close()
data_file = open("youtube_action_train_data_part2.pkl","rb")
part2_data,part2_labels = pickle.load(data_file,encoding='latin1')
all_data = np.append(part1_data,part2_data,axis = 0)
all_labels = np.append(part1_labels,part2_labels)
data_file.close()
#Randomizing the order of the data
all_data,all_labels = shuffle(all_data,all_labels)

#Splitting Data into train and test set
train_data = all_data[:5000,:,:,:,:]
train_labels = all_labels[:5000]
test_data = all_data[5000:,:,:,:,:]
test_labels = all_labels[5000:]

#data normalization 
#Hyperparameter Setting 
batch_size = 20
max_training_epochs = 10000
display_step = 100
length_train = len(train_data)
length_test = len(test_data)

#%----------------Forward Propagation----------------------
tf.reset_default_graph()
sigma_init = 0.1
input_frames = tf.placeholder(dtype=tf.uint8,shape = [None, 30, 64, 64,3],name = 'input_frames')
true_lbl = tf.placeholder(dtype=tf.int64,shape = [None],name = 'true_lbl')


#Data Preprocessing by scaling and normalizing
input_frms_float = tf.dtypes.cast(input_frames,tf.float32,name = 'input_frms_float')
mean_input = tf.math.reduce_mean(input_frms_float,axis=(2,3,4),keepdims=True)
std_input = tf.math.reduce_variance(input_frms_float,axis=(2,3,4),keepdims=True)
x = tf.nn.batch_normalization(input_frms_float,mean_input,std_input,0,1,1e-4)

#-------------------CNN----------------------------------
#Weigths of the First Convolutional Layer 
W1 = tf.Variable(tf.random_normal([5,5,3,32],mean=0,stddev=1/(5*5*3),dtype='float32'),name='W1')
W1_0 = tf.Variable(tf.zeros([32],dtype='float32'),name='W1_0')

#Weights of the Second Convolutional Layer
W2 = tf.Variable(tf.random_normal([5,5,32,32],mean=0,stddev=1/(5*5*3),dtype='float32'),name='W2')
W2_0 = tf.Variable(tf.zeros([32],dtype='float32'),name='W2_0')

#Weights of the Third Convolutional Layer
W3 = tf.Variable(tf.random_normal([3,3,32,64],mean=0,stddev=1/(3*3*32),dtype='float32'),name='W3')
W3_0 = tf.Variable(tf.zeros([64],dtype='float32'),name='W3_0')
Fc_list = []
for i in range(30):
    xi = x[:,i,:,:,:]
    Conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(xi,W1,strides=[1,1,1,1],padding='VALID'),W1_0,name='Conv1'))
    Pool1 = tf.nn.max_pool2d(Conv1,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='VALID',name='Pool1')
    Conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(Pool1,W2,strides=[1,1,1,1],padding='VALID'),W2_0,name='Conv2'))
    Pool2 = tf.nn.max_pool2d(Conv2,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='VALID',name='Pool2')
    Conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(Pool2,W3,strides=[1,1,1,1],padding='VALID'),W3_0,name='Conv3'))
    Fc_list.append(tf.reshape(Conv3,[-1,11*11*64],name='Fc'))
         

#-----------------------------LSTM-----------------------------
rnn_input = tf.stack(Fc_list,1)

#instantiate a LSTM model
num_units = 28
lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units)
h_val, _ = tf.nn.dynamic_rnn(lstm_cell,rnn_input,dtype=tf.float32)

Wc = tf.Variable(tf.random_normal([num_units,11],mean=0,stddev=1/(num_units),dtype='float32'),name='Wc')
bc = tf.Variable(tf.zeros([11],dtype='float32'),name='bc')


#collection of all the final output 
logits = tf.add(tf.matmul(h_val[:,29,:],Wc),bc)
probability = tf.nn.softmax(logits,name='probability')
action_lbl = tf.argmax(probability,axis=1,name='action_lbl')

#Create the collection 
tf.get_collection("validation_nodes")
#Add stuff to the collection 
tf.add_to_collection("validation_nodes",input_frames)
tf.add_to_collection("validation_nodes", action_lbl)

saver = tf.train.Saver()



#%-----------------Loss Function-------------------------------------------
loss_func = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_lbl,logits=logits,name='Loss')
cost = tf.div(tf.reduce_sum(loss_func),batch_size)
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_func)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
init = tf.global_variables_initializer()

train_err_list = []
test_err_list = []
train_acc_list = [] 
test_acc_list = []
train_loss_list = [] 
test_loss_list = []
with tf.Session() as sess:
    sess.run(init)
    epoch = 0 
    while epoch < max_training_epochs:

        #Random batch for SGD 
        random_inds = randint(0,length_train-1,batch_size) 
        batch_x = train_data[random_inds,:]
        batch_y = train_labels[random_inds]
        
        feed_dict_train = {input_frames:batch_x,true_lbl:batch_y};
        train_step.run(feed_dict=feed_dict_train)
        #print("Epoch number " + str(epoch))
        train_loss = cost.eval(feed_dict_train)
       
        #Training Error
        predictions = sess.run(action_lbl, feed_dict = {input_frames:batch_x})
        train_acc = np.sum(predictions==batch_y)*100/len(predictions)
        #print("Training Accuracy is " + str(train_acc))

        #Testing Error
        random_inds = randint(0,length_test-1,batch_size)
        test_x = test_data[random_inds,:]
        test_y = test_labels[random_inds]
        feed_dict_test = {input_frames:test_x,true_lbl:test_y};
        test_loss = cost.eval(feed_dict_test)
        predictions = sess.run(action_lbl, feed_dict = {input_frames: test_x})
        test_acc = np.sum(predictions==test_y)*100/len(predictions)
        #print("Testing Accuracy is " + str(test_acc))

        if (epoch+1) % display_step == 0:
            print("Epoch number " + str(epoch))
            print("Training Loss is" +str(train_loss))
            print("Testing Loss is" + str(test_loss))
            print("Training accuracy is " + str(train_acc))
            print("Testing accuracy is "+ str(test_acc))
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
        epoch = epoch + 1
    save_path = saver.save(sess,"my_model")
    random_inds = randint(0,length_test-1,100)
    test_x = test_data[random_inds,:]
    test_y = test_labels[random_inds]
    predictions = sess.run(action_lbl, feed_dict = {input_frames: test_x})
    labels = test_y
#%% plot the overall training and testing  accuracy
plt.plot(train_acc_list,color='r',label = 'Training')
plt.plot(test_acc_list,color='g',label = 'Testing')
plt.xlabel('per 100 iterations')
plt.ylabel('Overall Classification Accuracy (%)')
plt.legend(loc='best')
plt.savefig('average_acc_train_test.jpg')
plt.close()
#%% plot the overall training and testing loss
plt.plot(train_loss_list,color='r',label = 'Training')
plt.plot(test_loss_list,color='g',label = 'Testing')
plt.xlabel('per 100 iterations')
plt.ylabel('Cross Entropy Loss')
plt.legend(loc='best')
plt.savefig('loss_train_test.jpg')
plt.close()
sio.savemat('result_for_conf.mat', {'predictions':predictions,'true_lbls':labels})
exit()

