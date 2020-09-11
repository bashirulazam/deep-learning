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
seed(1)

#loading data and label 
data_file = open("youtube_train_data.pkl","rb")
all_data,all_labels = pickle.load(data_file,encoding='latin1')
train_data = all_data[:6000,:,:,:,:]
train_labels = all_labels[:6000,:,:,:]
test_data = all_data[6000:,:,:,:,:]
test_labels = all_labels[6000:,:,:,:]
data_file.close()

#data normalization 
#Hyperparameter Setting 
batch_size = 5
max_training_epochs = 15000
display_step = 100
length_train = len(train_data)
length_test = len(test_data)

#%----------------Forward Propagation----------------------
tf.reset_default_graph()
sigma_init = 0.1
input_frames = tf.placeholder(dtype=tf.uint8,shape = [None, 10, 64, 64,3],name = 'input_frames')
true_output = tf.placeholder(dtype=tf.float32,shape = [None,10,7,2],name = 'true_output')


#Data Preprocessing by scaling and normalizing
input_frms_float = tf.dtypes.cast(input_frames,tf.float32,name = 'input_frms_float')
mean_input = tf.math.reduce_mean(input_frms_float,axis=(2,3,4),keepdims=True)
std_input = tf.math.reduce_variance(input_frms_float,axis=(2,3,4),keepdims=True)
x = tf.nn.batch_normalization(input_frms_float,mean_input,std_input,0,1,1e-4)

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
for i in range(10):
    xi = x[:,i,:,:,:]
    Conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(xi,W1,strides=[1,1,1,1],padding='VALID'),W1_0,name='Conv1'))
    Pool1 = tf.nn.max_pool2d(Conv1,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='VALID',name='Pool1')
    Conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(Pool1,W2,strides=[1,1,1,1],padding='VALID'),W2_0,name='Conv2'))
    Pool2 = tf.nn.max_pool2d(Conv2,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='VALID',name='Pool2')
    Conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(Pool2,W3,strides=[1,1,1,1],padding='VALID'),W3_0,name='Conv3'))
    Fc_list.append(tf.reshape(Conv3,[-1,11*11*64],name='Fc'))
         
rnn_input = tf.stack(Fc_list,1)

#instantiate a LSTM model
num_units = 42
lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units)
h_val, _ = tf.nn.dynamic_rnn(lstm_cell,rnn_input,dtype=tf.float32)

#Layer between Hidden State and Output
Wc = tf.Variable(tf.random_normal([num_units,14],mean=0,stddev=1/(num_units),dtype='float32'),name='Wc')
bc = tf.Variable(tf.zeros([14],dtype='float32'),name='bc')


#collection of all the final output 
final_output_list = []
for i in range(10):
    output = tf.add(tf.matmul(h_val[:,i,:],Wc),bc)
    output = tf.reshape(output,[-1,7,2])
    final_output_list.append(output)

final_output = tf.stack(final_output_list,1)
joint_pos = tf.identity(final_output,name='joint_pos')

#Create the collection 
tf.get_collection("validation_nodes")
#Add stuff to the collection 
tf.add_to_collection("validation_nodes",input_frames)
tf.add_to_collection("validation_nodes", joint_pos)

saver = tf.train.Saver()



#%-----------------Loss Function-------------------------------------------
loss_func = tf.compat.v1.losses.mean_squared_error(labels=true_output,predictions=joint_pos)
cost = tf.div(tf.reduce_sum(loss_func),batch_size)
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_func)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
init = tf.global_variables_initializer()

train_err_list = []
test_err_list = []
with tf.Session() as sess:
    sess.run(init)
    train_err = []
    test_err = []
    cost_list = []
    epoch = 0 
    while epoch < max_training_epochs:

        #Random batchfor SGD 
        random_inds = randint(0,length_train-1,batch_size) 
        batch_x = train_data[random_inds,:]
        batch_y = train_labels[random_inds]
        
        feed_dict_train = {input_frames:batch_x,true_output:batch_y};
        train_step.run(feed_dict=feed_dict_train)
        #print("Epoch number " + str(epoch))
        cost_ep = cost.eval(feed_dict_train)
        #print(cost_ep)
        cost_list.append(cost_ep)
        
        #Training Error
        predictions = sess.run(joint_pos, feed_dict = {input_frames:batch_x})
        labels = batch_y
        train_err = np.mean(np.linalg.norm(predictions.reshape((-1,2)) - labels.reshape((-1,2)), axis = 1))

        #Testing Error
        random_inds = randint(0,length_test-1,batch_size)
        test_x = test_data[random_inds,:]
        test_y = test_labels[random_inds]
        predictions = sess.run(joint_pos, feed_dict = {input_frames: test_x})
        labels = test_y
        test_err = np.mean(np.linalg.norm(predictions.reshape((-1,2)) - labels.reshape((-1,2)), axis = 1))
        if (epoch+1) % display_step == 0:
            print("Epoch number " + str(epoch))
            print(cost_ep)
            print("Training Error is " + str(train_err))
            print("Testing Error is "+ str(test_err))
            train_err_list.append(train_err)
            test_err_list.append(test_err)
        epoch = epoch + 1
    save_path = saver.save(sess,"my_model")
    random_inds = randint(0,length_test-1,100)
    test_x = test_data[random_inds,:]
    test_y = test_labels[random_inds]
    predictions = sess.run(joint_pos, feed_dict = {input_frames: test_x})
    labels = test_y
    joint_err = np.mean(np.reshape(np.sqrt(np.sum(np.square(labels - predictions),axis=3)),[-1,7]),axis = 0)
print(joint_err)
#%% plot the training error
#%% plot the overall training accuracy
plt.plot(train_err_list)
plt.xlabel('per 100 iterations')
plt.ylabel('Average Pixel Distance error for training')
plt.savefig('average_error_train.jpg')
plt.close()

#%% plot the overall testing error
plt.plot(test_err_list)
plt.xlabel('per 100 iterations')
plt.ylabel('Average Pixel Distance error for testing')
plt.savefig('average_error_test.jpg')
plt.close()

#Visualization of Pose Estimation
figs,axs = plt.subplots(2,3,constrained_layout = True)
for i in range(6):
    image = test_x[0][i]
    r = np.int32(np.divide(i,3))
    c = np.int32(np.mod(i,3))
    axs[r,c].set_title(str(i))
    axs[r,c].imshow(image)
    pred = np.round(predictions[0][i])
    gt = np.round(labels[0][i])
    for j in range(7):
        axs[r,c].scatter(pred[j][0],pred[j][1],s=100,c='red',marker='x')
        axs[r,c].scatter(gt[j][0],gt[j][1],s=100,c='blue',marker='x')

figs.suptitle('Sequence of Images with Body pose markers')
#plt.show()
plt.savefig('body_pose.jpg')
plt.close()
joint_err_all = np.reshape(np.sqrt(np.sum(np.square(labels - predictions),axis=3)),[-1,7])

my_color = ['b', 'g', 'r', 'c','m','y','k']
my_label = ['head', 'right shoulder', 'left shoulder', 'right wrist', 'left wrist' , 'right elbow', 'left elbow']
for i in range(6):
    joint_err = np.array(joint_err_all[:,i])
    acc_list = []
    for dev in range(21):
        acc = np.sum(joint_err < dev )/ 1000 
        acc_list.append(acc)
    plt.plot(acc_list,color = my_color[i],label = my_label[i])
plt.legend(loc='best')
plt.ylabel('Accuracy[%]')
plt.xlabel('pixel distance from GT')
plt.title('prediction accuracy within 20 pixel')
plt.savefig('accuracy_curve.jpg')
exit()



