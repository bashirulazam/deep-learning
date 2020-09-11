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
def smooth(y,box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y,box,mode='same')
    return y_smooth

#loading data and label 
data_file = open("cifar_10_tf_train_test.pkl","rb")
train_x, train_y, test_x, test_y = pickle.load(data_file,encoding='latin1')
train_y = np.array(train_y)
test_y = np.array(test_y)
data_file.close()

#Hyperparameter Setting 
batch_size = 100
max_training_epochs = 120000
display_step = 100
length_train = len(train_x)
length_test = len(test_x)

#%----------------Forward Propagation----------------------
tf.reset_default_graph()
sigma_init = 0.1
input_img = tf.placeholder(dtype=tf.uint8,shape = [None, 32, 32, 3],name = 'input_img')
true_lbl = tf.placeholder(dtype=tf.int64,shape = [None],name = 'true_lbl')

#Data Preprocessing by scaling and normalizing
input_img_float = tf.dtypes.cast(input_img,tf.float32,name = 'input_img_float')
input_img_scaled = tf.divide(input_img_float,255.0,name='input_img_scaled')
x = tf.nn.batch_normalization(input_img_scaled,tf.reduce_mean(input_img_scaled,0),1,0,1,1e-4)

#First Convolutional Layer 
W1 = tf.Variable(tf.random_normal([5,5,3,32],mean=0,stddev=1/(5*5*3),dtype='float32'),name='W1')
W1_0 = tf.Variable(tf.zeros([32],dtype='float32'),name='W1_0')
Conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x,W1,strides=[1,1,1,1],padding='VALID'),W1_0,name='Conv1'))
Pool1 = tf.nn.max_pool2d(Conv1,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='VALID',name='Pool1')

#Second Convolutional Layer
W2 = tf.Variable(tf.random_normal([5,5,32,32],mean=0,stddev=1/(5*5*3),dtype='float32'),name='W2')
W2_0 = tf.Variable(tf.zeros([32],dtype='float32'),name='W2_0')
Conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(Pool1,W2,strides=[1,1,1,1],padding='VALID'),W2_0,name='Conv2'))
Pool2 = tf.nn.max_pool2d(Conv2,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='VALID',name='Pool2')

#Third Convolutional Layer
W3 = tf.Variable(tf.random_normal([3,3,32,64],mean=0,stddev=1/(3*3*32),dtype='float32'),name='W3')
W3_0 = tf.Variable(tf.zeros([64],dtype='float32'),name='W3_0')
Conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(Pool2,W3,strides=[1,1,1,1],padding='VALID'),W3_0,name='Conv3'))

#Fully Connected Layer
Fc = tf.reshape(Conv3,[-1,3*3*64],name='Fc')

#Output and Predicted Label 
W4 = tf.Variable(tf.random_normal([3*3*64,10],mean=0,stddev=1/(3*3*64),dtype='float32'),name='W3')
W4_0 = tf.Variable(tf.zeros([10],dtype='float32'),name='W4_0')
logits = tf.add(tf.matmul(Fc,W4),W4_0)
softmax_op = tf.nn.softmax(logits,name='softmax_op')
predict_lbl = tf.argmax(softmax_op,axis=1,name='predict_lbl')

#Create the collection 
tf.get_collection("validation_nodes")
#Add stuff to the collection 
tf.add_to_collection("validation_nodes",input_img)
tf.add_to_collection("validation_nodes", predict_lbl)

saver = tf.train.Saver()



#%-----------------Loss Function-------------------------------------------
loss_func = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_lbl,logits=logits,name='Loss')
cost = tf.div(tf.reduce_sum(loss_func),batch_size)
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_func)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    train_err = []
    test_err = []
    cost_list = []
    epoch = 0 
    while epoch < max_training_epochs:

        #Random batchfor SGD 
        random_inds = randint(0,length_train-1,batch_size) 
        batch_x = train_x[random_inds,:]
        batch_y = train_y[random_inds]
        
        feed_dict_train = {input_img:batch_x,true_lbl:batch_y};
        train_step.run(feed_dict=feed_dict_train)
        print("Epoch number " + str(epoch))
        cost_ep = cost.eval(feed_dict_train)
        print(cost_ep)
        cost_list.append(cost_ep)
        
        #Training Error
        predictions = sess.run(predict_lbl, feed_dict = {input_img:batch_x})
        train_acc = sum(predictions==batch_y)/len(predictions)
        print("Training Accuracy is " + str(train_acc))

        #Testing Error
        predictions = sess.run(predict_lbl, feed_dict = {input_img: test_x})
        test_acc = sum(predictions==test_y)/len(predictions)
        print("Testing Accuracy is " + str(test_acc))
        if (epoch+1) % display_step == 0:
            train_err.append(1 - train_acc)
            test_err.append(1 - test_acc)
        epoch = epoch + 1
    save_path = saver.save(sess,"my_model")
    weight = sess.run(W1)
    predictions = sess.run(predict_lbl, feed_dict = {input_img: test_x})
#%% plot the training error
#%% plot the overall training accuracy
plt.plot(train_err)
plt.xlabel('per 1000 iterations')
plt.ylabel('Overall Classification error for training')
plt.savefig('overall_error_train.jpg')
plt.close()

#%% plot the overall testing error
plt.plot(test_err)
plt.xlabel('per 1000 iterations')
plt.ylabel('Overall Classification error for testing')
plt.savefig('overall_error_test.jpg')
plt.close()
figs,axs = plt.subplots(4,8,constrained_layout = True)
for i in range(32):
    image = weight[:,:,:,i:i+1]
    image = image.reshape(5,5,3)
    r = np.int32(np.divide(i,8))
    c = np.int32(np.mod(i,8))
    axs[r,c].set_title(str(i))
    axs[r,c].imshow(image)
    axs[r,c].axis('off')
figs.suptitle('Weight Filters of the First Layer')
#plt.show()
plt.savefig('weights.jpg')
plt.close()
print(confusion_matrix(test_y,predictions,normalize='true'))
sio.savemat('result_for_conf.mat', {'predictions':predictions,'true_lbls':test_y})
exit()



