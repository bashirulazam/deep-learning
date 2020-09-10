import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.image as mpimg
import random
import pickle
from numpy.random import seed
from numpy.random import randint
seed(1)
# create the label vector 
def y_label(label):

    length = len(label)
    y = np.zeros([5,length])
    for i in range(length):
        y[int(label[i]-1),i] = 1
    return y

    
def Error_num(y2,test_label):
    out = np.zeros(5)
    for i in range(len(test_label)):
        if test_label[i] != y2[i]:
            out[test_label[i]] += 1
    score = np.zeros(5)
    for i in range(5):
        number = len(test_label) - np.count_nonzero(test_label-i)
        score[i] = out[i] / number

    return score

def create_data_label(datadir,label_file):
    file_list = os.listdir(datadir)
    file_list.sort()
    label = np.loadtxt(label_file)
    label = np.float32(y_label(label))
    length_data = len(file_list)
    data = np.ones([length_data,785])

    for i in range(length_data):
        im = mpimg.imread(datadir + '/' + file_list[i])
        im = np.reshape(im,[1,784])
        data[i:i+1,0:784] = im / 255

    data = np.float32(data)
    return data,label

#loading data and label 
train_datadir = 'data_prog2/train_data'
train_label_file = 'data_prog2/labels/train_label.txt'
test_datadir = 'data_prog2/test_data'
test_label_file = 'data_prog2/labels/test_label.txt'
train_data, train_label = create_data_label(train_datadir,train_label_file)
test_data,test_label = create_data_label(test_datadir,test_label_file)
length_train = len(train_data)

#Hyperparameter Setting 
lamb = 0.001
batch_size = 80
learning_rate = 0.15
training_epochs = 2000
display_step = 20

# construct models
x = tf.placeholder('float32',[785,None])
y = tf.placeholder('float32',[5,None])
theta = tf.Variable(tf.zeros([785,5],dtype='float32') + 0.001)
x_next = tf.matmul(theta,x,transpose_a=True)

#%% gradient calcuation for Theta
sig = tf.exp(tf.matmul(theta,x,transpose_a=True))
grad_regression = tf.multiply(theta,2*lamb)
grad_softmax = tf.divide(sig,tf.reduce_sum(sig,0))
grad_LCL = -tf.matmul(x,tf.subtract(y,grad_softmax),transpose_b=True)
grad_LCL_regression  = tf.add(grad_LCL,grad_regression)
grad = tf.divide(grad_LCL_regression,batch_size)


print(grad.shape)

#update_theta
theta_update = tf.assign(theta,tf.subtract(theta,learning_rate*grad))

#compare between estimated result and true result
y2 = tf.argmax(sig,0)
y3 = tf.argmax(y,0)
score = tf.reduce_mean(tf.cast(tf.equal(y2,y3),'float32'))


#Tensorflow optimization Starts
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    train_error = []
    test_error = [] 
    for epoch in range(training_epochs):
        random_inds = randint(0,length_train-1,batch_size) # Random index for SGD            
        batch_x = train_data[random_inds,:]
        batch_x = batch_x.transpose()
        batch_y = train_label[:,random_inds]
        test = test_data.transpose()
        train = train_data.transpose()
        sess.run(theta_update, feed_dict={x:batch_x,y:batch_y})
        y2_1 = sess.run(y2,feed_dict={x:train,y:train_label})
        train_error.append(Error_num(y2_1,np.argmax(train_label,0)))
        y2_2 = sess.run(y2,feed_dict={x:test,y:test_label})
        test_error.append(Error_num(y2_2,np.argmax(test_label,0)))

        if (epoch+1) % display_step == 0:
            score1 = sess.run(score,feed_dict={x:test,y:test_label})
            if score1 > 0.93:
                learning_rate = 0.9*learning_rate
            print('epoch:%d' % epoch, 'score:%f' % score1)
    Y = sess.run(y2,feed_dict={x:test,y:test_label})
    weight = sess.run(theta)

    filehandler = open('multiclass_parameters.txt','wb')
    pickle.dump(weight,filehandler)
    filehandler.close()

#Plotting Weights
for i in range(5):
    image = weight[0:784,i:i+1]
    image= image.reshape(28,28)
    plt.subplot(1,5,i+1)
    plt.imshow(image)
plt.colorbar()
#plt.show()
plt.savefig('weights.jpg')
plt.close()
#%% plot the training error
train_error = np.array(train_error)
for i in range(5):
    plt.plot(train_error[:,i])
    plt.xlabel('iteration')
    plt.ylabel('training error for digit ' + str(i))
    #plt.show()
    plt.savefig('training_error_digit_' + str(i) +'.jpg')
    plt.close()
#%% plot the testing error
test_error = np.array(test_error)
for i in range(5):
    plt.plot(test_error[:,i])
    plt.xlabel('iteration')
    plt.ylabel('testing error for digit ' + str(i))
    #plt.show()
    plt.savefig('testing_error_digit_' + str(i) +'.jpg')
    plt.close()

