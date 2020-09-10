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
    y = np.zeros([10,length])
    for i in range(length):
        y[int(label[i]),i] = 1
    return y

    
def Error_num(y2,test_label):
    out = np.zeros(10)
    for i in range(len(test_label)):
        if test_label[i] != y2[i]:
            out[test_label[i]] += 1
    score = np.zeros(10)
    for i in range(10):
        number = len(test_label) - np.count_nonzero(test_label-i)
        score[i] = out[i] / number

    return score

def create_data_label(datadir,label_file):
    file_list = os.listdir(datadir)
    file_list.sort()
    label = np.loadtxt(label_file)
    label = np.float32(y_label(label))
    length_data = len(file_list)
    data = np.ones([length_data,784])

    for i in range(length_data):
        im = mpimg.imread(datadir + '/' + file_list[i])
        im = np.reshape(im,[1,784])
        data[i:i+1,0:784] = im / 255

    data = np.float32(data)
    return data,label

#loading data and label 
train_datadir = 'data_prog2Spring18/train_data'
train_label_file = 'data_prog2Spring18/labels/train_label.txt'
test_datadir = 'data_prog2Spring18/test_data'
test_label_file = 'data_prog2Spring18/labels/test_label.txt'
train_data, train_label = create_data_label(train_datadir,train_label_file)
test_data,test_label = create_data_label(test_datadir,test_label_file)
length_train = len(train_data)
length_test = len(test_data)

#Hyperparameter Setting 
batch_size = 50
learning_rate = 0.15
max_training_epochs = 3000
display_step = 100
subset_len = 5000

#%----------------Forward Propagation----------------------
sigma_init = 0.1
X = tf.placeholder('float32',[784,1])
Y = tf.placeholder('float32',[10,1])
#X = tf.Variable(tf.random_normal([784,1],mean=0,stddev=1,dtype='float32'),name='X')
#Y = tf.Variable(tf.random_normal([10,1],mean=0,stddev=1,dtype='float32'),name='Y')
W1 = tf.Variable(tf.random_normal([784,100],mean=0,stddev=sigma_init,dtype='float32'),name='W1')
W1_0 = tf.Variable(tf.zeros([100,1],dtype='float32'),name='W1_0')
W2 = tf.Variable(tf.random_normal([100,100],mean=0,stddev=sigma_init,dtype='float32'),name='W2')
W2_0 = tf.Variable(tf.zeros([100,1],dtype='float32'),name='W2_0')
W3 = tf.Variable(tf.random_normal([100,10],mean=0,stddev=sigma_init,dtype='float32'),name='W3')
W3_0 = tf.Variable(tf.zeros([10,1],dtype='float32'),name='W3_0')
#%Forward Pass 
z1 = tf.add(tf.matmul(W1,X,transpose_a=True),W1_0,name='z1')
#H1 = my_Relu(z1,'H1')
H1 = tf.math.maximum(z1,tf.zeros_like(z1),name='H1')
z2 = tf.add(tf.matmul(W2,H1,transpose_a=True),W2_0,name='z2')
#H2 = my_Relu(z2,'H2')
H2 = tf.math.maximum(z2,tf.zeros_like(z2),name='H2')
z3 = tf.add(tf.matmul(W3,H2,transpose_a=True),W3_0,name='z3')
#Y_pred = my_softmax(z3)
Y_pred = tf.math.divide(tf.math.exp(z3),tf.reduce_sum(tf.math.exp(z3)),name='Y_pred')

#%-----------------Loss Function-------------------------------------------
loss = -tf.reduce_sum(tf.multiply(Y,tf.log(Y_pred)))

#%-----------------Backward Propagation for Output Layer-------------------

#delY formation
del_Y = -tf.divide(Y,Y_pred)

#del_W3 Calculation
#delz3_delW3 formation
delz3_delW3_elem = tf.concat([H2,tf.zeros_like(H2)],1)
for k in range(8):
  delz3_delW3_elem = tf.concat([delz3_delW3_elem,tf.zeros_like(H2)],1)
delz3_delW3_list = []
for k in range(10):
  delz3_delW3_list.append(tf.roll(delz3_delW3_elem,k,1))
delz3_delW3 = tf.stack(delz3_delW3_list,2)

#dely_delz3 formation
temp = -tf.matmul(Y_pred,Y_pred,transpose_b=True)
temp_diag = tf.reshape(tf.diag(Y_pred),[Y_pred.shape[0],Y_pred.shape[0]])
dely_delz3 = tf.add(temp,temp_diag)

tempz3 = tf.matmul(dely_delz3,del_Y)
del_W3 = tf.reshape(tf.matmul(delz3_delW3,tempz3),W3.shape)

#del_H2 Calculation
del_H2 = tf.reshape(tf.matmul(W3,tempz3),H2.shape)

#del_W3_0 Calculation
del_W3_0 = tempz3

#%-----------Backward Propagation for Second Hidden Layer---------------

#del_W2 Calculation

#delz2_delW2 formation
delz2_delW2_elem = tf.concat([H1,tf.zeros_like(H1)],1)
for k in range(98):
  delz2_delW2_elem = tf.concat([delz2_delW2_elem,tf.zeros_like(H1)],1)
delz2_delW2_list = []
for k in range(100):
  delz2_delW2_list.append(tf.roll(delz2_delW2_elem,k,1))
delz2_delW2 = tf.stack(delz2_delW2_list,2)

#del_phi_z2_del_z2 formation
temp = tf.sign(tf.maximum(z2,tf.zeros_like(z2)))
del_phi_z2_del_z2 = tf.reshape(tf.diag(temp),[z2.shape[0],z2.shape[0]])


tempz2 = tf.matmul(del_phi_z2_del_z2,del_H2)
del_W2 = tf.reshape(tf.matmul(delz2_delW2,tempz2),W2.shape)
#del_H1 Calculation
del_H1 = tf.reshape(tf.matmul(W2,tempz2),H1.shape)

#del_W2_0 Calculation
del_W2_0 = tempz2

#%--------------Backward Propagation for First Hidden Layer-------------------

#del_W1 Calculation

#delz1_delW1 formation
delz1_delW1_elem = tf.concat([X,tf.zeros_like(X)],1)
for k in range(98):
  delz1_delW1_elem = tf.concat([delz1_delW1_elem,tf.zeros_like(X)],1)
delz1_delW1_list = []
for k in range(100):
  delz1_delW1_list.append(tf.roll(delz1_delW1_elem,k,1))
delz1_delW1 = tf.stack(delz1_delW1_list,2)

#del_phi_z1_del_z1 formation
temp = tf.sign(tf.maximum(z1,tf.zeros_like(z1)))
del_phi_z1_del_z1 = tf.reshape(tf.diag(temp),[z1.shape[0],z1.shape[0]])

tempz1 = tf.matmul(del_phi_z1_del_z1,del_H1)

del_W1 = tf.reshape(tf.matmul(delz1_delW1,tempz1),W1.shape)

#del_W1_0 Calculation
del_W1_0 = tempz1

#------------------------% Update Thetas--------------------
W1_update = tf.assign(W1,tf.subtract(W1,learning_rate*del_W1))
W1_0_update = tf.assign(W1_0,tf.subtract(W1_0,learning_rate*del_W1_0))
W2_update = tf.assign(W2,tf.subtract(W2,learning_rate*del_W2))
W2_0_update = tf.assign(W2_0,tf.subtract(W2_0,learning_rate*del_W2_0))
W3_update = tf.assign(W3,tf.subtract(W3,learning_rate*del_W3))
W3_0_update = tf.assign(W3_0,tf.subtract(W3_0,learning_rate*del_W3_0))

#Formation of Theta and Theta Update

theta = []
theta.append(W1)
theta.append(W1_0)
theta.append(W2)
theta.append(W2_0)
theta.append(W3)
theta.append(W3_0)


#compare between estimated result and true result
y2 = tf.argmax(Y_pred,0)
y3 = tf.argmax(Y,0)
score = tf.reduce_mean(tf.cast(tf.equal(y2,y3),'float32'))

error_train_mean_for_plot = []
error_test_mean_for_plot = []

#Tensorflow optimization Starts
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    train_error = []
    test_error = []
    epoch = 0 
    while epoch < max_training_epochs:

        #Random batchfor SGD 
        random_inds = randint(0,length_train-1,batch_size) 
        batch_x = train_data[random_inds,:]
        batch_x = batch_x.transpose()
        batch_y = train_label[:,random_inds]
        test = test_data.transpose()
        train = train_data.transpose()
        
        #Initialize average gradient to zero in each epoch
        del_W1_avg = np.zeros(W1.shape)
        del_W1_0_avg = np.zeros(W1_0.shape)
        del_W2_avg = np.zeros(W2.shape)
        del_W2_0_avg = np.zeros(W2_0.shape)
        del_W3_avg = np.zeros(W3.shape)
        del_W3_0_avg = np.zeros(W3_0.shape)
        

        #Computation of Gradients for random 50 samples
        for k in range(batch_size):
            xs = np.reshape(batch_x[:,k],(784,1))
            ys = np.reshape(batch_y[:,k],(10,1))           
            del_W1_avg = del_W1_avg + sess.run(del_W1,feed_dict={X:xs,Y:ys})
            del_W1_0_avg = del_W1_0_avg + sess.run(del_W1_0,feed_dict={X:xs,Y:ys})
            del_W2_avg = del_W2_avg + sess.run(del_W2,feed_dict={X:xs,Y:ys})
            del_W2_0_avg = del_W2_0_avg + sess.run(del_W2_0,feed_dict={X:xs,Y:ys})
            del_W3_avg = del_W3_avg + sess.run(del_W3,feed_dict={X:xs,Y:ys})
            del_W3_0_avg = del_W3_0_avg + sess.run(del_W3_0,feed_dict={X:xs,Y:ys})

        #Update of weight and bias parameters with average gradients
        sess.run(W1_update,feed_dict={del_W1:del_W1_avg/batch_size})
        sess.run(W1_0_update,feed_dict={del_W1_0:del_W1_0_avg/batch_size})
        sess.run(W2_update,feed_dict={del_W2:del_W2_avg/batch_size})
        sess.run(W2_0_update,feed_dict={del_W2_0:del_W2_0_avg/batch_size})
        sess.run(W3_update,feed_dict={del_W3:del_W3_avg/batch_size})
        sess.run(W3_0_update,feed_dict={del_W3_0:del_W3_0_avg/batch_size})

        print(sess.run(loss,feed_dict={X:xs,Y:ys}))
        y2_train = np.zeros([subset_len,1])
        y2_test = np.zeros([subset_len,1])
        score_train_all = np.zeros([subset_len,1])
        score_test_all = np.zeros([subset_len,1])

        #Training Error for random train data
        random_inds = randint(0,length_train-1,subset_len)
        train_subset = train_data[random_inds,:]
        train_subset = train_subset.transpose()
        train_label_subset = train_label[:,random_inds]
        if (epoch+1) % display_step == 0:
            for i in range(subset_len):
                xs_train = np.reshape(train_subset[:,i],(784,1))
                ys_train = np.reshape(train_label_subset[:,i],(10,1))
                y2_train[i] = sess.run(y2,feed_dict={X:xs_train,Y:ys_train})
                score_train_all[i] = sess.run(score,feed_dict={X:xs_train,Y:ys_train})
         
        
            #Testing Error for random test data
            random_inds = randint(0,length_test-1,subset_len)
            test_subset = test_data[random_inds,:]
            test_subset = test_subset.transpose()
            test_label_subset = test_label[:,random_inds]

            for i in range(subset_len):
                xs_test = np.reshape(test_subset[:,i],(784,1));
                ys_test = np.reshape(test_label_subset[:,i],(10,1));
                y2_test[i] = sess.run(y2,feed_dict={X:xs_test,Y:ys_test})
                score_test_all[i] = sess.run(score,feed_dict={X:xs_test,Y:ys_test})
        
            train_error.append(Error_num(y2_train,np.argmax(train_label_subset,0)))
            test_error.append(Error_num(y2_test,np.argmax(test_label_subset,0)))
                               
            score_train_mean = np.mean(score_train_all)
            score_test_mean = np.mean(score_test_all)
            error_train_mean_for_plot.append(1-score_train_mean)
            error_test_mean_for_plot.append(1-score_test_mean)
            #if score1 > 0.93:
            #    learning_rate = 0.9*learning_rate
            print('epoch:%d' % epoch, 'score for training:%f' % score_train_mean, 'score for testing:%f' % score_test_mean)

        epoch +=1
        print('epoch:%d' % epoch)

    weight = sess.run(theta)

    filehandler = open('nn_parameters.txt','wb')
    pickle.dump(weight,filehandler,protocol=2)
    filehandler.close()

#%% plot the training error
train_error = np.array(train_error)
for i in range(10):
    plt.plot(train_error[:,i])
    plt.xlabel('per 100 iterations')
    plt.ylabel('training error for digit ' + str(i))
    #plt.show()
    plt.savefig('training_error_digit_' + str(i) +'.jpg')
    plt.close()
#%% plot the testing error
test_error = np.array(test_error)
for i in range(10):
    plt.plot(test_error[:,i])
    plt.xlabel('per 100 iterations')
    plt.ylabel('testing error for digit ' + str(i))
    #plt.show()
    plt.savefig('testing_error_digit_' + str(i) +'.jpg')
    plt.close()

#%% plot the overall training error
plt.plot(error_train_mean_for_plot)
plt.xlabel('per 100 iterations')
plt.ylabel('Overall Classificatio error for training')
plt.savefig('overall_error_train.jpg')
plt.close()

#%% plot the overall testing error
plt.plot(error_test_mean_for_plot)
plt.xlabel('per 100 iterations')
plt.ylabel('Overall Classificatio error for testing')
plt.savefig('overall_error_test.jpg')
plt.close()



