import os
import numpy as np
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
import h5py
import pickle
import csv
from tqdm import tqdm

path = '/share/jproject/fg538/r-006-gpu-4/data'
data = []

with h5py.File('{}/train-data.h5'.format(path), 'r') as f:
    for i in tqdm(range(9)):
        arr = f['train-images-{}'.format(i)][:]
        for a in arr:
            data.append(a)
            
labels = []
with open('{}/train-labels.pkl'.format(path), 'rb') as f:
    labels = pickle.load(f)    
    
classes = []
for folder in os.listdir('{}/train/'.format(path)):
    classes.append(folder)
    
classes_map = {classes[i]:i for i in range((len(classes)))}
labels_100 = [classes_map[label] for label in labels]

x_train, x_test, y_train, y_test = train_test_split(data, labels_100, test_size=.2, random_state=0)
x_train = np.array(x_train)

train_x = np.zeros((1, 227,227,3)).astype(np.float32)
train_y = np.zeros((1, 100))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]

net_data = np.load("/share/jproject/fg538/r-006-gpu-4/alexnet.npy").item()

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups =  tf.split(input, group, 3)   #tf.split(3, group, input)
        kernel_groups = tf.split(kernel, group, 3)  #tf.split(3, group, kernel) 
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)          #tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])
    
x = tf.placeholder(tf.float32, (None,) + xdim)
y_ = tf.placeholder(tf.float32, [None, 100])


#conv1
#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
conv1W = tf.Variable(net_data["conv1"][0])
conv1b = tf.Variable(net_data["conv1"][1])
conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
conv1 = tf.nn.relu(conv1_in)

#lrn1
#lrn(2, 2e-05, 0.75, name='norm1')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn1 = tf.nn.local_response_normalization(conv1,
                                          depth_radius=radius,
                                          alpha=alpha,
                                          beta=beta,
                                          bias=bias)

#maxpool1
#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


#conv2
#conv(5, 5, 256, 1, 1, group=2, name='conv2')
k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
conv2W = tf.Variable(net_data["conv2"][0])
conv2b = tf.Variable(net_data["conv2"][1])
conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv2 = tf.nn.relu(conv2_in)


#lrn2
#lrn(2, 2e-05, 0.75, name='norm2')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn2 = tf.nn.local_response_normalization(conv2,
                                          depth_radius=radius,
                                          alpha=alpha,
                                          beta=beta,
                                          bias=bias)

#maxpool2
#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#conv3
#conv(3, 3, 384, 1, 1, name='conv3')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
conv3W = tf.Variable(net_data["conv3"][0])
conv3b = tf.Variable(net_data["conv3"][1])
conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv3 = tf.nn.relu(conv3_in)

#conv4
#conv(3, 3, 384, 1, 1, group=2, name='conv4')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
conv4W = tf.Variable(net_data["conv4"][0])
conv4b = tf.Variable(net_data["conv4"][1])
conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv4 = tf.nn.relu(conv4_in)


#conv5
#conv(3, 3, 256, 1, 1, group=2, name='conv5')
k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
conv5W = tf.Variable(net_data["conv5"][0])
conv5b = tf.Variable(net_data["conv5"][1])
conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv5 = tf.nn.relu(conv5_in)

#maxpool5
#max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#fc6
#fc(4096, name='fc6')
fc6W = tf.Variable(net_data["fc6"][0])
fc6b = tf.Variable(net_data["fc6"][1])
fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

#fc7
#fc(4096, name='fc7')
fc7W = tf.Variable(net_data["fc7"][0])
fc7b = tf.Variable(net_data["fc7"][1])
fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

'''
#fc8
#fc(1000, relu=False, name='fc8')
fc8W = tf.Variable(tf.random_normal([4096, 100], stddev=tf.sqrt(2/(4096+100)), mean=0))
fc8b = tf.Variable(tf.zeros([100]))
fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

'''
#fc8
#fc(1000, relu=True, name='fc8')
fc8W = tf.Variable(net_data["fc8"][0])
fc8b = tf.Variable(net_data["fc8"][1])
fc8 = tf.nn.relu_layer(fc7, fc8W, fc8b)

#fc9
#fc(1000, relu=True, name='fc9')
fc9W = tf.Variable(tf.random_normal([1000, 1000], stddev=tf.sqrt(2/(1000+1000)), mean=0))
fc9b = tf.Variable(tf.zeros([1000]))
fc9 = tf.nn.relu_layer(fc8, fc9W, fc9b)

#fc10
#fc(100, relu=False, name='fc10')
fc10W = tf.Variable(tf.random_normal([1000, 100], stddev=tf.sqrt(2/(1000+100)), mean=0))
fc10b = tf.Variable(tf.zeros([100]))
fc10 = tf.nn.xw_plus_b(fc9, fc10W, fc10b)


#prob
#softmax(name='prob'))
prob = tf.nn.softmax(fc10)


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=fc10))
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)


mini_batch_size = 128
indices = np.arange(len(x_train))
model_path = "/share/jproject/fg538/r-006-gpu-4/models/alexnet/alexnet2.ckpt"
saver = tf.train.Saver()

epochs = 0

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    y_train_one_hot = sess.run(tf.one_hot(y_train, 100))
    y_test_one_hot = sess.run(tf.one_hot(y_test, 100))
    
    saver.restore(sess, model_path)

    with open("/share/jproject/fg538/r-006-gpu-4/models/alexnet/losses.csv", 'a') as f:
        writer = csv.writer(f)
        
        for i in range(0):
            start_time = time.time()
            
            np.random.shuffle(indices)
            
            avg_cost = 0
            temp_count = 0
            
            for j in range(0, len(indices), mini_batch_size): 
                if temp_count%100==0:
                    print('batch:', temp_count+1)
                temp_count += 1
                
                batch_indices = indices[j:j+64]
                batch_xs = x_train[batch_indices]
                batch_ys = y_train_one_hot[batch_indices]

                _, cost = sess.run([train_step, cross_entropy], feed_dict = {x : batch_xs, y_ : batch_ys})
                
                avg_cost += cost
                
            avg_cost /= temp_count

            print("Epoch:", '%02d'%(i+1), 
                  "\t train loss={:.9f}".format(avg_cost),
                  "\t time:{}\n".format(time.time()-start_time))

            #saver.save(sess, model_path)

            writer.writerow((i, avg_cost))
            
        correct_prediction = tf.equal(tf.argmax(prob, 1), tf.argmax(y_, 1))
        accuracy = 100*tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        top_5 = 100*tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=prob, targets=tf.argmax(y_, 1), k=5),
                                           tf.float32))
        
        top1 = 0
        top5 = 0
        test_count = 0
        print()
        for k in range(0, len(x_test), 1000):
            print('batch:', test_count+1)
            test_count += 1
            top1 += len(x_test[k:k+1000])*sess.run(accuracy, feed_dict={x: x_test[k:k+1000], 
                                                                        y_: y_test_one_hot[k:k+1000]})
            top5 += len(x_test[k:k+1000])*sess.run(top_5, feed_dict={x: x_test[k:k+1000], 
                                                                     y_: y_test_one_hot[k:k+1000]})
        
        print(top1/len(x_test), top5/len(x_test))
        

