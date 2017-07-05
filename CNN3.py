# -*- coding: utf-8 -*-
"""
Created on Mon May 29 21:34:31 2017

 - Storing and restoring the variables
 
@author: jpatel
"""

#import section
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import math
import scipy.io as sio


# Convolutional Layer parameters
filter_size = 3          # all Convolution filters are 3 x 3 pixels.

num_filters1 = 32         

num_filters2 = 32         # There are 36 of these filters.

num_filters3 = 16
num_filters4 = 16

num_filters5 = 8
num_filters6 = 8

num_filters7 = 8
num_filters8 = 8


# Fully-connected layer parameters
fc_size1 = 24  
fc_size2 = 24
fc_size3 = 16
#for weights
w = 1
 
WEIGHT_DECAY_FACTOR = 0.005

# Image dimensions
img_size = 96

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 3

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size * num_channels

global img_shape

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size,num_channels)

# Number of classes, one class for each of 10 digits.
num_classes = 2


# weight and Bises
def new_weights(name,shape):
    return tf.get_variable(name,shape=shape,initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                                                       mode='FAN_IN',
                                                                                                       uniform=False,
                                                                                                       seed=None,
                                                                                                       dtype=tf.float32))
        
def new_biases(length):
    return tf.Variable(tf.constant(0.01, shape=[length]))

def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.
    
    global w               
    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    
    #since weights needs name we will generate the weights name with random global 
    name = 'weight%d'%(w)
    w += 1
    
    # Create new weights aka. filters with the given shape.
    weights = new_weights(name = name,shape=shape)
    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases
    # Use pooling to down-sample the image resolution?
    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
    # Rectified Linear Unit (ReLU).
    layer = tf.nn.relu(layer)

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights
    
def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()
    
    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features    

def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?
    global w
    #since weights needs name we will generate the weights name with random global 
    name = 'weight%d'%(w)
    w += 1 
    
    # Create new weights and biases.
    weights = new_weights(name = name,shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer,weights 
    
#PLACEHOLDERS

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')

x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

y_true_cls = tf.argmax(y_true, dimension=1)

#Convolutional layers
#Convset1
layer_conv1, weights_conv1 = new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size,
                   num_filters=num_filters1,
                   use_pooling=False)
#convset2
layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size,
                   num_filters=num_filters2,
                   use_pooling=True)

#Convset3
layer_conv3, weights_conv3 = new_conv_layer(input=layer_conv2,
                   num_input_channels=num_filters2,
                   filter_size=filter_size,
                   num_filters=num_filters3,
                   use_pooling=False)
layer_conv4, weights_conv4 = new_conv_layer(input=layer_conv3,
                   num_input_channels=num_filters3,
                   filter_size=filter_size,
                   num_filters=num_filters4,
                   use_pooling=True)
                                      
                   
#Convset4
layer_conv5, weights_conv5 = new_conv_layer(input=layer_conv4,
                   num_input_channels=num_filters4,
                   filter_size=filter_size,
                   num_filters=num_filters5,
                   use_pooling=False)
layer_conv6, weights_conv6 = new_conv_layer(input=layer_conv5,
                   num_input_channels=num_filters5,
                   filter_size=filter_size,
                   num_filters=num_filters6,
                   use_pooling=True)
                                      
#Convset5
layer_conv7, weights_conv7 = new_conv_layer(input=layer_conv6,
                   num_input_channels=num_filters6,
                   filter_size=filter_size,
                   num_filters=num_filters7,
                   use_pooling=False)
layer_conv8, weights_conv8 = new_conv_layer(input=layer_conv7,
                   num_input_channels=num_filters7,
                   filter_size=filter_size,
                   num_filters=num_filters8,
                   use_pooling=True)
                   
                   
layer_flat, num_features = flatten_layer(layer_conv8)
                   
layer_fc1,weights_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size1,
                         use_relu=True)
keep_prob = tf.placeholder(dtype=tf.float32)
layer_fc1_drop = tf.nn.dropout(layer_fc1,keep_prob)


layer_fc2,weights_fc2  = new_fc_layer(input=layer_fc1_drop,
                         num_inputs=fc_size1,
                         num_outputs=fc_size2,
                         use_relu=True)                                      
layer_fc2_drop = tf.nn.dropout(layer_fc2,keep_prob)

layer_fc3,weights_fc3  = new_fc_layer(input=layer_fc2_drop,
                         num_inputs=fc_size2,
                         num_outputs=fc_size3,
                         use_relu=True)                                      
layer_fc3_drop = tf.nn.dropout(layer_fc3,keep_prob)


layer_fc4,weights_fc4 = new_fc_layer(input=layer_fc3_drop,
                         num_inputs=fc_size3,
                         num_outputs=num_classes,
                         use_relu=False)                                      

w = 1;

#Prediction
y_pred = tf.nn.softmax(layer_fc4)

y_pred_cls = tf.argmax(y_pred, dimension=1)


#data loss / Cost of data
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc4,
                                                        labels=y_true)
                                                        
#Calculate the regularization loss

#l1 regularizer / change it to l2 to get the l2 loss
l2_regularizer = tf.contrib.layers.l2_regularizer(scale=WEIGHT_DECAY_FACTOR, scope=None)
weights = tf.trainable_variables() # all vars of your graph
regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, weights)
                                                        

#total cost
cost = tf.reduce_mean(cross_entropy + regularization_penalty)


#learning rate 
lr = tf.placeholder(tf.float32)

#Optimization
optimizer = tf.train.MomentumOptimizer(learning_rate=lr,momentum=0.9).minimize(cost)

#Performance measure
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
                                                        
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#####
#TENSORFLOW SESSION

session = tf.Session()
#session.run(tf.global_variables_initializer())
session.run(tf.initialize_all_variables())

### SAVE PARAMETERS
saver = tf.train.Saver()
save_dir = '/home/jpatel/Results/Weights/' #directory name

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#Path ofthe directory where weights will be saved
save_path = os.path.join(save_dir, 'best_validation')


train_batch_size = 192

# Counter for total number of iterations performed so far.
total_epoch = 0

# Best validation accuracy seen so far.
best_validation_accuracy = 0.0
trainingAcc = []
trainingCost = []
#accuracy for validation set
validAcc = []

#initial epoch
e = 0

####
def optimize(epoch):
    # Ensure we update the global variable rather than a local copy.
    global total_epoch
    global validAcc
    global trainingCost
    global trainingAcc
    global e
    global best_validation_accuracy
    
    # Start-time used for printing time-usage below.
    start_time = time.time()

    while e in range(total_epoch,
                   total_epoch + epoch):
         
        #initialize the temporary storage for EVERY EPOCH 
        costMBL = []
        accMBL = []   

                            
        #For every epoch we will call all the baches of the data
        for batch in range(1,5):
            
            cw_dataXs,cw_labelYs = get_data(batch)

            num_test = cw_dataXs.shape[0]
            
            i=0
            while i < num_test:
                # The ending index for the next batch is denoted j.
                j = min(i + train_batch_size, num_test)

                # Get the images from the train-batch between index i and j.
                images = cw_dataXs[i:j, :]
                # Get the associated labels.
                labels = cw_labelYs[i:j, :]
                #learning rate 
                l_rate = get_rate(e)

                # Create a feed-dict with these images and labels.
                feed_dict_train = {x: images,
                              y_true: labels,keep_prob:0.3,lr:l_rate}  
                
                _,costMB,accMB = session.run([optimizer,cost,accuracy], feed_dict=feed_dict_train)
                
                #save the results
                    
                costMBL.append(costMB)
                accMBL.append(accMB)
                
                i=j
        
        trainingCost.append( float( sum(costMBL) / len(costMBL) ) )
        trainingAcc.append( float( sum(accMBL) / len(accMBL) ) )
        
        
        
        # Print status every 100 iterations.
        if e % 1 == 0:
            # Calculate the accuracy on the training-set.
            acc = print_test_accuracy(batch=5)
            
            #save the result
            validAcc.append(acc)
            
            # Message for printing.
            msg = "Optimization Epoch: {0:>6}, Validation Accuracy: {1:>6.1%}"
            # Print it.
            print(msg.format(e + 1, acc))
        
           
        #update the epoch
        e += 1
        
        #update the learning rate
        l_rate = get_rate(e)  
        
        msg = 'Learning Rate: {0}'
        #Print the learning rate 
        print(msg.format(l_rate))
        
    # Update the total number of iterations performed.
    total_epoch += epoch

          
    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

    saver.save(sess=session, save_path=save_path)
    print("Model stored in file: %s" % save_path)

######

#Learning rate
def get_rate(ep): #pass the current number of epoch
    
    if ep<=60:
        return 0.01/(2**(ep/5))
    else:
        return 1e-6        
        
# Split the test-set into smaller batches of this size.
test_batch_size = 256

def print_test_accuracy(batch=6):
    
    #get the test data
    cw_dataXs,cw_labelYs = get_data(batch)
    
    # Number of images in the test-set.
    num_test = cw_dataXs.shape[0]
    
    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = cw_dataXs[i:j, :]

        # Get the associated labels.
        labels = cw_labelYs[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict_test = {x: images,
                           y_true: labels,keep_prob:1.0} 

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict_test)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = cw_labelYs[:,1]

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))
    
    return acc

def plot_examples(n = 0,batch=12,cls=False):
    
    #get the test data
    cw_dataXs,cw_labelYs = get_data(batch)
    
    # Number of images in the test-set.
    num_test = cw_dataXs.shape[0]
    
    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = cw_dataXs[i:j, :]

        # Get the associated labels.
        labels = cw_labelYs[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict_test = {x: images,
                           y_true: labels,keep_prob:1.0} 

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict_test)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = cw_labelYs[:,1]

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)
    
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == cls)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = cw_dataXs[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = cls_true[incorrect]
    
    # Plot the first 9 images.
    plot_images(images=images[n:n+32],
                cls_true=cls_true[n:n+32],
                cls_pred=cls_pred[n:n+32])



def plot_images(images, cls_true, cls_pred=None,img_shape=img_shape):
    
    # Create figure with 3x3 sub-plots.
    fig,axes = plt.subplots(4,8)
    fig.subplots_adjust(hspace=0.25,wspace=0.2)
    
    #preprocess the images to show them correctly    
    M = sio.loadmat('/home/jpatel/Python/DatasetMAT_96/meanVal.mat')
    meanVal = M['meanVal']
    
    S = sio.loadmat('/home/jpatel/Python/DatasetMAT_96/stdVal.mat')
    stdVal = S['stdVal']
    
    #Reverse operations of an feature scaling
    images = (images * stdVal)+meanVal;
    
    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='gray')
        
        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def plot_conv_weights(weights, input_channel=0):
    # Assume weights are TensorFlow ops for 4-dim variables
    # e.g. weights_conv1 or weights_conv2.
    
    # Retrieve the values of the weight-variables from TensorFlow.
    # A feed-dict is not necessary because nothing is calculated.
    w = session.run(weights)
    
    # Print mean and standard deviation.
    print("Mean: {0:.5f}, Stdev: {1:.5f}".format(w.mean(), w.std()))
    
    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = int(math.ceil(math.sqrt(num_filters)))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i<num_filters:
            # Get the weights for the i'th filter of the input channel.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='gist_gray')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
    
    return w

def plot_fc_weights(weights):
    # Assume weights are TensorFlow ops for 4-dim variables
    # e.g. weights_conv1 or weights_conv2.
    
    # Retrieve the values of the weight-variables from TensorFlow.
    # A feed-dict is not necessary because nothing is calculated.
    w = session.run(weights)
    
    # Print mean and standard deviation.
    print("Mean: {0:.5f}, Stdev: {1:.5f}".format(w.mean(), w.std()))
    
    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    return w

 
def plot_conv_layer(layer, image):
    # Assume layer is a TensorFlow op that outputs a 4-dim tensor
    # which is the output of a convolutional layer,
    # e.g. layer_conv1 or layer_conv2.

    # Create a feed-dict containing just one image.
    # Note that we don't need to feed y_true because it is
    # not used in this calculation.
    feed_dict = {x: [image]}

    # Calculate and retrieve the output values of the layer
    # when inputting that image.
    values = session.run(layer, feed_dict=feed_dict)

    # Number of filters used in the conv. layer.
    num_filters = values.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = int(math.ceil(math.sqrt(num_filters)))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot the output images of all the filters.
    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters.
        if i<num_filters:
            # Get the output image of using the i'th filter.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, interpolation='nearest', cmap='gray')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def get_data(n):

    M = sio.loadmat('/home/jpatel/Python/DatasetMAT_96/meanVal.mat')
    meanVal = M['meanVal']
    
    S = sio.loadmat('/home/jpatel/Python/DatasetMAT_96/stdVal.mat')
    stdVal = S['stdVal']
    
    #n = batch number 
    
    #load the .mat file
    filename='/home/jpatel/Python/DatasetMAT_96/batch%d.mat'%(n)
    I = sio.loadmat(filename)
    
    key1 = 'data%d'%(n)    
    a = I[key1]
    key2 = 'label%d'%(n)
    b = I[key2]
    
    c = np.c_[a.reshape(len(a), -1), b.reshape(len(b), -1)]    
    
    a = c[:, :a.size//len(a)].reshape(a.shape)
    b = c[:, a.size//len(a):].reshape(b.shape)
    
    return (np.array(a,dtype=np.float32)/255-np.array(meanVal)) / np.array(stdVal),np.array(b,dtype=np.int)
    

def plotloss(trainingCost):
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Cost')
    plt.title('Cost Function vs Epoch')
    plt.grid(True)
    plt.plot(trainingCost, label='Training Loss')
    plt.show()

def plotaccuracy(accuracyTrain,accuracyValid):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epoch')
    plt.grid(True)
    plt.plot(accuracyTrain, label='Training Accuracy')
    plt.plot(accuracyValid,label='Validation Accuracy')
    plt.show()

