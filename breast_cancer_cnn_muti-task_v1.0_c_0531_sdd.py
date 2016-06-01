"""A very simple CNN classifier, modified to display data in TensorBoard."""
## Author: Sun Dongdong
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy
np = numpy
import random
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


from sklearn.cross_validation import KFold


# create weight variables, 

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

length = 25600
length_root = 160


f_class = 'os_year_01_label_1097.txt' # opens the csv file
f_data = 'breast_cancer_CNV_f.csv'
variable_len = 0;
try:
    d_class = np.loadtxt(f_class, dtype=float)
    d_matrix = np.loadtxt(f_data, delimiter=',', dtype=float)

    
    #add zeros
    d_matrix_fill = numpy.zeros([d_matrix.shape[0], length], numpy.float)
    d_matrix_fill[:d_matrix.shape[0], 0:24775] = d_matrix[:,0:24775]
    #d_matrix_fill[:d_matrix.shape[0], 20:125] = d_matrix[:,21:126]
    #d_matrix_fill[:d_matrix.shape[0], 881:899] = d_matrix[:,884:902]
    d_matrix = d_matrix_fill
    variable_len = len(d_matrix[0])

   

finally:
    pass

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')
flags.DEFINE_integer('max_steps', 10000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')


def create_variables():
    x_input = tf.placeholder(tf.float32, [None, variable_len], name='x-input')

    x = x_input

    x_image = tf.reshape(x, [-1,length_root,length_root,1])
    with tf.name_scope('cnn1'):
        W_conv1 = weight_variable([5, 5, 1, 1024])
        b_conv1 = bias_variable([1024])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)
    with tf.name_scope('cnn2'):
        W_conv2 = weight_variable([5, 5, 1024, 2048])
        b_conv2 = bias_variable([2048])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)
        

    with tf.name_scope('cnn3'):
        W_conv3 = weight_variable([3, 3, 2048, 512])
        b_conv3 = bias_variable([512])

        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = max_pool_2x2(h_conv3)

    with tf.name_scope('dcl'):
        W_fc1 = weight_variable([51200, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool3, [-1, 51200])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        
        
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Use a name scope to organize nodes in the graph visualizer
    with tf.name_scope('Wx_b'):
    #x = tf.matmul(x, W1)
        W_fc2 = weight_variable([1024, 2])
        b_fc2 = bias_variable([2])

        y = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        #y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    summary_weights = tf.histogram_summary('weights', W_fc2)
    summary_biases = tf.histogram_summary('biases', b_fc2)
   
    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 2], name='y-input')
    # More name scopes will clean up the graph representation
    with tf.name_scope('xent'):
        #loss = tf.reduce_sum(tf.abs(y_-y))
        cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
        summary_crossentropy = tf.scalar_summary('cross entropy', cross_entropy)
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(
        FLAGS.learning_rate).minimize(cross_entropy)
    
    

    with tf.name_scope('test'):
         
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        summary_accuracy = tf.scalar_summary('accuracy', accuracy)

    # Merge all the summaries and write them out to /tmp/mnist_logs
    merged = tf.merge_summary([summary_weights, summary_biases,
                               summary_crossentropy, summary_accuracy])
    return x_input, y_, keep_prob, train_step, merged, accuracy, y

def main(_):
    # Train the model, and feed in test data and record summaries every 10 steps
    cls = [];
    a = 0;
    b = 0;
    for row in d_class:
        cls.append([row == 1 and 1, row==0 and 1])
        if row == 0:
            a += 1
        b += 1

  
    cls = numpy.array(cls).astype(float)

    ## the following code implemets a 10-fold cross-validation tensorflow
    kf = KFold(d_matrix.shape[0], n_folds=10)
    class_predict = numpy.zeros([d_matrix.shape[0]])
    i = 0
    for train_indc, test_indc in kf:
        i += 1
        print('K fold: %s' % (i))
        class_predict[test_indc] = train_steps(d_matrix[train_indc], 
        cls[train_indc], 
        #d_class[train_indc],
        d_matrix[test_indc], 
        cls[test_indc],
        )

    class_origin = numpy.zeros([cls.shape[0]])
    for i in range(cls.shape[0]):
        class_origin[i] = cls[i,0]
    #print(class_origin)
    #print(class_predict)
    
    fpr, tpr, _ = roc_curve(class_origin, class_predict)
    auc_val = auc(fpr, tpr)
  
    print('Overall auc: %0.4f' % (auc_val))

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.4f)' % auc_val )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('plot1.png', format='png')
    ## save score
    save_score(class_origin,class_predict)
    
def save_score(class_origin,class_predict):
    fiw1 = open('class_origin.txt','w')
    fiw2 = open('class_predict.txt','w')
    for i in class_origin:
        fiw1.write(str(i)+'\n')
    fiw1.close()
    for i in class_predict:
        fiw2.write(str(i)+'\n')
    fiw2.close()

def train_steps(
  train_data, 
  train_class, 
  test_data, 
  test_class, 
  ):

    x_input, y_, keep_prob, train_step, merged, accuracy, y = create_variables()

    sess = tf.InteractiveSession()

    tf.initialize_all_variables().run()
    train_indc = list(range(train_data.shape[0]))
    batch_size = int((train_data.shape[0])/1000)
  
    


    for i in range(FLAGS.max_steps):
        if i % 9== 0:  # Record summary data and the accuracy

            feed = {x_input: test_data, y_:test_class, keep_prob: 1}
            #print(len(test_data))
            
            result = sess.run([merged, accuracy, y], feed_dict=feed)
           
            #summary_str = result[0]
            acc = result[1]
            #writer.add_summary(summary_str, i)
            y_predict = numpy.zeros([result[2].shape[0]])
            y_origin = numpy.zeros([test_class.shape[0]])
            for n in range(result[2].shape[0]):
                y_predict[n] = result[2][n,0]
                #print(y_predict[n])
                y_origin[n] = test_class[n,0]
                #print(y_origin[n])

            fpr, tpr, _ = roc_curve(y_origin, y_predict)
            auc_val = auc(fpr, tpr)
      
            print('Accuracy at step %s: acc: %0.2f auc: %0.4f' % (i, acc, auc_val))

            if i == FLAGS.max_steps-1:
                return y_predict


            random.shuffle(train_indc)
  
        else:
            data = train_data[train_indc[i%9*batch_size:(i%9+1)*batch_size]]
            cls = train_class[train_indc[i%9*batch_size:(i%9+1)*batch_size]]
            #for j in range(len(cls)):
            feed = {x_input: data, y_:cls, keep_prob:1}
           
            #print(feed)
            sess.run(train_step, feed_dict=feed)

    sess.close()

if __name__ == '__main__':
    tf.app.run()
