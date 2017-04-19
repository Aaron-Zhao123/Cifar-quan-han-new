import cifar10
import tensorflow as tf
import numpy as np
import sys
import os
import pickle
import time
import getopt

from cifar10 import img_size, num_channels, num_classes
from tensorflow.python.framework import ops

class Usage(Exception):
    def __init__ (self,msg):
        self.msg = msg

def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
    Returns:
    Variable Tensor
    """
    dtype = tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var
def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
    Returns:
    Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var

def initialize_variables(exist, parent_dir, NUMBER_OF_CLUSTER, pretrain):
    NUM_CHANNELS = 3
    IMAGE_SIZE = 32
    NUM_CLASSES = 10
    if (exist == 1):
        if (pretrain == 1):
            with open(parent_dir + 'weights/'+'weightspt'+str(NUMBER_OF_CLUSTER)+'.pkl','rb') as f:
                weights_orgs, biases_orgs, cluster_index,centroids = pickle.load(f)
        else:
            with open(parent_dir + 'weights/'+'weights'+str(NUMBER_OF_CLUSTER)+'.pkl','rb') as f:
                weights_orgs, biases_orgs, cluster_index,centroids = pickle.load(f)

    centroids_var = {
        'cov1': tf.Variable(centroids['cov1']),
        'cov2': tf.Variable(centroids['cov2']),
        'fc1': tf.Variable(centroids['fc1']),
        'fc2': tf.Variable(centroids['fc2']),
        'fc3': tf.Variable(centroids['fc3'])
    }
    # weights_index = {
    #     'cov1': tf.constant(cluster_index['cov1'],tf.float32),
    #     'cov2': tf.constant(cluster_index['cov2'],tf.float32),
    #     'fc1': tf.constant(cluster_index['fc1'],tf.float32),
    #     'fc2': tf.constant(cluster_index['fc2'],tf.float32),
    #     'fc3': tf.constant(cluster_index['fc3'],tf.float32)
    # }

    #  reconfig weights_index
    weights_index = {}
    keys = ['cov1','cov2','fc1','fc2', 'fc3']
    for key in keys:
        s = np.shape(cluster_index[key])
        sn = s + (NUMBER_OF_CLUSTER,)
        tmp = np.zeros(sn)
        for i in range(1, NUMBER_OF_CLUSTER + 1):
            tmp[i-1] = (cluster_index[key] == i)
        weights_index[key] = tmp
    print(np.shape(weights_index['fc1']))
    sys.exit()
    weights_index = {
        'cov1': cluster_index['cov1'],
        'cov2': cluster_index['cov2'],
        'fc1': cluster_index['fc1'],
        'fc2': cluster_index['fc2'],
        'fc3': cluster_index['fc3']
    }
    biases = {
        'cov1': tf.Variable(biases_orgs['cov1']),
        'cov2': tf.Variable(biases_orgs['cov2']),
        'fc1': tf.Variable(biases_orgs['fc1']),
        'fc2': tf.Variable(biases_orgs['fc2']),
        'fc3': tf.Variable(biases_orgs['fc3'])
    }
    return (biases, centroids_var, weights_index, cluster_index, centroids)

def compute_weights(weights_index, centroids_var, Number_of_cluster):
    keys = ['cov1','cov2','fc1','fc2', 'fc3']
    weights = {}
    for key in keys:
        # weights[key] = tf.to_float(tf.equal(weights_index[key], 1)) * centroids_var[key][0]
        # print(weights_index[key])
        for i in range(1, Number_of_cluster + 1):
            if (i == 1):
                # weights[key] = tf.cast(tf.equal(weights_index[key], 1),tf.float32) * tf.cast(centroids_var[key][0], tf.float32)
                weights[key] = (weights_index[key] == 1) * tf.cast(centroids_var[key][0], tf.float32)
            # else:
            #     # weights[key] = weights[key] + tf.to_float(tf.equal(weights_index[key], i)) * tf.cast(centroids_var[key][i-1], tf.float32)
            #     weights[key] = weights[key] + (weights_index[key] == i) * tf.cast(centroids_var[key][i-1], tf.float32)
    return weights


def initialize_weights_mask(first_time_training, mask_dir):
    NUM_CHANNELS = 3
    NUM_CLASSES = 10
    first_time_training = 1 # temp solution
    if (first_time_training == 1):
        print('setting initial mask value')
        weights_mask = {
            'cov1': np.ones([5, 5, NUM_CHANNELS, 64]),
            'cov2': np.ones([5, 5, 64, 64]),
            'fc1': np.ones([6 * 6 * 64, 384]),
            'fc2': np.ones([384, 192]),
            'fc3': np.ones([192, NUM_CLASSES])
        }
        biases_mask = {
            'cov1': np.ones([64]),
            'cov2': np.ones([64]),
            'fc1': np.ones([384]),
            'fc2': np.ones([192]),
            'fc3': np.ones([NUM_CLASSES])
        }
    else:
        with open(mask_dir,'rb') as f:
            (weights_mask, biases_mask) = pickle.load(f)
    return (weights_mask, biases_mask)

def prune_info(weights, counting):
    if (counting == 0):
        (non_zeros, total) = calculate_non_zero_weights(weights['cov1'].eval())
        print('cov1 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        print('some numbers: non zeros:{}, total:{}'.format(non_zeros, total))
        # print(weights['cov1'].eval())
        (non_zeros, total) = calculate_non_zero_weights(weights['cov2'].eval())
        print('cov2 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        (non_zeros, total) = calculate_non_zero_weights(weights['fc1'].eval())
        print('fc1 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        (non_zeros, total) = calculate_non_zero_weights(weights['fc2'].eval())
        print('fc2 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        (non_zeros, total) = calculate_non_zero_weights(weights['fc3'].eval())
        print('fc3 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        print('some numbers: non zeros:{}, total:{}'.format(non_zeros, total))
    if (counting == 1):
        (non_zeros, total) = calculate_non_zero_weights(weights['fc1'].eval())
        print('take fc1 as example, {} nonzeros, in total {} weights'.format(non_zeros, total))
def plot_weights(weights,pruning_info):
        keys = ['cov1','cov2','fc1', 'fc2','fc2']
        fig, axrr = plt.subplots( 2, 2)  # create figure &  axis
        fig_pos = [(0,0), (0,1), (1,0), (1,1)]
        index = 0
        for key in keys:
            weight = weights[key].eval().flatten()
            # print (weight)
            size_weight = len(weight)
            weight = weight.reshape(-1,size_weight)[:,0:size_weight]
            x_pos, y_pos = fig_pos[index]
            #take out zeros
            weight = weight[weight != 0]
            # print (weight)
            hist,bins = np.histogram(weight, bins=100)
            width = 0.7 * (bins[1] - bins[0])
            center = (bins[:-1] + bins[1:]) / 2
            axrr[x_pos, y_pos].bar(center, hist, align = 'center', width = width)
            axrr[x_pos, y_pos].set_title(key)
            index = index + 1
        fig.savefig('fig_v3/weights'+pruning_info)
        plt.close(fig)


def cov_network(images, weights, biases, keep_prob):
    BATCH_SIZE = 128
    NUM_CLASSES = 10
    IMAGE_SIZE = 32
    NUM_CHANNELS = 3
    # cov1_weight = weights['cov1'] * weights_mask['cov1']
    # conv1
    conv = tf.nn.conv2d(images, weights['cov1'], [1, 1, 1, 1], padding='SAME')
    # conv = tf.nn.conv2d(images, cov1_weight, [1, 1, 1, 1], padding='SAME')

    pre_activation = tf.nn.bias_add(conv, biases['cov1'])
    conv1 = tf.nn.relu(pre_activation)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

    # conv2
    # cov2_weight = weights['cov2'] * weights_mask['cov2']
    conv = tf.nn.conv2d(norm1, weights['cov2'], [1, 1, 1, 1], padding='SAME')
    # conv = tf.nn.conv2d(norm1, cov2_weight, [1, 1, 1, 1], padding='SAME')
    pre_activation = tf.nn.bias_add(conv, biases['cov2'])
    conv2 = tf.nn.relu(pre_activation)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # local3
    # Move everything into depth so we can perform a single matrix multiply.
    # dim = 1
    # for d in pool2.get_shape()[1:].as_list():
    #   dim *= d
    # print(pool2.get_shape().as_list())
    # reshape = tf.reshape(pool2, [-1, IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64])
    reshape = tf.reshape(pool2, [-1, 6 * 6 * 64])
    # reshape = tf.reshape(pool2, [BATCH_SIZE, dim])
    # print(reshape)

    # fc1_weight = weights['fc1'] * weights_mask['fc1']
    local3 = tf.nn.relu(tf.matmul(reshape, weights['fc1']) + biases['fc1'])
    # dropout
    local3_drop = tf.nn.dropout(local3, keep_prob)
# # local4
    # fc2_weight = weights['fc2'] * weights_mask['fc2']
    local4 = tf.nn.relu(tf.matmul(local3_drop, weights['fc2']) + biases['fc2'])
    local4_drop = tf.nn.dropout(local4, keep_prob)
# # We don't apply softmax here because
# # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
# # and performs the softmax internally for efficiency.
    # fc3_weight = weights['fc3'] * weights_mask['fc3']
    softmax_linear = tf.add(tf.matmul(local4_drop, weights['fc3']), biases['fc3'])
    return softmax_linear

def loss(logits, labels):
  """Add L2Loss to all the trainable variables.
  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')

class format_data():
    def __init__ (self, images, labels):
        self.images = images
        self.labels = labels
        self.batch_cnt = 0
        self.data_size = len(images)
    def feed_next_batch(self, batch_size):
        start_pt = self.batch_cnt
        self.batch_cnt += batch_size
        if (self.batch_cnt <= self.data_size):
            return (self.images[start_pt:self.batch_cnt],
                    self.labels[start_pt:self.batch_cnt])
        else:
            self.batch_cnt = 0
            return (self.images[start_pt:self.data_size],
                    self.labels[start_pt:self.data_size])


def save_pkl_model(weights, biases, save_dir):
    # name = os.path.join(data_path, "cifar-10-batches-py/", filename)
    # if not os.path.exists(path):
    #     os.makedirs(path)
    keys = ['cov1','cov2','fc1','fc2','fc3']
    weights_val = {}
    biases_val = {}
    for key in keys:
        weights_val[key] = weights[key].eval()
        biases_val[key] = biases[key].eval()
    with open(save_dir, 'wb') as f:
        print('Created a pickle file')
        pickle.dump((weights_val, biases_val), f)

def pre_process_image(image, training):
    # This function takes a single image as input,
    # and a boolean whether to build the training or testing graph.

    num_channels = 3
    img_size_cropped = 24

    if training:
        # For training, add the following to the TensorFlow graph.

        # Randomly crop the input image.
        image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, num_channels])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

        # Randomly adjust hue, contrast and saturation.
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

        # Some of these functions may overflow and result in pixel
        # values beyond the [0, 1] range. It is unclear from the
        # documentation of TensorFlow 0.10.0rc0 whether this is
        # intended. A simple solution is to limit the range.

        # Limit the image pixels between [0, 1] in case of overflow.
        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)
    else:
        # For training, add the following to the TensorFlow graph.

        # Crop the input image around the centre so it is the same
        # size as images that are randomly cropped during training.
        image = tf.image.resize_image_with_crop_or_pad(image,
                                                       target_height=img_size_cropped,
                                                       target_width=img_size_cropped)

    return image

def calculate_non_zero_weights(weight):
    count = (weight != 0).sum()
    size = len(weight.flatten())
    return (count,size)

def pre_process(images, training):
    # Use TensorFlow to loop over all the input images and call
    # the function above which takes a single image as input.
    images = tf.map_fn(lambda image: pre_process_image(image, training), images)
    return images

def mask_gradients(weights, grads_and_names, weight_masks, biases, biases_mask):
    new_grads = []
    keys = ['cov1', 'cov2', 'fc1', 'fc2', 'fc3']
    for grad,var_name in grads_and_names:
        flag = 0
        index = 0
        for key in keys:
            if (weights[key]== var_name):
                mask = weight_masks[key]
                new_grads.append((tf.multiply(tf.constant(mask, dtype = tf.float32),grad),var_name))
                flag = 1
            if (weights[key]== var_name):
                mask = biases_mask[key]
                new_grads.append((tf.multiply(tf.constant(mask, dtype = tf.float32),grad),var_name))
                flag = 1
        # if flag is not set
        if (flag == 0):
            new_grads.append((grad,var_name))
    return new_grads

def ClipIfNotNone(grad):
    if grad is None:
        return grad
    return tf.clip_by_value(grad, -1, 1)

def main(argv = None):
    if (argv is None):
        argv = sys.argv
    try:
        try:
            opts = argv
            TRAIN = 1
            NUMBER_OF_CLUSTER = 8
            pretrain = 0
            for item in opts:
                print (item)
                opt = item[0]
                val = item[1]
                if (opt == '-pcov'):
                    pruning_cov = val
                if (opt == '-pfc'):
                    pruning_fc = val
                if (opt == '-t'):
                    TRAIN = val
                if (opt == '-cluster'):
                    NUMBER_OF_CLUSTER = val
                if (opt == '-pretrain'):
                    pretrain = val
                if (opt == '-parent_dir'):
                    parent_dir = val
                if (opt == '-base_model'):
                    base_model = val
            # print('pruning count is {}, {}'.format(pruning_cov, pruning_fc))
            print('pretrain is {}'.format(pretrain))
        except getopt.error, msg:
            raise Usage(msg)
        NUM_CLASSES = 10
        dropout = 0.8 # probability of keep
        BATCH_SIZE = 128
        NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
        INITIAL_LEARNING_RATE = 0.001
        LEARNING_RATE_DECAY_FACTOR = 0.1
        NUM_EPOCHS_PER_DECAY = 350.0
        MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
        DISPLAY_FREQ = 20
        TEST = 0
        TRAIN_OR_TEST = 0
        NUM_CHANNELS = 3
        DOCKER = 0
        mask_dir = parent_dir
        model_dir = parent_dir
        PREV_MODEL_EXIST = 1


        # ops.reset_default_graph()
        # cls_train returns as an integer, labels is the array
        print('pruning on cov is {}. on fc is {}'.format(pruning_cov, pruning_fc))
        (weights_mask,biases_mask)= initialize_weights_mask(0, mask_dir + 'masks/' + 'base.pkl' )
        cifar10.maybe_download_and_extract()
        class_names = cifar10.load_class_names()

        images_train, cls_train, labels_train = cifar10.load_training_data()
        images_test, cls_test, labels_test = cifar10.load_test_data()
        t_data = format_data(images_train, labels_train)
        test_data = format_data(images_test, labels_test)

        DATA_CNT = len(images_train)
        NUMBER_OF_BATCH = DATA_CNT / BATCH_SIZE

        training_data_list = []

        biases, centroids_var, weights_index, cluster_index, centroids = initialize_variables(PREV_MODEL_EXIST, parent_dir,NUMBER_OF_CLUSTER, pretrain)
        weights = compute_weights(weights_index, centroids_var, NUMBER_OF_CLUSTER)

        x = tf.placeholder(tf.float32, [None, 32, 32, 3])
        y = tf.placeholder(tf.float32, [None, NUM_CLASSES])

        if (TRAIN == 1):
            TRAIN_OR_TEST = 1
        else:
            TRAIN_OR_TEST = 0

        keep_prob = tf.placeholder(tf.float32)
        images = pre_process(x, TRAIN_OR_TEST)
        # images = pre_process(x, 1)
        pred = cov_network(images, weights, biases, keep_prob)
        # print(pred)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y)
        loss_value = tf.reduce_mean(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



        # global_step = tf.contrib.framework.get_or_create_global_step()
        #
        # num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
        # decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
        #
        # # Decay the learning rate exponentially based on the number of steps.
        # lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
        #                               global_step,
        #                               decay_steps,
        #                               LEARNING_RATE_DECAY_FACTOR,
        #                               staircase=True)
        #
        # opt = tf.train.GradientDescentOptimizer(lr)
        # grads = opt.compute_gradients(loss_value)
        # org_grads = [(ClipIfNotNone(grad), var) for grad, var in grads]
        # new_grads = mask_gradients(weights, org_grads, weights_mask, biases, biases_mask)

        # # Apply gradients.
        # train_step = opt.apply_gradients(new_grads, global_step=global_step)
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss_value)
        # variable_averages = tf.train.ExponentialMovingAverage(
        #   MOVING_AVERAGE_DECAY, global_step)
        # variables_averages_op = variable_averages.apply(tf.trainable_variables())


        accuracy_list = np.zeros(30)
        accuracy_list = np.zeros(5)
        init = tf.global_variables_initializer()
        # Launch the graph
        print('Graph launching ..')
        with tf.Session() as sess:
            sess.run(init)

            keys = ['cov1', 'cov2', 'fc1', 'fc2', 'fc3']
            # for key in keys:
            #     sess.run(weights[key].assign(weights[key].eval(session= sess)*weights_mask[key]))
                # sess.run(biases[key].assign(biases[key].eval()*biases_mask[key]))

            print('pre train pruning info')
            prune_info(weights, 0)
            print(78*'-')
            start = time.time()
            if TRAIN == 1:
                for i in range(0,10000):
                    (batch_x, batch_y) = t_data.feed_next_batch(BATCH_SIZE)
                    train_acc, cross_en = sess.run([accuracy, loss_value], feed_dict = {
                                    x: batch_x,
                                    y: batch_y,
                                    keep_prob: 1.0})
                    if (i % DISPLAY_FREQ == 0):
                        # prune_info(weights, 0)
                        print('This is the {}th iteration, time is {}'.format(
                            i,
                            time.time() - start
                        ))
                        print("accuracy is {} and cross entropy is {}".format(
                            train_acc,
                            cross_en
                        ))
                        # accuracy_list = np.concatenate((np.array([train_acc]),accuracy_list[0:29]))
                        accuracy_list = np.concatenate((np.array([train_acc]),accuracy_list[0:4]))
                        if (np.mean(accuracy_list) > 0.8):
                            print("training accuracy is large, show the list: {}".format(accuracy_list))
                            NUMBER_OF_BATCH = 10000 / BATCH_SIZE
                            t_acc = []
                            for i in range(0,NUMBER_OF_BATCH):
                                (batch_x, batch_y) = test_data.feed_next_batch(BATCH_SIZE)
                                test_acc = sess.run(accuracy, feed_dict = {
                                                        x: batch_x,
                                                        y: batch_y,
                                                        keep_prob: 1.0})
                                t_acc.append(test_acc)
                            print("test accuracy is {}".format(t_acc))
                            test_acc = np.mean(t_acc)
                            # test_acc = sess.run(accuracy, feed_dict = {
                            #                         x: images_test,
                            #                         y: labels_test,
                            #                         keep_prob: 1.0})
                            # accuracy_list = np.zeros(30)
                            accuracy_list = np.zeros(5)
                            print('test accuracy is {}'.format(test_acc))
                            if (test_acc > 0.823):
                                print('Exiting the training, test accuracy is {}'.format(test_acc))
                                break
                    _ = sess.run(train_step, feed_dict = {
                                    x: batch_x,
                                    y: batch_y,
                                    keep_prob: dropout})
            if (TRAIN == 1):
                keys = ['cov1','cov2','fc1','fc2','fc3']
                weights_save = {}
                centroids_save = {}
                for key in keys:
                    centroids_save[key] = centroids_var[key].eval()
                    weights_save[key] = weights[key].eval()

                with open(parent_dir + 'weights/'+ 'weights'+str(NUMBER_OF_CLUSTER)+'.pkl','wb') as f:
                    pickle.dump((weights_save, biases_orgs, cluster_index,centroids_save),f)


            NUMBER_OF_BATCH = 10000 / BATCH_SIZE
            t_acc = []
            for i in range(0,NUMBER_OF_BATCH):
                (batch_x, batch_y) = test_data.feed_next_batch(BATCH_SIZE)
                test_acc = sess.run(accuracy, feed_dict = {
                                        x: batch_x,
                                        y: batch_y,
                                        keep_prob: 1.0})
                t_acc.append(test_acc)
            print("test accuracy is {}".format(t_acc))
                # save_pkl_model(weights, biases, model_name)
        return np.mean(t_acc)
    except Usage, err:
        print >> sys.stderr, err.msg
        print >> sys.stderr, "for help use --help"
        return
if __name__ == '__main__':
    sys.exit(main())
