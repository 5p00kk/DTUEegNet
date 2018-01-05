import time
import tensorflow as tf

class eegNet(object):
    """Builds the model"""

    def buildNetwork(self, netInput):
        """
        Build the EegNet
        """

        start_time = time.time()

        print("build network started")

        self.conv1 = self.conv_layer_eeg(netInput, "conv1", 20)
        self.pool1 = self.max_pooling_layer_eeg(self.conv1, 2, 'pool1')

        self.conv2 = self.conv_layer_eeg(self.pool1, "conv2", 40)
        self.pool2 = self.max_pooling_layer_eeg(self.conv2, 2, 'pool2')
        
        self.conv3 = self.conv_layer_eeg(self.pool2, "conv3", 60)
        self.pool3 = self.max_pooling_layer_eeg(self.conv3, 3, 'pool3')

        with tf.variable_scope('dense1'):
            self.dense1 = tf.layers.dense(self.pool3, 100, name="dense1")

        with tf.variable_scope('dense2'):
            self.dense2 = tf.layers.dense(self.dense1, 2, name="dense2")

        #Calculate softmax - this might be not used as we use softmax_cross_entropy_with_logits
        with tf.name_scope("softmax"):
            self.softmax_layer = tf.nn.softmax(self.dense2)

        Select the best solution
        with tf.name_scope("argmax"):
           self.argmax_layer = tf.argmax(self.softmax_layer)

        print(("Network build finished: %ds" % (time.time() - start_time)))

    def conv_layer_eeg(self, prev_layer, name, size_out):
        with tf.variable_scope(name):
            #Added weight initialization as described in the paper
            conv = tf.layers.conv1d(
                inputs=prev_layer,
                filters=size_out,
                kernel_size=3,
                padding="VALID",
                use_bias=True,
                bias_initializer=tf.zeros_initializer(),
                activation=tf.nn.relu,
                name = name)
        print(name)
        print(conv.shape)
        return conv
    
    def max_pooling_layer_eeg(self, prev_layer, poolSize, name):
        with tf.variable_scope(name):
            maxpool_layer = tf.layers.max_pooling1d(inputs = prev_layer, pool_size = poolSize, strides=poolSize, padding='VALID', name=name)
        print(name)
        print(maxpool_layer.shape)
        return maxpool_layer 