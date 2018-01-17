import time
import tensorflow as tf

class eegNet(object):
    """Builds the model"""

    def buildNetwork(self, netInput, phase):
        """
        Build the EegNet
        """

        self.phase = phase

        start_time = time.time()

        print("build network started")

        self.conv1 = self.conv_layer_eeg(netInput, "conv1", 32)
        self.pool1 = self.max_pooling_layer_eeg(self.conv1, 2, 'pool1')

        self.conv2 = self.conv_layer_eeg(self.pool1, "conv2", 64)
        self.pool2 = self.max_pooling_layer_eeg(self.conv2, 2, 'pool2')
        
        self.conv3 = self.conv_layer_eeg(self.pool2, "conv3", 128)
        self.pool3 = self.max_pooling_layer_eeg(self.conv3, 3, 'pool3')

        with tf.variable_scope('flattened'):
            self.flattened = tf.contrib.layers.flatten(self.pool3)
            print('flattened shape')
            print(self.flattened.shape)

        with tf.variable_scope('dense1'):
            self.dense1 = tf.layers.dense(self.flattened, 50, name="dense1", activation = tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            print('dense1 shape')
            print(self.dense1.shape)

        with tf.variable_scope('dropout1'):
            self.dropout1 = tf.layers.dropout(self.dense1, rate=0.50, training=self.phase)

        with tf.variable_scope('dense2'):
            self.dense2 = tf.layers.dense(self.dropout1, 25, name="dense1", activation = tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            print('dense1 shape')
            print(self.dense1.shape)

        with tf.variable_scope('dropout2'):
            self.dropout2 = tf.layers.dropout(self.dense2, rate=0.50, training=self.phase)

        with tf.variable_scope('dense3'):
            self.dense3 = tf.layers.dense(self.dropout2, 2, name="dense2")
            print('dense2 shape')
            print(self.dense3.shape)

        #Calculate softmax - this might be not used as we use softmax_cross_entropy_with_logits
        with tf.name_scope("softmax"):
            self.softmax_layer = tf.nn.softmax(self.dense3)
            print('softmax shape')
            print(self.softmax_layer.shape)

        # Select the best solution
        with tf.name_scope("argmax"):
           self.argmax_layer = tf.argmax(self.softmax_layer,1)
           print('argmax shape')
           print(self.argmax_layer.shape)

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
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                bias_initializer=tf.zeros_initializer(),
                activation=None,
                name = name)

            conv = tf.nn.relu(self.batch_norm_layer(conv))
            conv = tf.layers.dropout(conv, rate=0.1, training=self.phase)

        print(name)
        print(conv.shape)
        return conv
    
    def max_pooling_layer_eeg(self, prev_layer, poolSize, name):
        with tf.variable_scope(name):
            maxpool_layer = tf.layers.max_pooling1d(inputs = prev_layer, pool_size = poolSize, strides=poolSize, padding='VALID', name=name)
        print(name)
        print(maxpool_layer.shape)
        return maxpool_layer 

    def batch_norm_layer(self, BNinput):
        return tf.contrib.layers.batch_norm(BNinput, is_training = self.phase)