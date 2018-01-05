import time
import os
import batchLoader
import eegNetLoader
import tensorflow as tf

model = eegNetLoader.eegNet()
batch = batchLoader.batch()
# batch_input, batch_label = batch.getTrain(3)

timestr = time.strftime("%Y%m%d-%H%M%S")
tensorboard_path=os.path.join(".\\Tensorboard", timestr)

print('Start')

dataSet_ph = tf.placeholder(tf.float32, [None, 125, 3])
labels_ph= tf.placeholder(tf.int32, [None, 1])

model.buildNetwork(dataSet_ph)

# while batch.getEpoch() == 0:
	# batch_input, batch_label = batch.getTrain(3)
	# print(batch_label)

# Define the training operations
lossOp = trainingOps.calcLoss(model.dense2, labels_ph)
trainOp = trainingOps.trainNetwork(lossOp)
accOp= trainingOps.calc_accuracy(model.argmax_layer, labels_ph)

print('The end')

init =  tf.global_variables_initializer()
with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.5)))) as sess:
	sess.run(init)
	#merged = tf.summary.merge_all()
	tensorboard_writer = tf.summary.FileWriter(tensorboard_path, sess.graph)