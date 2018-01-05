import time
import os
import batchLoader
import eegNetLoader
import trainingOps
import tensorflow as tf
import numpy as np

model = eegNetLoader.eegNet()
batch = batchLoader.batch()
# batch_input, batch_label = batch.getTrain(3)

timestr = time.strftime("%Y%m%d-%H%M%S")
tensorboard_path=os.path.join(".\\Tensorboard", timestr)

print('Start')

dataSet_ph = tf.placeholder(tf.float32, [None, 750, 3])
labels_ph= tf.placeholder(tf.int32, [None, 1])

model.buildNetwork(dataSet_ph)

# while batch.getEpoch() == 0:
	# batch_input, batch_label = batch.getTrain(3)
	# print(batch_label)

# Define the training operations
lossOp = trainingOps.calcLoss(model.dense2, labels_ph)
trainOp = trainingOps.trainNetwork(lossOp)
accOp = trainingOps.calcAccuracy(model.argmax_layer, labels_ph)

print('The end')

init =  tf.global_variables_initializer()
with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.5)))) as sess:
	sess.run(init)
	merged = tf.summary.merge_all()
	tensorboard_writer = tf.summary.FileWriter(tensorboard_path, sess.graph)

	print("number of trainable parameters :",np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
	current_epoch = 0
	i = 0

	while current_epoch <= 20:
		batchData, batchLabels = batch.getTrain(5)
		i = i+1

		feed_dict = {dataSet_ph: batchData, labels_ph: batchLabels}
		fetches_train = [model.argmax_layer, merged, trainOp, lossOp]
		result, summary, _ , loss = sess.run(fetches = fetches_train, feed_dict=feed_dict)
		
		tensorboard_writer.add_summary(summary,i)

		if (i%100)==0:
			[G_acc] = sess.run(fetches = [accOp], feed_dict=feed_dict)
			print(i,"	Test loss",loss,"	G_acc", G_acc*100)

		if batch.getEpoch() > current_epoch:
			current_epoch = batch.getEpoch()
			print("NUMBER EPOCHS: ", current_epoch)

	print("Finished training")