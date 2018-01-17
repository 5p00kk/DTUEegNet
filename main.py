import time
import os
import batchLoader
import eegNetLoader
import trainingOps
import tensorflow as tf
import numpy as np
import scipy.io as sio

model = eegNetLoader.eegNet()
batch = batchLoader.batch()
# batch_input, batch_label = batch.getTrain(3)

timestr = time.strftime("%Y%m%d-%H%M%S")
tensorboard_path=os.path.join(".\\Tensorboard", timestr)

print('Start')

dataSet_ph = tf.placeholder(tf.float32, [None, 176, 3])
labels_ph= tf.placeholder(tf.int32, [None, 1])
phase_ph = tf.placeholder(tf.bool, name='phase')

model.buildNetwork(dataSet_ph, phase_ph)

# while batch.getEpoch() == 0:
	# batch_input, batch_label = batch.getTrain(3)
	# print(batch_label)

# Define the training operations
lossOp = trainingOps.calcLoss(model.dense3, labels_ph)
trainOp = trainingOps.trainNetwork(lossOp)
accOp = trainingOps.calcAccuracy(model.argmax_layer, labels_ph)

print('The end')

init =  tf.global_variables_initializer()
with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
	sess.run(init)
	tensorboard_writer = tf.summary.FileWriter(tensorboard_path, sess.graph)

	print("number of trainable parameters :",np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
	current_epoch = 0
	i = 0
	valData, valLabels = batch.getValidation()
	cumulativeTrainLoss = 0
	cumulativeTrainAcc = 0
	maxValAcc = 0;

	while current_epoch <= 1000:
		batchData, batchLabels = batch.getTrain(16)
		i = i+1

		feed_dict = {dataSet_ph: batchData, labels_ph: batchLabels, phase_ph: 1}
		fetches_train = [trainOp, lossOp, accOp]
		_ , train_loss, train_acc = sess.run(fetches = fetches_train, feed_dict=feed_dict)

		cumulativeTrainAcc = cumulativeTrainAcc + train_acc
		cumulativeTrainLoss = cumulativeTrainLoss + train_loss

		if (i%300)==0:
			feed_valid = {dataSet_ph: valData, labels_ph: valLabels, phase_ph: 0}
			fetches_valid = [accOp, lossOp]
			[validation_acc, validation_loss] = sess.run(fetches = fetches_valid, feed_dict=feed_valid)

			print(i," Train loss",cumulativeTrainLoss/300,"    Train_acc", cumulativeTrainAcc/3, " Valid loss",validation_loss,"    Valid_acc", validation_acc*100)
			#print("Training pred and label")
			#print(result)
			#print(batchLabels)
			#print(sm)
			if(validation_acc*100 > maxValAcc):
				maxValAcc = validation_acc*100
			
			cumulativeTrainLoss = 0
			cumulativeTrainAcc = 0

		if batch.getEpoch() > current_epoch:
			current_epoch = batch.getEpoch()
			if (current_epoch%10)==0:
				print("Max val acc: ", maxValAcc)
			print("NUMBER EPOCHS: ", current_epoch)

	print("Finished training")