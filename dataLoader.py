import scipy.io as sio
import numpy as np

def loadData(path):
	#Load data
	loadedData = sio.loadmat(path)

	#Define the size of data including channels dimension
	reshapeSize = (loadedData['channel1'].shape[0],loadedData['channel1'].shape[1],1)

	#Append channels together
	dataSet = np.resize(loadedData['channel1'],reshapeSize)
	dataSet = np.append(dataSet, np.resize(loadedData['channel2'],reshapeSize), 2)
	dataSet = np.append(dataSet, np.resize(loadedData['channel3'],reshapeSize), 2)

	labels = loadedData['target_labels']
	return dataSet, labels