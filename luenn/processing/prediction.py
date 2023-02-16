import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import os
import time
from progress.bar import Bar
import h5py
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import numpy as np


def splitwise_prediction(model,chunk_size,frames):
	path = os.path.join(os.getcwd(), 'log')
	if not os.path.exists(path):
		os.mkdir(path)
	saved_directory = os.path.join(path, 'outputs.hdf5')

	sets = int(frames.shape[0]/chunk_size)
	if frames.shape[0]%chunk_size!=0:
		sets += 1
	split_range = list(np.array(list(range(1,sets+1)),dtype=np.int16)*chunk_size)
	frame_split = np.split(frames, split_range)
	processing_time = 0
	frame_counter = 0
	with Bar('Processed batches...', fill='=', max=sets) as bar:
		for b_id in range(0,sets):
			if b_id==0:
				fxy = h5py.File(saved_directory, "w")
			else:
				fxy = h5py.File(saved_directory, "a")
			set_size = frame_split[b_id].shape[0]
			frame_set = frame_split[b_id].reshape(set_size,64,64,1)
			t = time.process_time()
			set_prd = model.predict(frame_set)
			processing_time += (time.process_time()-t)
			frame_counter+=set_size
			if b_id==0:
				fxy.create_dataset('outputs', data=set_prd, compression="gzip", maxshape=(None,256,256,2))
			else:
				fxy['outputs'].resize(frame_counter,axis=0)
				fxy['outputs'][-set_size:] = set_prd
			bar.next()
	predictions = fxy['outputs'][0:]
	print('LUENN prediction is done. Processing time = {} seconds'.format(processing_time))
	print('Total number of processed Frames are {} '.format(predictions.shape[0]))
	print('Output prediction frames saved...key=outputs')
	fxy.close()
	fxy = h5py.File(saved_directory, "r")
	yh = fxy['outputs'][0:]
	fxy.close()
	return yh