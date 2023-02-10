def splitwise_prediction(model,chunk_size,frames,saved_directory):
	sets = int(frames.shape[0]/chunk_size)
	split_range = list(np.array(list(range(1,sets+1)),dtype=np.int16)*chunk_size)
	frame_split = np.split(frames, split_range)
	processing_time = 0
	frame_counter = 0
	for i in range(0,len(frame_split)):
		if i==0:
			fxy = h5py.File('./'+saved_directory+'_outputs.hdf5', "w")
		else:
			fxy = h5py.File('./'+saved_directory+'_outputs.hdf5', "a")
		set_size = frame_split[i].shape[0]
		frame_set = frame_split[i].reshape(set_size,64,64,1)
		t = time.process_time()
		set_prd = model.predict(frame_set)
		processing_time += (time.process_time()-t)
		if i%4==0:
			print('Processed Frames : '+str(frame_counter))
		frame_counter+=set_size
		if i==0:
			fxy.create_dataset('outputs', data=set_prd, compression="gzip", maxshape=(None,256,256,2))
		else:
			fxy['outputs'].resize(frame_counter,axis=0)
			fxy['outputs'][-set_size:] = set_prd
	print('Saved')
	predictions = fxy['outputs'][0:]
	print(predictions.shape)
	fxy.close()
	return predictions,processing_time