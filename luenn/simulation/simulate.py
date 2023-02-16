import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import pandas as pd
import numpy as np
import torch
import os
import h5py



def frame_simulation(sim_test,GTs,param):
	path = os.path.join(os.getcwd(), 'log')
	if not os.path.exists(path):
		os.mkdir(path)
	saved_directory = os.path.join(path, 'inputs.hdf5')
	fxy = h5py.File(saved_directory, "w")
	img_size = param['simulation']['img_size']
	N_frames = GTs.frame_id.max()
	x_sim = np.zeros((N_frames,int(img_size[0]),int(img_size[1]),1))
	for f in range(0,N_frames):
		GTs_Frame = GTs[GTs.frame_id==f+1]
		xyzs      = torch.tensor(GTs_Frame.XYZ_set.to_list())
		intensity = torch.tensor(GTs_Frame.photons.to_list())	
		frame = sim_test.psf.forward(xyzs,intensity)
		frame_bg,bg = sim_test.background.forward(frame)
		frame_bg_n = sim_test.noise.forward(frame_bg)
		x_sim[f,:,:,0] +=np.array(frame_bg_n[0,:,:]).T
	fxy.create_dataset('inputs', data=x_sim, compression="gzip", maxshape=(None,64,64,1))
	print('Input frames saved...key=inputs')
	return x_sim

