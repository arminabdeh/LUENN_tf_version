import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


import pandas as pd
import numpy as np
import os
import sys 

def emitters_sampling(param, save=True):
	N_frames = param['sampling']['N_frames'] 
	n_min    = param['sampling']['n_min']    
	n_max    = param['sampling']['n_max']    
	Imean    = param['sampling']['Imean']    
	Isig     = param['sampling']['Isig']     
	x_min =  param['sampling']['domain_pool'][0][0]
	x_max =  param['sampling']['domain_pool'][0][1]
	y_min =  param['sampling']['domain_pool'][1][0]
	y_max =  param['sampling']['domain_pool'][1][1]
	z_min =  param['sampling']['domain_pool'][2][0]
	z_max =  param['sampling']['domain_pool'][2][1]
	x_px_size = param['sampling']['x_px_size']
	y_px_size = param['sampling']['y_px_size']
	ns = list(abs(np.random.uniform(n_min,n_max, N_frames)))
	x_domain = np.random.uniform(x_min,x_max,100000)
	y_domain = np.random.uniform(y_min,y_max,100000)
	z_domain = np.random.uniform(z_min,z_max,100000)
	i_domain = np.random.normal(Imean,Isig, 100000)
	emitter_dic = {'frame_id':[],'seed_id':[],'X_tr_px':[],'Y_tr_px':[],
	'X_tr_nm':[],'Y_tr_nm':[],'Z_tr_nm':[],'photons':[],'XYZ_set':[]}
	GT = []
	for f in range(0,N_frames):
		N = int(ns[f])
		xs = np.random.choice(x_domain,N,replace=False)
		ys = np.random.choice(y_domain,N,replace=False)
		zs = np.random.choice(z_domain,N,replace=False)
		Is = np.random.choice(i_domain,N,replace=False)
		for nn in range(0,N):
			GT_frame = emitter_dic.copy()
			GT_frame['frame_id'] = int(f+1)
			GT_frame['seed_id']  = int(nn+1)
			GT_frame['X_tr_px']  = xs[nn]
			GT_frame['Y_tr_px']  = ys[nn]
			GT_frame['X_tr_nm']  = xs[nn]*x_px_size
			GT_frame['Y_tr_nm']  = ys[nn]*y_px_size
			GT_frame['Z_tr_nm']  = zs[nn]
			GT_frame['XYZ_set']  = [xs[nn],ys[nn],zs[nn]]
			GT_frame['photons']  = Is[nn]
			GT.append(GT_frame)
	GT_Frames = pd.DataFrame(GT)
	if save:
		path = os.path.join(os.getcwd(), 'log')
		if not os.path.exists(path):
			os.mkdir(path)
		saved_directory = os.path.join(path, 'GTs.pkl')
		GT_Frames.to_pickle(saved_directory)
		print('GT file saved...')
	return GT_Frames


