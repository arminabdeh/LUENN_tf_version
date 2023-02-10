import pandas as pd
import numpy as np
import torch


def Emitters_sampling(N_frame,n_min,n_max,Imean,Isig,x_range,y_range,z_range,x_px_size,y_px_size):
	x_min = x_range[0]
	x_max = x_range[1]
	y_min = y_range[0]
	y_max = y_range[1]
	z_min = z_range[0]
	z_max = z_range[1]
	ns = list(abs(np.random.uniform(n_min,n_max, N_frame)))
	x_domain = np.random.uniform(x_min,x_max,100000)
	y_domain = np.random.uniform(y_min,y_max,100000)
	z_domain = np.random.uniform(z_min,z_max,100000)
	i_domain = np.random.normal(Imean,Isig, 100000)
	emitter_dic = {'frame_id':[],'seed_id':[],'X_tr_px':[],'Y_tr_px':[],'X_tr_nm':[],'Y_tr_nm':[],'Z_tr_nm':[],'XYZ_set':[],'photons':[]}
	GT = []
	for f in range(0,N_frame):
		xyzs = []
		phot = []
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
	return GT_Frames

def Frame_simulations(sim_test,GTs,img_size):
	N_frames = GTs.frame_id.max()
	x_sim = np.zeros((N_frames,img_size[0],img_size[1]))
	for f in range(0,N_frames):
		GTs_Frame = GTs[GTs.frame_id==f+1]
		xyzs      = torch.tensor(GTs_Frame.XYZ_set.to_list())
		intensity = torch.tensor(GTs_Frame.photons.to_list())	
		frame = sim_test.psf.forward(xyz,intensity)
		frame_bg,bg = sim_test.background.forward(frame)
		frame_bg_n = sim_test.noise.forward(frame_bg)
		x_sim[f,:,:] +=np.array(frame_bg_n[0,:,:]).T
	return x_sim

def Frame_labelings(xyz,phot,dist):
	print('frame_label')
	# y,x = np.mgrid[-2:2.1:1,-2:2.1:1]
	# pos = np.dstack((x, y))
	# rv_shift = multivariate_normal([0, 0], [[1, 0], [0,1]])
	# normal_dist_shift = rv_shift.pdf(pos)/np.sum(rv_shift.pdf(pos))
	# Dis = 10000*normal_dist_shift/0.16210282
	# Dis = np.array(Dis,dtype=np.int32)
	Y_train = np.zeros((1,256,256,2), dtype=np.float32)
	# for n in range(0,len(xyz)):
	# 	xf = xyz[n][1]*4
	# 	yf = xyz[n][0]*4
	# 	xi = int(xf+.5)
	# 	yj = int(yf+.5)
	# 	z_true = xyz[n][2]
	# 	zr = 3.14159265359*((z_true+750.)/1500.)
	# 	channel_cos = np.array(dist*np.cos(zr))
	# 	channel_sin = np.array(dist*np.sin(zr))
	# 	if (xi>3 and xi<252 and yj>3 and yj<252):
	# 		Y_train[0,xi-2:xi+3,yj-2:yj+3,0] += channel_cos
	# 		Y_train[0,xi-2:xi+3,yj-2:yj+3,1] += channel_sin
	return Y_train

