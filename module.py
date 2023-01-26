import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py
import time
from warnings import simplefilter
from scipy.stats import multivariate_normal
from scipy.ndimage.measurements import center_of_mass, label, maximum, maximum_position
from scipy.spatial.distance import cdist,pdist
from scipy import ndimage
from scipy.ndimage import find_objects,labeled_comprehension,standard_deviation,mean,standard_deviation
import math
from skimage.feature import peak_local_max
import decode
import decode.utils
import decode.neuralfitter.train.live_engine
import pickle
import pandas as pd
import tensorflow as tf
import scipy
from scipy.optimize import linear_sum_assignment as matchfinder
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
import decode
import random
import torch
from scipy.signal import peak_widths

simplefilter(action='ignore', category=FutureWarning)


class Localization_3D:
	def __init__(self,psf):
		self.psf       = psf
	def sigma(self,x,w):
		w1 = w[0]
		w2 = w[1]
		w3 = w[2]
		x1 = x[0]
		x2 = x[1]
		x3 = x[2]
		t1 = w1**((x3**2)-(x2**2))
		t2 = w2**((x1**2)-(x3**2))
		t3 = w3**((x2**2)-(x1**2))
		t4 = w1**(x3-x2)
		t5 = w2**(x1-x3)
		t6 = w3**(x2-x1)
		mu = 0.5*(math.log(t1*t2*t3)/math.log(t4*t5*t6))
		va = np.abs(0.5*(x3-x2)*(x1-x3)*(x2-x1)/math.log(t4*t5*t6))
		return mu,va
	def AS_Localization(self,properties):
		threshold_clean       = properties['threshold_clean'][0]
		threshold_abs         = properties['threshold_abs'][0]
		threshold_distance    = properties['threshold_distance'][0]
		skip_min              = properties['skip_min'][0]
		skip_max              = properties['skip_max'][0]
		threshold_freq_sum    = properties['threshold_freq_sum'][0]
		threshold_freq_max    = properties['threshold_freq_max'][0]
		radius_lat            = properties['radius_lat'][0]
		radius_axi            = properties['radius_axi'][0]
		bias_x                = properties['bias_x'][0]
		bias_y                = properties['bias_y'][0]
		bias_z                = properties['bias_z'][0]
		px_size               = properties['px_size'][0]
		z_range               = properties['z_range'][0]
		psfs_cos = self.psf[:,:,0]
		psfs_sin = self.psf[:,:,1]
		psfs_norm  = np.sqrt(np.square(psfs_cos)+np.square(psfs_sin))
		eps = 10e-8
		psfs_Z    = np.arccos(np.divide(psfs_cos,psfs_norm+eps))/3.14159265359
		psfs_norm = psfs_norm/10000.
		psfs_norm_clean = np.where(psfs_norm<=threshold_clean,0,psfs_norm)
		label, features = scipy.ndimage.label(psfs_norm_clean)
		local_maximals = peak_local_max(psfs_norm,threshold_abs=threshold_abs,exclude_border=True,min_distance=threshold_distance,labels=label)
		count_detected = len(local_maximals)
		candidates = []
		result_dic = {'probability':[],'X_pr_px':[],'Y_pr_px':[],'Z_pr_px':[],'XY_pr_px':[],'XYZ_pr_px':[],'X_pr_nm':[],'Y_pr_nm':[],'Z_pr_nm':[],
		'XY_pr_nm':[],'XYZ_pr_nm':[],'Id_i':[],'Id_j':[],'Freq_max':[],'Freq_sum':[],'Sigma_X':[],'Sigma_Y':[],'Sigma_Z':[],'Sigma_I':[]}
		for i in range(0,count_detected):
			Id_i = local_maximals[i][0]
			Id_j = local_maximals[i][1]
			if (Id_i>skip_min[0] and Id_i<skip_max[0] and Id_j>skip_min[1] and Id_j<skip_max[1]):
				I_max = psfs_norm[Id_i,Id_j]
				I_sum = psfs_norm[Id_i-1,Id_j]+psfs_norm[Id_i+1,Id_j]+psfs_norm[Id_i,Id_j]+psfs_norm[Id_i,Id_j-1]+psfs_norm[Id_i,Id_j+1]
				I_std = np.std(psfs_norm[Id_i-2:Id_i+3,Id_j-2:Id_j+3])	
				Dist_X = [psfs_norm[Id_i-radius_lat,Id_j],psfs_norm[Id_i,Id_j],psfs_norm[Id_i+radius_lat,Id_j]]
				Dist_Y = [psfs_norm[Id_i,Id_j-radius_lat],psfs_norm[Id_i,Id_j],psfs_norm[Id_i,Id_j+radius_lat]]
				x_correction,s_x = self.sigma([-1*radius_lat,0.,radius_lat],Dist_X)
				y_correction,s_y = self.sigma([-1*radius_lat,0.,radius_lat],Dist_Y)
				X_px = ((y_correction+Id_j)/4.)+(bias_x/px_size[0])
				Y_px = ((x_correction+Id_i)/4.)+(bias_y/px_size[1])
				Z_px = (np.average(psfs_Z[Id_i-radius_axi:Id_i+radius_axi+1,Id_j-radius_axi:Id_j+radius_axi+1],weights=psfs_norm[Id_i-radius_axi:Id_i+radius_axi+1,Id_j-radius_axi:Id_j+radius_axi+1]))*64
				Z_px +=64*bias_z/z_range
				X_nm = X_px*px_size[0]
				Y_nm = Y_px*px_size[1]
				Z_nm = (np.average(psfs_Z[Id_i-radius_axi:Id_i+radius_axi+1,Id_j-radius_axi:Id_j+radius_axi+1],weights=psfs_norm[Id_i-radius_axi:Id_i+radius_axi+1,Id_j-radius_axi:Id_j+radius_axi+1])*z_range)-(z_range/2)
				Z_nm +=bias_z
				res = result_dic.copy()
				res['probability'] =  I_max
				res['X_pr_px']     =  X_px
				res['Y_pr_px']     =  Y_px
				res['Z_pr_px']     =  Z_px
				res['XY_pr_px']    =  [X_px,Y_px]
				res['XYZ_pr_px']   =  [X_px,Y_px,Z_px]
				res['X_pr_nm']     =  X_nm
				res['Y_pr_nm']     =  Y_nm
				res['Z_pr_nm']     =  Z_nm
				res['XY_pr_nm']    =  [X_nm,Y_nm]
				res['XYZ_pr_nm']   =  [X_nm,Y_nm,Z_nm]
				res['Id_i']       = Id_i
				res['Id_j']       = Id_j
				res['Freq_max']   =  I_max
				res['Freq_sum']   =  I_sum
				res['Sigma_X'] =  s_x*(px_size[0]/4.)
				res['Sigma_Y'] =  s_y*(px_size[1]/4.)
				res['Sigma_Z'] =  np.std(psfs_Z[Id_i-2:Id_i+3,Id_j-2:Id_j+3])
				res['Sigma_I'] =  I_std
				candidates.append(res)
		data_candids = pd.DataFrame(candidates)
		if len(data_candids)>0:
			data_candids_filter = data_candids[(data_candids['Freq_sum']>threshold_freq_sum)|(data_candids['Freq_max']>threshold_freq_max)]
		if len(data_candids)==0:
			candidates.append(result_dic)
			data_candids_filter = data_candids
		return data_candids_filter
	
# -------------------------------------------------------
def consolidated_Z_range(recalls_dense):
    Zmax_id = np.argmax(recalls_dense.Recall)
    Zmax    = recalls_dense.Z_ave.to_list()[Zmax_id]
    Rmax    = recalls_dense.Recall.to_list()[Zmax_id]
    res    = peak_widths(np.array(recalls_dense.Recall),[Zmax_id],rel_height=0.5)
    Z_step = recalls_dense.Z_ave.to_list()[1]- recalls_dense.Z_ave.to_list()[0]
    Z_min  = recalls_dense.Z_ave.to_list()[0]
    left_min_recall = recalls_dense.Recall[0:Zmax_id].min()
    right_min_recall = recalls_dense.Recall[Zmax_id:].min()
    left_sign = (Rmax/2.) - left_min_recall
    right_sign = (Rmax/2.) - right_min_recall
    if left_sign<=0.:
        zh_min = -700.
    else:
        zh_min = ((res[2]*Z_step)-700)[0]
    if right_sign<=0.:
        zh_max = 700. 
    else:
        zh_max = ((res[3]*Z_step)-700)[0]
    xplot = [zh_min,zh_max]
    yplot = [Rmax/2.,Rmax/2.]
    FWHM = zh_max-zh_min
    ConsZR = FWHM*Rmax
    return Zmax,Rmax,FWHM,ConsZR,xplot,yplot



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

def frame_simulation_with_GTs(N_frame,sim_test,n_min,n_max,Imean,Isig,x_range,y_range,z_range):
	x_min = x_range[0]
	x_max = x_range[1]
	y_min = y_range[0]
	y_max = y_range[1]
	z_min = z_range[0]
	z_max = z_range[1]
	x_sim = np.zeros((N_frame,64,64), dtype=np.float32)
	ns = list(abs(np.random.uniform(n_min,n_max, N_frame)))
	x_domain = np.random.uniform(x_min,x_max,100000)
	y_domain = np.random.uniform(y_min,y_max,100000)
	z_domain = np.random.uniform(z_min,z_max,100000)
	i_domain = np.random.normal(Imean,Isig, 100000)
	GTs = {'frame_id':[],'seed_id':[],'X_px':[],'Y_px':[],'Z_px':[],'XY_px':[],'XYZ_px':[],'X_nm':[],'Y_nm':[],'Z_nm':[],'Intensity':[]}
	GT = []
	for t in range(0,N_frame):
		xyzs = []
		phot = []
		N = int(ns[t])
		xs = np.random.choice(x_domain,N,replace=False)
		ys = np.random.choice(y_domain,N,replace=False)
		zs = np.random.choice(z_domain,N,replace=False)
		Is = np.random.choice(i_domain,N,replace=False)
		for nn in range(0,N):
			GT_frame = GTs.copy()
			GT_frame['frame_id'] = int(t+1)
			GT_frame['seed_id']  = int(nn+1)
			GT_frame['X_px']     = xs[nn]
			GT_frame['Y_px']     = ys[nn]
			GT_frame['Z_px']     = 64.0*(zs[nn]+750)/1500
			GT_frame['XY_px']    = [xs[nn],ys[nn]]
			GT_frame['XYZ_px']   = [xs[nn],ys[nn],64.0*(zs[nn]+750)/1500]
			GT_frame['X_nm']     = xs[nn]*100.
			GT_frame['Y_nm']     = ys[nn]*100.
			GT_frame['Z_nm']     = zs[nn]
			GT_frame['Intensity'] = Is[nn]
			GT.append(GT_frame)
			xyzs.append([xs[nn],ys[nn],zs[nn]])
			phot.append(Is[nn])
		xyz = torch.tensor(xyzs)
		intensity = torch.tensor(phot)		
		frame = sim_test.psf.forward(xyz,intensity)
		frame_bg,bg = sim_test.background.forward(frame)
		frame_bg_n = sim_test.noise.forward(frame_bg)
		x_sim[t,:,:] +=np.array(frame_bg_n[0,:,:]).T
	GT_Frames = pd.DataFrame(GT)
	return x_sim,GT_Frames

def match_finder(true_xy,pred_xy,true_xyz,pred_xyz,true_z,pred_z):
	if len(pred_xy)==0:
		Results = []
		T_miss_id = list(range(len(true_xyz)))
		for tt in T_miss_id:
			ids = {'Condition':[],'GT_id':[],'PR_id':[]}
			ids['Condition']='FN'
			ids['GT_id']= tt
			ids['PR_id']= 'NA'
			Results.append(ids)
	elif len(true_xy)==0:
		Results = []
		P_Fals_id = list(range(len(pred_xy)))
		for pf in P_Fals_id:
			ids = {'Condition':[],'GT_id':[],'PR_id':[]}
			ids['Condition']='FP'
			ids['PR_id']= pf
			ids['GT_id']= 'NA'
			Results.append(ids)
	else:
		cost2D = cdist(true_xy,pred_xy)
		cost3D = cdist(true_xyz,pred_xyz)
		tr_id = list(range(len(true_xyz)))
		pr_id = list(range(len(pred_xyz)))
		id1,id2 = matchfinder(np.sqrt(np.sqrt(cost3D)))
		T_pair_id = []
		P_pair_id = []
		Results = []
		for i in range(0,len(id1)):
			ids = {'Condition':[],'GT_id':[],'PR_id':[]}
			if cost2D[id1[i],id2[i]]<=250:
				if abs(true_z[id1[i]]-pred_z[id2[i]])<=500.:
					T_pair_id.append(id1[i])
					P_pair_id.append(id2[i])
					ids['Condition']='TP'
					ids['GT_id']=id1[i]
					ids['PR_id']=id2[i]
					Results.append(ids)
		T_miss_id = list(set(tr_id)-set(T_pair_id))
		P_Fals_id = list(set(pr_id)-set(P_pair_id))
		for tt in T_miss_id:
			ids = {'Condition':[],'GT_id':[],'PR_id':[]}
			ids['Condition']='FN'
			ids['GT_id']= tt
			ids['PR_id']= 'NA'
			Results.append(ids)
		for pp in P_Fals_id:
			ids = {'Condition':[],'GT_id':[],'PR_id':[]}
			ids['Condition']='FP'
			ids['GT_id']= 'NA'
			ids['PR_id']= pp
			Results.append(ids)
	Res_data = pd.DataFrame.from_dict(Results)
	return Res_data


class Precision_parameters:
	def __init__(self, xe, ye, ze):
		self.xe = xe
		self.ye = ye
		self.ze = ze
	def rmse_3D_decode(self):
		xerr = np.array(self.xe)
		yerr = np.array(self.ye)
		zerr = np.array(self.ze)
		rmse = np.sqrt(np.mean(((xerr**2)+(yerr**2)+(zerr**2))/3))
		return rmse
	def rmse_2D_decode(self):
		xerr = np.array(self.xe)
		yerr = np.array(self.ye)
		rmse = np.sqrt(np.mean(((xerr**2)+(yerr**2))/2))
		return rmse
	def rmse_3D_challenge(self):
		xerr = np.array(self.xe)
		yerr = np.array(self.ye)
		zerr = np.array(self.ze)
		rmse = np.sqrt(np.mean(((xerr**2)+(yerr**2)+(zerr**2))))
		return rmse
	def rmse_2D_challenge(self):
		xerr = np.array(self.xe)
		yerr = np.array(self.ye)
		rmse = np.sqrt(np.mean(((xerr**2)+(yerr**2))))
		return rmse

	def rmse_x(self):
		xerr = np.array(self.xe)
		xrmse  = np.sqrt(np.mean(xerr**2))
		return xrmse
	def rmse_y(self):
		yerr = np.array(self.ye)
		yrmse  = np.sqrt(np.mean(yerr**2))
		return yrmse
	def rmse_z(self):
		zerr = np.array(self.ze)
		zrmse  = np.sqrt(np.mean(zerr**2))
		return zrmse	
			
	def std_x(self):
		xerr = np.array(self.xe)
		xstd  = np.std(xerr)
		return xstd	
	def std_y(self):
		yerr = np.array(self.ye)
		ystd  = np.std(yerr)
		return ystd	
	def std_z(self):
		zerr = np.array(self.ze)
		zstd  = np.std(zerr)
		return zstd	

	def del_x(self):
		xerr = np.array(self.xe)
		xdel  = np.mean(xerr)
		return xdel	
	def del_y(self):
		yerr = np.array(self.ye)
		ydel  = np.mean(yerr)
		return ydel	
	def del_z(self):
		zerr = np.array(self.ze)
		zdel  = np.mean(zerr)
		return zdel

	def mae_x(self):
		xerr = np.array(self.xe)
		xmae  = np.mean(np.abs(xerr))
		return xmae	
	def mae_y(self):
		yerr = np.array(self.ye)
		ymae  = np.mean(np.abs(yerr))
		return ymae	
	def mae_z(self):
		zerr = np.array(self.ze)
		zmae  = np.mean(np.abs(zerr))
		return zmae

class Classification:
	def __init__(self, rmse_z,rmse_3d,rmse_2d,TP,FP,FN):
		self.rmse_z  = rmse_z
		self.rmse_3d = rmse_3d
		self.rmse_2d = rmse_2d
		self.TP = TP
		self.FP = FP
		self.FN = FN
		self.JI = []
	def Jaccardian_Index(self):
		N_total = self.TP+self.FP+self.FN
		self.JI = self.TP*100/N_total
		return self.JI
	def efficiency_2D(self):
		t1 = (1-(self.Jaccardian_Index()*0.01))**2
		t2 = ((self.rmse_2d**2)*2)*(.01**2)
		self.Eff_lateral = (1-math.sqrt(t1+t2))*100.
		return self.Eff_lateral
	def efficiency_axial(self):
		t1 = (1-(self.Jaccardian_Index()*0.01))**2
		t2 = (self.rmse_z**2)*(.005**2)
		self.Eff_axial = (1-math.sqrt(t1+t2))*100.
		return self.Eff_axial
	def efficiency_3D(self):
		t1 = self.efficiency_2D()
		t2 = self.efficiency_axial()
		self.Eff_volume = (t1+t2)/2
		return self.Eff_volume



def Match_filling(GT_i,PR_i,f):
	processed = []
	process_dic = {'frame_id':[],'seed_id':[],'label':[], 'X_tr_px':[],'Y_tr_px':[],'Z_tr_px':[], 'X_tr_nm':[],'Y_tr_nm':[],'Z_tr_nm':[],
	'Intensity':[],'probability':[], 'X_pr_px':[],'Y_pr_px': [],'Z_pr_px':[], 'X_pr_nm':[],'Y_pr_nm':[],'Z_pr_nm':[], 'Id_i':[],'Id_j':[],
	'Freq_max':[],'Freq_sum':[], 'Sigma_X':[],'Sigma_Y':[],'Sigma_Z':[],'lat_Err':[],'Sigma_I':[],'Tot_Err':[],'X_Err':[],'Y_Err':[],'Z_Err':[]}
	# ================ ground truth =================
	gt_x_px      = GT_i['X_px'].to_list()
	gt_y_px      = GT_i['Y_px'].to_list()
	gt_z_px      = GT_i['Z_px'].to_list()
	gt_x_nm      = GT_i['X_nm'].to_list()
	gt_y_nm      = GT_i['Y_nm'].to_list()
	gt_z_nm      = GT_i['Z_nm'].to_list()
	gt_xy_px     = GT_i['XY_px'].to_list()
	gt_xyz_px    = GT_i['XYZ_px'].to_list()
	gt_intensity = GT_i['Intensity'].to_list()
	gt_xy_nm = []
	gt_xyz_nm = []
	for ll in range(0,len(gt_x_nm)):
		gt_xy_nm.append([gt_x_nm[ll],gt_y_nm[ll]])
		gt_xyz_nm.append([gt_x_nm[ll],gt_y_nm[ll],gt_z_nm[ll]])

	# ================ Predictions =================
	pr_N      = len(PR_i)
	seed_counter = 1
	if pr_N==0:
		for emp in range(0,len(gt_x_px)):
			process_dic = process_dic.copy()
			process_dic['frame_id']    = int(f)
			process_dic['seed_id']     = seed_counter
			process_dic['label']       = 'FN'
			process_dic['X_tr_px']     = gt_x_px[emp]
			process_dic['Y_tr_px']     = gt_y_px[emp]
			process_dic['Z_tr_px']     = gt_z_px[emp]
			process_dic['X_tr_nm']     = gt_x_nm[emp]
			process_dic['Y_tr_nm']     = gt_y_nm[emp]
			process_dic['Z_tr_nm']     = gt_z_nm[emp]
			process_dic['Intensity']   = gt_intensity[emp]
			process_dic['probability'] = 'Nan'
			process_dic['X_pr_px']     = 'Nan'
			process_dic['Y_pr_px']     = 'Nan'
			process_dic['Z_pr_px']     = 'Nan'
			process_dic['X_pr_nm']     = 'Nan'
			process_dic['Y_pr_nm']     = 'Nan'
			process_dic['Z_pr_nm']     = 'Nan'
			process_dic['Id_i']        = 'Nan'
			process_dic['Id_j']        = 'Nan'
			process_dic['Freq_max']    = 'Nan'
			process_dic['Freq_sum']    = 'Nan'
			process_dic['Sigma_X']     = 'Nan'
			process_dic['Sigma_Y']     = 'Nan'
			process_dic['Sigma_Z']     = 'Nan'
			process_dic['Sigma_I']     = 'Nan'
			process_dic['lat_Err']    = 'Nan'
			process_dic['Tot_Err']    = 'Nan'
			process_dic['X_Err']     = 'Nan'
			process_dic['Y_Err']     = 'Nan'
			process_dic['Z_Err']     = 'Nan'
			processed.append(process_dic)
			seed_counter+=1 
	if pr_N>0 :
		pr_pro    = PR_i['probability'].to_list()
		pr_x_px   = PR_i['X_pr_px'].to_list()    
		pr_y_px   = PR_i['Y_pr_px'].to_list()    
		pr_z_px   = PR_i['Z_pr_px'] .to_list()   
		pr_xy_px  = PR_i['XY_pr_px'].to_list()   
		pr_xyz_px = PR_i['XYZ_pr_px'].to_list()  
		pr_x_nm   = PR_i['X_pr_nm'].to_list()    
		pr_y_nm   = PR_i['Y_pr_nm'].to_list()    
		pr_z_nm   = PR_i['Z_pr_nm'].to_list()    
		pr_xy_nm  = PR_i['XY_pr_nm'].to_list()   
		pr_xyz_nm    = PR_i['XYZ_pr_nm'].to_list()  
		pr_class_i   = PR_i['Id_i'].to_list()       
		pr_class_j   = PR_i['Id_j'].to_list()        
		pr_class_max = PR_i['Freq_max'].to_list()   
		pr_class_sum = PR_i['Freq_sum'].to_list()   
		pr_x_sig  = PR_i['Sigma_X'].to_list()    
		pr_y_sig  = PR_i['Sigma_Y'].to_list()    
		pr_z_sig  = PR_i['Sigma_Z'].to_list()    
		pr_i_sig  = PR_i['Sigma_I'].to_list()     

		# ================ matching =================  	
		pairing_dataset = match_finder(gt_xy_nm,pr_xy_nm,gt_xyz_nm,pr_xyz_nm,gt_z_nm,pr_z_nm)
		pairing_dataset_TP = pairing_dataset[pairing_dataset['Condition']=='TP']
		pairing_dataset_FP = pairing_dataset[pairing_dataset['Condition']=='FP']
		pairing_dataset_FN = pairing_dataset[pairing_dataset['Condition']=='FN']
		gtid = pairing_dataset_TP['GT_id'].to_list()
		prid = pairing_dataset_TP['PR_id'].to_list()
		fnid = pairing_dataset_FN['GT_id'].to_list()
		fpid = pairing_dataset_FP['PR_id'].to_list()
		for sd in range(0,len(gtid)):
			xe = gt_x_nm[gtid[sd]]-pr_x_nm[prid[sd]]
			ye = gt_y_nm[gtid[sd]]-pr_y_nm[prid[sd]]
			ze = gt_z_nm[gtid[sd]]-pr_z_nm[prid[sd]]
			le = Precision_parameters(xe,ye,ze).rmse_2D_decode()
			te = Precision_parameters(xe,ye,ze).rmse_3D_decode()
			process_dic = process_dic.copy()
			process_dic['frame_id']    = int(f)
			process_dic['seed_id']     = seed_counter
			process_dic['label']       = 'TP'
			process_dic['X_tr_px']     = gt_x_px[gtid[sd]]
			process_dic['Y_tr_px']     = gt_y_px[gtid[sd]]
			process_dic['Z_tr_px']     = gt_z_px[gtid[sd]]
			process_dic['X_tr_nm']     = gt_x_nm[gtid[sd]]
			process_dic['Y_tr_nm']     = gt_y_nm[gtid[sd]]
			process_dic['Z_tr_nm']     = gt_z_nm[gtid[sd]]
			process_dic['Intensity']   = gt_intensity[gtid[sd]]
			process_dic['probability'] = pr_pro[prid[sd]]
			process_dic['X_pr_px']     = pr_x_px[prid[sd]]
			process_dic['Y_pr_px']     = pr_y_px[prid[sd]]
			process_dic['Z_pr_px']     = pr_z_px[prid[sd]]
			process_dic['X_pr_nm']     = pr_x_nm[prid[sd]]
			process_dic['Y_pr_nm']     = pr_y_nm[prid[sd]]
			process_dic['Z_pr_nm']     = pr_z_nm[prid[sd]]
			process_dic['Id_i']        = pr_class_i[prid[sd]]
			process_dic['Id_j']        = pr_class_j[prid[sd]]
			process_dic['Freq_max']    = pr_class_max[prid[sd]]
			process_dic['Freq_sum']    = pr_class_sum[prid[sd]]
			process_dic['Sigma_X']     = pr_x_sig[prid[sd]]
			process_dic['Sigma_Y']     = pr_y_sig[prid[sd]]
			process_dic['Sigma_Z']     = pr_z_sig[prid[sd]]
			process_dic['Sigma_I']     = pr_i_sig[prid[sd]]
			process_dic['lat_Err']    = le
			process_dic['Tot_Err']    = te
			process_dic['X_Err']     = xe
			process_dic['Y_Err']     = ye
			process_dic['Z_Err']     = ze
			processed.append(process_dic)
			seed_counter+=1			
		for sd in range(0,len(fnid)):
			process_dic = process_dic.copy()
			process_dic['frame_id']    = int(f)
			process_dic['seed_id']     = seed_counter
			process_dic['label']       = 'FN'
			process_dic['X_tr_px']     = gt_x_px[fnid[sd]]
			process_dic['Y_tr_px']     = gt_y_px[fnid[sd]]
			process_dic['Z_tr_px']     = gt_z_px[fnid[sd]]
			process_dic['X_tr_nm']     = gt_x_nm[fnid[sd]]
			process_dic['Y_tr_nm']     = gt_y_nm[fnid[sd]]
			process_dic['Z_tr_nm']     = gt_z_nm[fnid[sd]]
			process_dic['Intensity']   = gt_intensity[fnid[sd]]
			process_dic['probability'] = 'Nan'
			process_dic['X_pr_px']     = 'Nan'
			process_dic['Y_pr_px']     = 'Nan'
			process_dic['Z_pr_px']     = 'Nan'
			process_dic['X_pr_nm']     = 'Nan'
			process_dic['Y_pr_nm']     = 'Nan'
			process_dic['Z_pr_nm']     = 'Nan'
			process_dic['Id_i']        = 'Nan'
			process_dic['Id_j']        = 'Nan'
			process_dic['Freq_max']    = 'Nan'
			process_dic['Freq_sum']    = 'Nan'
			process_dic['Sigma_X']     = 'Nan'
			process_dic['Sigma_Y']     = 'Nan'
			process_dic['Sigma_Z']     = 'Nan'
			process_dic['Sigma_I']     = 'Nan'
			process_dic['lat_Err']    = 'Nan'
			process_dic['Tot_Err']    = 'Nan'
			process_dic['X_Err']     = 'Nan'
			process_dic['Y_Err']     = 'Nan'
			process_dic['Z_Err']     = 'Nan'
			processed.append(process_dic)
			seed_counter+=1
		for sd in range(0,len(fpid)):
			process_dic = process_dic.copy()
			process_dic['frame_id']    = int(f)
			process_dic['seed_id']     = seed_counter
			process_dic['label']       = 'FP'
			process_dic['X_tr_px']     = 'Nan'
			process_dic['Y_tr_px']     = 'Nan'
			process_dic['Z_tr_px']     = 'Nan'
			process_dic['X_tr_nm']     = 'Nan'
			process_dic['Y_tr_nm']     = 'Nan'
			process_dic['Z_tr_nm']     = 'Nan'
			process_dic['Intensity']   = 'Nan'
			process_dic['probability'] = pr_pro[fpid[sd]]
			process_dic['X_pr_px']     = pr_x_px[fpid[sd]]
			process_dic['Y_pr_px']     = pr_y_px[fpid[sd]]
			process_dic['Z_pr_px']     = pr_z_px[fpid[sd]]
			process_dic['X_pr_nm']     = pr_x_nm[fpid[sd]]
			process_dic['Y_pr_nm']     = pr_y_nm[fpid[sd]]
			process_dic['Z_pr_nm']     = pr_z_nm[fpid[sd]]
			process_dic['Id_i']        = pr_class_i[fpid[sd]]
			process_dic['Id_j']        = pr_class_j[fpid[sd]]
			process_dic['Freq_max']    = pr_class_max[fpid[sd]]
			process_dic['Freq_sum']    = pr_class_sum[fpid[sd]]
			process_dic['Sigma_X']     = pr_x_sig[fpid[sd]]
			process_dic['Sigma_Y']     = pr_y_sig[fpid[sd]]
			process_dic['Sigma_Z']     = pr_z_sig[fpid[sd]]
			process_dic['Sigma_I']     = pr_i_sig[fpid[sd]]
			process_dic['lat_Err']    = 'Nan'
			process_dic['Tot_Err']    = 'Nan'
			process_dic['X_Err']     = 'Nan'
			process_dic['Y_Err']     = 'Nan'
			process_dic['Z_Err']     = 'Nan'
			processed.append(process_dic)
			seed_counter+=1
	processed_dataframe = pd.DataFrame.from_dict(processed)
	return processed_dataframe

def frame_label(xyz,phot,dist):
	# y,x = np.mgrid[-2:2.1:1,-2:2.1:1]
	# pos = np.dstack((x, y))
	# rv_shift = multivariate_normal([0, 0], [[1, 0], [0,1]])
	# normal_dist_shift = rv_shift.pdf(pos)/np.sum(rv_shift.pdf(pos))
	# Dis = 10000*normal_dist_shift/0.16210282
	# Dis = np.array(Dis,dtype=np.int32)
	Y_train = np.zeros((1,256,256,2), dtype=np.float32)
	for n in range(0,len(xyz)):
		xf = xyz[n][1]*4
		yf = xyz[n][0]*4
		xi = int(xf+.5)
		yj = int(yf+.5)
		z_true = xyz[n][2]
		zr = 3.14159265359*((z_true+750.)/1500.)
		channel_cos = np.array(dist*np.cos(zr))
		channel_sin = np.array(dist*np.sin(zr))
		if (xi>3 and xi<252 and yj>3 and yj<252):
			Y_train[0,xi-2:xi+3,yj-2:yj+3,0] += channel_cos
			Y_train[0,xi-2:xi+3,yj-2:yj+3,1] += channel_sin
	return Y_train
