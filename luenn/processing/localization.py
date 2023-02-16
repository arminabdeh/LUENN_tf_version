
import luenn.utils as utils
import numpy as np
import scipy
from skimage.feature import peak_local_max
import pickle
import pandas as pd
from scipy.optimize import linear_sum_assignment as matchfinder
from scipy.spatial.distance import cdist,pdist
import time
import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 



def seed_candidate_3D(psf,param, z_range=1500.,skip=False):
	if skip:
		i_skip_min =  int(4*param['sampling']['domain_pool'][0][0])
		i_skip_max =  int(4*param['sampling']['domain_pool'][0][1])
		j_skip_min =  int(4*param['sampling']['domain_pool'][1][0])
		j_skip_max =  int(4*param['sampling']['domain_pool'][1][1])
	else:
		i_skip_min =  0
		i_skip_max =  255
		j_skip_min =  0
		j_skip_max =  255
	x_px_size = param['sampling']['x_px_size']
	y_px_size = param['sampling']['y_px_size']
	threshold_clean       = param['localization']['threshold_clean']
	threshold_abs         = param['localization']['threshold_abs']
	threshold_distance    = param['localization']['threshold_distance']
	threshold_freq_sum    = param['localization']['threshold_freq_sum']
	threshold_freq_max    = param['localization']['threshold_freq_max']
	radius_lat            = param['localization']['radius_lat']
	radius_axi            = param['localization']['radius_axi']
	psfs_cos = psf[:,:,0]
	psfs_sin = psf[:,:,1]
	psfs_norm  = np.sqrt(np.square(psfs_cos)+np.square(psfs_sin))
	eps = 10e-8
	psfs_Z    = np.arccos(np.divide(psfs_cos,psfs_norm+eps))/3.14159265359
	psfs_norm = psfs_norm/10000.
	psfs_norm_clean = np.where(psfs_norm<=threshold_clean,0,psfs_norm)
	label, features = scipy.ndimage.label(psfs_norm_clean)
	local_maximals = peak_local_max(psfs_norm,threshold_abs=threshold_abs,exclude_border=True,min_distance=threshold_distance,labels=label)
	count_detected = len(local_maximals)
	candidates = []
	result_dic = {'X_pr_px':[],'Y_pr_px':[],'X_pr_nm':[],'Y_pr_nm':[],'Z_pr_nm':[],'Id_i':[],'Id_j':[],'Freq_max':[],'Freq_sum':[]}
	for i in range(0,count_detected):
		Id_i = local_maximals[i][0]
		Id_j = local_maximals[i][1]
		if (Id_i>=i_skip_min and Id_i<=i_skip_max and Id_j>=j_skip_min and Id_j<=j_skip_max):
			I_max = psfs_norm[Id_i,Id_j]
			I_sum = psfs_norm[Id_i-1,Id_j]+psfs_norm[Id_i+1,Id_j]+psfs_norm[Id_i,Id_j]+psfs_norm[Id_i,Id_j-1]+psfs_norm[Id_i,Id_j+1]
			Dist_X = [psfs_norm[Id_i-radius_lat,Id_j],psfs_norm[Id_i,Id_j],psfs_norm[Id_i+radius_lat,Id_j]]
			Dist_Y = [psfs_norm[Id_i,Id_j-radius_lat],psfs_norm[Id_i,Id_j],psfs_norm[Id_i,Id_j+radius_lat]]
			x_correction = utils.sigma([-1*radius_lat,0.,radius_lat],Dist_X)
			y_correction = utils.sigma([-1*radius_lat,0.,radius_lat],Dist_Y)
			X_px = (y_correction+Id_j)/4.
			Y_px = (x_correction+Id_i)/4.
			X_nm = X_px*x_px_size
			Y_nm = Y_px*y_px_size
			Z_nm = (np.average(psfs_Z[Id_i-radius_axi:Id_i+radius_axi+1,Id_j-radius_axi:Id_j+radius_axi+1],weights=psfs_norm[Id_i-radius_axi:Id_i+radius_axi+1,Id_j-radius_axi:Id_j+radius_axi+1])*z_range)-(z_range/2)
			res = result_dic.copy()
			res['X_pr_px']     =  X_px
			res['Y_pr_px']     =  Y_px
			res['X_pr_nm']     =  X_nm
			res['Y_pr_nm']     =  Y_nm
			res['Z_pr_nm']     =  Z_nm
			res['Id_i']       = Id_i
			res['Id_j']       = Id_j
			res['Freq_max']   =  I_max
			res['Freq_sum']   =  I_sum
			candidates.append(res)
	data_candids = pd.DataFrame(candidates)
	if len(data_candids)>0:
		data_candids_filter = data_candids[(data_candids['Freq_sum']>threshold_freq_sum)|(data_candids['Freq_max']>threshold_freq_max)]
	if len(data_candids)==0:
		candidates.append(result_dic)
		data_candids_filter = data_candids
	return data_candids_filter
	
def match_finder(PR_Frame,GT_Frame):
	pred_xy = []
	pred_xyz = []
	pr_xyz = list(PR_Frame[['X_pr_nm', 'Y_pr_nm', 'Z_pr_nm']].to_records(index=False))
	for pr in pr_xyz:
		pred_xy.append([pr[0],pr[1]])
		pred_xyz.append([pr[0],pr[1],pr[2]])

	true_xy = []
	true_xyz = []
	tr_xyz = list(GT_Frame[['X_tr_nm', 'Y_tr_nm', 'Z_tr_nm']].to_records(index=False))
	for tr in tr_xyz:
		true_xy.append([tr[0],tr[1]])
		true_xyz.append([tr[0],tr[1],tr[2]])

	pred_z = PR_Frame.Z_pr_nm.to_list()
	true_z = GT_Frame.Z_tr_nm.to_list()


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
			if cost2D[id1[i],id2[i]]<=250.:
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


def localization_3D(psf,param,GT_Frame=[],save=True):
	print(psf.shape)

	t1 = time.process_time()
	PR_Frame = seed_candidate_3D(psf,param)
	if len(GT_Frame)==0:
		GtPr_frame = PR_Frame
		if save:
				path = os.path.join(os.getcwd(), 'log')
				if not os.path.exists(path):
					os.mkdir(path)
				saved_directory = os.path.join(path, 'localization.csv')
				GtPr_frame.to_csv(saved_directory)
				print('localization is done. File saved...')
	else:
		labels = match_finder(PR_Frame,GT_Frame)
		labels_TP = labels[labels.Condition=='TP']
		labels_FP = labels[labels.Condition=='FP']
		labels_FN = labels[labels.Condition=='FN']
		GtPr_frame = []
		seed_counter = 1
		for i in range(0,len(labels_TP)):
			seed_pr_id = labels_TP.PR_id.to_list()[i]
			PR_seed =PR_Frame.iloc[seed_pr_id:seed_pr_id+1,0:]
			seed_tr_id = labels_TP.GT_id.to_list()[i]
			GT_seed = GT_Frame.iloc[seed_tr_id:seed_tr_id+1,0:8]
			GT_seed['seed_id'] = seed_counter
			PR_seed['label']='TP'
			GtPr_seed = GT_seed.join(PR_seed,how='cross')
			if i==0:
				GtPr_frame = GtPr_seed
			else:
				GtPr_frame = pd.concat([GtPr_frame,GtPr_seed])
			seed_counter+=1

		for j in range(0,len(labels_FN)):
			seed_tr_id = labels_FN.GT_id.to_list()[j]
			GT_seed = GT_Frame.iloc[seed_tr_id:seed_tr_id+1,0:8]
			GT_seed['seed_id'] = seed_counter
			GT_seed['label']='FN'
			if len(GtPr_frame)==0:
				GtPr_frame = GT_seed
			else:
				GtPr_frame = pd.concat([GtPr_frame,GT_seed])
			seed_counter+=1

		for k in range(0,len(labels_FP)):
			seed_pr_id = labels_FP.PR_id.to_list()[k]
			PR_seed = PR_Frame.iloc[seed_pr_id:seed_pr_id+1,0:]
			F_id = GtPr_frame.frame_id.max()
			PR_seed['label']='FP'
			PR_seed['frame_id']=F_id
			PR_seed['seed_id']=0
			if len(GtPr_frame)==0:
				GtPr_frame = PR_seed
			else:
				GtPr_frame = pd.concat([GtPr_frame,PR_seed])

		GtPr_frame = GtPr_frame.reset_index(drop=True)
		t2 = time.process_time()
		# Not reporting yet...
		processing_time = t2-t1

		if save:
			path = os.path.join(os.getcwd(), 'log')
			if not os.path.exists(path):
				os.mkdir(path)
			saved_directory = os.path.join(path, 'localization_'+str(GtPr_frame.frame_id.max())+'.csv')
			GtPr_frame.to_csv(saved_directory)
			print('localization is done. File saved...')
	return GtPr_frame