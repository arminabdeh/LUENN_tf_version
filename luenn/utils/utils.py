import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import h5py
from tensorflow.keras.models import load_model
import pandas as pd
import yaml
import math
import numpy as np
from scipy.stats import multivariate_normal

def model_load(load_directory):
	model = load_model(load_directory)
	return model

def hdf5_load(load_directory):
	input_file = h5py.File(load_directory, "r")
	frames     = input_file['inputs'][0:]
	return frames

def dic_load(load_directory):
	input_file = pd.read_pickle(load_directory)
	return input_file

def param_load(load_directory):
	with open('parameters.yaml', 'r') as f:
		param = yaml.safe_load(f)	
	return param

def model_base():
	# directory should be fixed!
	model_base = load_model('./data/luenn_base.hd5')
	return model_base

def sigma(x,w):
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
	return mu

def label_frame(GT_Frame,z_range=1500.,peak=False,swip=False):
	xyz = GT_Frame['XYZ_set'].to_list()
	Y_train = np.zeros((1,256,256,2), dtype=np.float32)
	if peak:
		if swip:
			raise Exception("Swipping is not possible when only \"peak\" is going to be turned on.")
		else:
			for n in range(0,len(xyz)):
					xf = xyz[n][1]*4
					yf = xyz[n][0]*4
					xi = int(xf+.5)
					yj = int(yf+.5)
					z_true = xyz[n][2]
					zr = 3.14159265359*((z_true+(0.5*z_range))/z_range)
					channel_cos = np.array(np.cos(zr))
					channel_sin = np.array(np.sin(zr))
					if (xi>=0 and xi<=255 and yj>=0 and yj<=255):
						Y_train[0,xi,yj,0] += channel_cos
						Y_train[0,xi,yj,1] += channel_sin
			

	else:
		y,x = np.mgrid[-2:2.1:1,-2:2.1:1]
		pos = np.dstack((x, y))
		if swip:
			for n in range(0,len(xyz)):
				xf = xyz[n][1]*4
				yf = xyz[n][0]*4
				xi = int(xf+.5)
				yj = int(yf+.5)
				xm = xf-xi
				ym = yf-yj
				rv_shift = multivariate_normal([xm, ym], [[1, 0], [0,1]])
				normal_dist_shift = rv_shift.pdf(pos)/np.sum(rv_shift.pdf(pos))
				Dis = 10000*normal_dist_shift/0.16210282
				dist = np.array(Dis,dtype=np.float32)
				z_true = xyz[n][2]
				zr = 3.14159265359*((z_true+(0.5*z_range))/z_range)
				channel_cos = np.array(dist*np.cos(zr))
				channel_sin = np.array(dist*np.sin(zr))
				if (xi>3 and xi<252 and yj>3 and yj<252):
					Y_train[0,xi-2:xi+3,yj-2:yj+3,0] += channel_cos
					Y_train[0,xi-2:xi+3,yj-2:yj+3,1] += channel_sin
		else:
			rv_shift = multivariate_normal([0, 0], [[1, 0], [0,1]])
			normal_dist_shift = rv_shift.pdf(pos)/np.sum(rv_shift.pdf(pos))
			Dis = 10000*normal_dist_shift/0.16210282
			dist = np.array(Dis,dtype=np.float32)	
			for n in range(0,len(xyz)):
				xf = xyz[n][1]*4
				yf = xyz[n][0]*4
				xi = int(xf+.5)
				yj = int(yf+.5)
				z_true = xyz[n][2]
				zr = 3.14159265359*((z_true+(0.5*z_range))/z_range)
				channel_cos = np.array(dist*np.cos(zr))
				channel_sin = np.array(dist*np.sin(zr))
				if (xi>3 and xi<252 and yj>3 and yj<252):
					Y_train[0,xi-2:xi+3,yj-2:yj+3,0] += channel_cos
					Y_train[0,xi-2:xi+3,yj-2:yj+3,1] += channel_sin

	return Y_train

def normal_label(frame):
	if frame.shape[-1]==2:
		return np.sqrt(frame[:,:,0]**2+frame[:,:,1]**2)
	else:
		raise Exception("frame should have two channels")

def seed_zoom(psf,pickle_file,seed_id,zoom_size,gt=True):
	seed_loc = pickle_file[pickle_file.seed_id==seed_id]
	if gt:
		id_j = int(seed_loc.X_tr_px.to_list()[0]*4.)
		id_i = int(seed_loc.Y_tr_px.to_list()[0]*4.)
	else:
		id_j = int(seed_loc.X_pr_px.to_list()[0]*4)
		id_i = int(seed_loc.Y_pr_px.to_list()[0]*4)
	border_size = int(zoom_size/2)
	psf_zoom = psf[id_i-border_size:id_i+border_size+1,id_j-border_size:id_j+border_size+1]
	return psf_zoom
# ==============================modify required
# Update required
def subpixel_bias_err(processed_dataframe,step_size_px):
	processed_dataframe_TP = processed_dataframe[processed_dataframe.label=='TP']
	processed_dataframe_TP['X_Subpixel_location']=list((processed_dataframe_TP['X_tr_nm']/100)-np.array((processed_dataframe_TP['X_tr_nm']/100),dtype=np.int32)-0.5)
	processed_dataframe_TP['Y_Subpixel_location']=list((processed_dataframe_TP['Y_tr_nm']/100)-np.array((processed_dataframe_TP['Y_tr_nm']/100),dtype=np.int32)-0.5)
	mesh_xy = list(np.arange(-50,50+(step_size_px*100),step_size_px*100,dtype=np.float32)/100.)
	ratio = []
	for ii in range(0,len(mesh_xy)-1):
		processed_dataframe_TP_x = processed_dataframe_TP[(processed_dataframe_TP.X_Subpixel_location>=mesh_xy[ii])&(processed_dataframe_TP.X_Subpixel_location<=mesh_xy[ii+1])]
		for jj in range(0,len(mesh_xy)-1):
			processed_dataframe_TP_xy = processed_dataframe_TP_x[(processed_dataframe_TP_x.Y_Subpixel_location>=mesh_xy[jj])&(processed_dataframe_TP_x.Y_Subpixel_location<=mesh_xy[jj+1])]
			err_x = processed_dataframe_TP_xy['X_tr_nm']-processed_dataframe_TP_xy['X_pr_nm']
			del_x = np.mean(err_x)
			err_y = processed_dataframe_TP_xy['Y_tr_nm']-processed_dataframe_TP_xy['Y_pr_nm']
			del_y = np.mean(err_y)
			err_z = processed_dataframe_TP_xy['Z_tr_nm']-processed_dataframe_TP_xy['Z_pr_nm']
			del_z = np.mean(err_z)
			tot_err = np.sqrt(del_x**2+del_y**2+del_z**2)
			ratio.append(tot_err/np.std(processed_dataframe_TP_xy.Tot_Err))
	ratio = np.array(ratio)
	ratio = ratio.reshape(int(len(ratio)**0.5),int(len(ratio)**0.5))
	return ratio
# Update required
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

# Update required
def dense_filtering():
	frames       = [25000,25000,20000,15000,10000,5000,5000,1500,1000,1000]   
	seeds        = [.6944432 ,2, 4,9,20,46,109,180,300,600]
	process_time = [421.98,535.09,633.77,604.0,518.1,277.0,355.7,85.1,52.45,86.7]  
	filters      = [[1.,0.784],[1.,0.77],[1.,0.78],[1.,0.78],[1.,0.80],[1.,0.7],[1.,0.53],[1.,0.48],[1.,0.48],[1.,0.53]]
	case ='LSNR'
	Result = []
	for dense_id in range(1,11):
		print('density_'+str(dense_id))
		num_frames = frames[dense_id-1]
		num_seeds  = seeds[dense_id-1]
		num_seeds_total  = num_frames*num_seeds
		JIs = filters[dense_id-1]
		Filters = ['NO','Yes']
		times = process_time[dense_id-1]
		Results = {'Case':[],'Level':[],'Filtered':[],'Density_id':[],'Density':[],'JI':[],'rmse_3D':[],'rmse_lat':[],'rmse_axi':[],'3D_eff':[],'T_ms_per_frame':[],'T_ms_per_seed':[]}
		for i in range(0,len(JIs)):
			processed_dataframe = pd.read_pickle('./post_results/'+case+'/localization/Results_Dense_'+str(dense_id)+'_'+case+'_localization.pkl')
			processed_dataframe_TP =processed_dataframe[processed_dataframe['label'] == 'TP']
			processed_dataframe_FP =processed_dataframe[processed_dataframe['label'] == 'FP']
			processed_dataframe_FN =processed_dataframe[processed_dataframe['label'] == 'FN']
			TP_num = len(processed_dataframe_TP) 
			FP_num = len(processed_dataframe_FP) 
			FN_num = len(processed_dataframe_FN) 
			Target_ij = JIs[i]
			num_TP_after_filter = int(Target_ij*len(processed_dataframe_TP))
			processed_dataframe_TP = processed_dataframe_TP.astype({'Tot_Err':float})
			pr_dataset_filter = processed_dataframe_TP.nsmallest(num_TP_after_filter,'Tot_Err')
			TP_num_new = len(pr_dataset_filter) 
			TP_out     = TP_num - TP_num_new
			err_x = pr_dataset_filter['X_tr_nm']-pr_dataset_filter['X_pr_nm']
			err_y = pr_dataset_filter['Y_tr_nm']-pr_dataset_filter['Y_pr_nm']
			err_z = pr_dataset_filter['Z_tr_nm']-pr_dataset_filter['Z_pr_nm']
			Post_process = Precision_parameters(err_x,err_y,err_z)
			rmse_z = Post_process.rmse_z()
			rmse_3d = Post_process.rmse_3D_decode()
			rmse_2d = Post_process.rmse_2D_decode()
			classificiation = Classification(rmse_z,rmse_3d,rmse_2d,TP_num_new,FP_num+TP_out,FN_num)
			True_density = num_seeds/(4.4**2)

			res = Results.copy()
			res['Filtered']= Filters[i]
			res['Case']= 'LUENN'
			res['Level']= case
			res['Density_id']= dense_id
			res['Density']= True_density
			res['JI']= classificiation.Jaccardian_Index()
			res['rmse_3D']= Post_process.rmse_3D_decode()
			res['rmse_lat']= Post_process.rmse_2D_decode()
			res['rmse_axi']= Post_process.rmse_z()
			res['3D_eff']= classificiation.efficiency_3D()
			res['T_ms_per_frame']= times/num_frames
			res['T_ms_per_seed']= times/num_seeds_total
			Result.append(res)

