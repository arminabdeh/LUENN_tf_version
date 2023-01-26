import numpy as np
import random
import pickle
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.stats import multivariate_normal
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
import decode
import decode.utils
import decode.neuralfitter.train.live_engine
import torch
import pandas as pd
import h5py
import time
import module as module
from module import Localization_3D
from module import Precision_parameters
from module import Classification

xylabel =  {'fontsize':'14','fontweight':'bold'}


gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
tf.config.experimental.set_visible_devices(devices=gpus[0:], device_type="GPU")
# ==========================Callbacks===========================
prop = [{'mode':'HSNR','Density_id':'8','threshold_clean':.02,'threshold_abs':.03,
'threshold_distance':3,'threshold_freq_sum':.20,
'threshold_freq_max':.1,'radius_lat':int(1),
'radius_axi':int(1),'skip_min':[int(3),int(3)],
'skip_max':[int(252),int(252)],'bias_x':0.,'bias_y':0.,'bias_z':0.,
'px_size':[100.,100.],'z_range':1500}]
prop = pd.DataFrame(prop)
prop.to_pickle('./artificial/MT0_ld_prop.pkl')
prop = pd.read_pickle('./artificial/MT0_ld_prop.pkl')

def rendering(epoch):
	prop = pd.read_pickle('./artificial/MT0_ld_prop.pkl')
	num_of_seeds_target = 30
	GT_Frames = pd.read_pickle('./artificial/GTs/MT0_GT_S'+str(num_of_seeds_target)+'.pkl')
	GT_Frames['Z_px'] = 0
	GT_Frames['XY_px'] = 0
	GT_Frames['XYZ_px'] = 0
	# ============= Frames  ===========================
	fxy = h5py.File('./artificial/Frames/inputs_HSNR_S'+str(num_of_seeds_target)+'.hdf5', "r")
	x_sim = fxy['x_sim'][0:]
	fxy.close()
	# ============= model and prediction ===========================

	model_base_HSNR = load_model('./models/MT0_LD.h5')
	y_sim, times = module.splitwise_prediction(model_base_HSNR,256,x_sim,'./artificial/test'+str(num_of_seeds_target))
	for f in range(0,y_sim.shape[0]):
		preds_i = y_sim[f,:,:,:]
		frame_id = f+1
		GT_i    = GT_Frames[(GT_Frames['frame_id']==frame_id)&(GT_Frames['Intensity']>0.1)]
		PR_i = Localization_3D(preds_i).AS_Localization(prop)
		GT_i = GT_i[(GT_i.X_px>1)&(GT_i.X_px<63)&(GT_i.Y_px>1)&(GT_i.Y_px<63)]
		if f==0:
			processed_dataframe = module.Match_filling(GT_i,PR_i,frame_id)
		else:
			processed_dataframe_i = module.Match_filling(GT_i,PR_i,frame_id)
			processed_dataframe = pd.concat([processed_dataframe,processed_dataframe_i],ignore_index=True)
	processed_dataframe_TP = processed_dataframe[processed_dataframe['label']=='TP']
	processed_dataframe_FP = processed_dataframe[processed_dataframe['label']=='FP']
	processed_dataframe_FN = processed_dataframe[processed_dataframe['label']=='FN']
	TP_num = len(processed_dataframe_TP)
	FP_num = len(processed_dataframe_FP)
	FN_num = len(processed_dataframe_FN)
	err_x = processed_dataframe_TP['X_tr_nm']-processed_dataframe_TP['X_pr_nm']
	err_y = processed_dataframe_TP['Y_tr_nm']-processed_dataframe_TP['Y_pr_nm']
	err_z = processed_dataframe_TP['Z_tr_nm']-processed_dataframe_TP['Z_pr_nm']
	Post_process = Precision_parameters(err_x,err_y,err_z)
	rmse_3d = Post_process.rmse_3D_decode()
	rmse_2d = Post_process.rmse_2D_decode() 
	rmse_z  = Post_process.rmse_z()
	classificiation = Classification(rmse_z,rmse_3d,rmse_2d,TP_num,FP_num,FN_num)
	JI_full = classificiation.Jaccardian_Index()
	Eff_2d  = classificiation.efficiency_2D()
	Eff_3d  = classificiation.efficiency_3D()
	cm = plt.cm.get_cmap('jet')
	plt.style.use('dark_background')
	plt.figure(figsize=(14,12),dpi=400)
	plt.scatter(processed_dataframe_TP.X_pr_nm,processed_dataframe_TP.Y_pr_nm,c=processed_dataframe_TP.Z_pr_nm, vmin=-650, vmax=650, s=0.041, cmap=cm)
	plt.xlim(0,6400)
	plt.ylim(0,6400)
	plt.colorbar()
	plt.gca().invert_yaxis()
	plt.text(5000,6000, 'JI    :'+str(round(JI_full,2)),xylabel)
	plt.text(5000,5700, 'RMSE  :'+str(round(rmse_3d,2)),xylabel)
	plt.text(5000,5400, 'TP_num:'+str(round(TP_num,1)),xylabel)
	plt.text(5000,5100, 'FP_num:'+str(round(FP_num,1)),xylabel)
	plt.text(5000,4900, 'FN_num:'+str(round(FN_num,1)),xylabel)
	plt.savefig('./results/Prediction_rendered_'+str(epoch)+'.jpg')

train_loss = []
validation_loss = []
def plotter(train_loss,validation_loss):
	N = len(train_loss)
	plt.figure(figsize=(10,10))
	plt.plot(np.arange(1, N+1), train_loss, label="train_loss",c='b',ls='-')
	plt.plot(np.arange(1, N+1), validation_loss, label="val_loss",c='r',ls='-')
	plt.title("Training Loss on Dataset")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss")
	plt.legend(loc="upper right")
	plt.savefig(filepath_results+'plot_AS20_MT0.jpg')
	plt.close()
class CustomCallback(keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs=None):
		train_loss.append(logs["loss"])
		validation_loss.append(logs["val_loss"])
		plotter(train_loss,validation_loss)
		if (epoch%2==0 and epoch!=0):
			rendering(epoch)



def frame_label(xyz,phot):
	Y_train = np.zeros((1,256,256,2), dtype=np.float32)
	swap = 1
	for n in range(0,len(xyz)):
		xf = xyz[n][1]*4
		yf = xyz[n][0]*4
		xi = int(xf+0.5)
		yj = int(yf+0.5)
		z_true = xyz[n][2]
		zr = 3.14159265359*((z_true+750.)/1500.)
		channel_cos = np.cos(zr)*10000.0
		channel_sin = np.sin(zr)*10000.0
		if (xi>2 and xi<254 and yj>2 and yj<254):
			if (Y_train[0,xi,yj,0]==0 and Y_train[0,xi,yj,1]==0):
				Y_train[0,xi,yj,0] += channel_cos
				Y_train[0,xi,yj,1] += channel_sin
	return Y_train
def frame_generations(batch_size,sim_test,N_ave,N_sig,loc,t_min,t_max):
	x_train = np.zeros((batch_size,64,64,1), dtype=np.float32)
	y_train = np.zeros((batch_size,256,256,2), dtype=np.float32)
	for t in range(0,batch_size):
		list_ids = list(np.random.choice(list(np.arange(t_min,t_max)),3,replace=False))
		dic_pickle0 = pd.read_pickle('./Tubes/'+loc+'/Tube'+str(list_ids[0])+'.pkl')
		dic_pickle1 = pd.read_pickle('./Tubes/'+loc+'/Tube'+str(list_ids[1])+'.pkl')
		dic_pickle2 = pd.read_pickle('./Tubes/'+loc+'/Tube'+str(list_ids[2])+'.pkl')
		num_seeds_per_tube =  list(np.random.choice(list(np.abs(np.random.normal(N_ave,N_sig, 10000))+1),3,replace=False))           
		dic_pickle0 = dic_pickle0.sample(int(num_seeds_per_tube[0]/3.0))
		dic_pickle1 = dic_pickle1.sample(int(num_seeds_per_tube[1]/3.0))
		dic_pickle2 = dic_pickle2.sample(int(num_seeds_per_tube[2]/3.0))
		dic_pickle = pd.concat([dic_pickle0,dic_pickle1,dic_pickle2],ignore_index=True)
		N = int(len(dic_pickle))
		xyzs = []
		phot = []
		for s in range(0,N):
			data_seed     = dic_pickle.iloc[s,:]
			xyzs.append([data_seed.X_px,data_seed.Y_px,data_seed.Z_nm])
			phot.append(np.abs(np.random.normal(19900,6500, 1)[0])+100)
		xyz = torch.tensor(xyzs)
		intensity = torch.tensor(phot)
		frame = sim_test.psf.forward(xyz,intensity)
		frame_bg,bg = sim_test.background.forward(frame)
		frame_bg_n = sim_test.noise.forward(frame_bg)
		x_train[t,:,:,0] +=np.array(frame_bg_n[0,:,:]).T
		label_frame = frame_label(xyzs,phot)
		y_train[t,:,:,:] += label_frame[0,:,:,:]
	return x_train,y_train

 # ==========================Models===========================


filepath_model_saved0  = './models/AI_AS_20_LD_f2c_v3_35_Fine.h5'
filepath_model_saved1  = './models/MT0_LD.h5'
filepath_results      = './results/'

# ==========================Step 1 Training ===========================
tf.keras.backend.clear_session()
param_path = './param/param_hsnr_3d_as_train20.yaml'
param = decode.utils.param_io.load_params(param_path)
camera = decode.simulation.camera.Photon2Camera.parse(param)
simulator0, simulator = decode.neuralfitter.train.live_engine.setup_random_simulation(param)
X_test,Y_test = frame_generations(512,simulator,20,5,'train',1,31)
path_in  = './validation/in_1'
path_out = './validation/ou_1'
dataset_in = tf.data.Dataset.from_tensor_slices(X_test)
tf.data.experimental.save(dataset_in, path_in, compression='GZIP', shard_func=None)
dataset_out = tf.data.Dataset.from_tensor_slices(Y_test)
tf.data.experimental.save(dataset_out, path_out, compression='GZIP', shard_func=None)
dataset_in_1  = tf.data.experimental.load(path_in,tf.TensorSpec(shape=(64,64,1), dtype=tf.float32),compression='GZIP')
dataset_out_1 = tf.data.experimental.load(path_out,tf.TensorSpec(shape=(256,256,2), dtype=tf.float32),compression='GZIP')
dataset_test = tf.data.Dataset.zip((dataset_in_1,dataset_out_1))
batch_size = 64
print('Validation dataset size:  '+str(len(dataset_test)))
dataset_test = dataset_test.batch(batch_size)

def get_data_train(batch_size):
	while True:
		x,y = frame_generations(batch_size,simulator,20,5,'train',1,31)
		yield x,y
model = load_model(filepath_model_saved0)
b1 = 0.9
b2 = 0.999

initial_learning_rate = 0.00080
saver = keras.callbacks.ModelCheckpoint(filepath_model_saved1,monitor='val_loss', verbose=2, save_weights_only=False,save_best_only=True, mode='min', save_freq='epoch')
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,decay_steps=2048,decay_rate=0.9)
opt = keras.optimizers.Adam(lr_schedule, beta_1=b1, beta_2=b2)
loss_function = 'mse'
model.compile(loss=loss_function, optimizer=opt)

totalepochs= 10000
train_steps= 256
datagen_train = get_data_train(4)

model.fit(datagen_train, steps_per_epoch=train_steps,epochs=totalepochs, validation_data=dataset_test,verbose=2, callbacks=[CustomCallback(),saver])
# # ==========================STEP1===========================