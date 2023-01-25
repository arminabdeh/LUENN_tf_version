import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import module as module
from module import Localization_3D
from module import Precision_parameters
from module import Classification
import matplotlib as mpl
dense_id = 3
directory1 = './post_results/Uncer/LSNR/Dense_'+str(dense_id)+'_LSNR_localization_unc.pkl'
dataset_1  = pd.read_pickle(directory1)
dataset_1_TP =dataset_1[dataset_1['label'] == 'TP']
err_x = dataset_1_TP.X_Err
err_y = dataset_1_TP.Y_Err
err_z = dataset_1_TP.Z_Err
Post_process = Precision_parameters(err_x,err_y,err_z)
rmse_3d_1 = Post_process.rmse_3D_decode()

directory2 = './post_results/Uncer/MSNR/Dense_'+str(dense_id)+'_MSNR_localization_unc.pkl'
dataset_2  = pd.read_pickle(directory2)
dataset_2_TP =dataset_2[dataset_2['label'] == 'TP']
err_x = dataset_2_TP.X_Err
err_y = dataset_2_TP.Y_Err
err_z = dataset_2_TP.Z_Err
Post_process = Precision_parameters(err_x,err_y,err_z)
rmse_3d_2 = Post_process.rmse_3D_decode()

directory3 = './post_results/Uncer/HSNR/Dense_'+str(dense_id)+'_HSNR_localization_unc.pkl'
dataset_3  = pd.read_pickle(directory3)
dataset_3_TP =dataset_3[dataset_3['label'] == 'TP']
err_x = dataset_3_TP.X_Err
err_y = dataset_3_TP.Y_Err
err_z = dataset_3_TP.Z_Err
Post_process = Precision_parameters(err_x,err_y,err_z)
rmse_3d_3 = Post_process.rmse_3D_decode()


mpl.rc('font',family='Times New Roman')
xylabel =  {'fontsize':'16','fontweight':'bold'}
fig = plt.figure(figsize=(7.,5.8),dpi=600,layout='tight',constrained_layout=True)
axes = fig.subplots(1, 1,squeeze=True,sharey=False)

axes[0].scatter(dataset_1_TP.Tot_Err,Data_Result_i.Tot_Unc,c=colors[0],linestyle= '-',lw=1.6,label='LSNR',marker='.',markersize=.0001,alpha=0.8)
plt.savefig('./Test.jpg')