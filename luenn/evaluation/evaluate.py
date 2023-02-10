import pandas as pd
import math
class Classification:
	def __init__(self,TP,FP,FN,GT):
		self.GT = GT
		self.TP = TP
		self.FP = FP
		self.FN = FN
		self.JI = []
		self.RL = []
		self.PR = []
		if self.GT<=0:
			raise Exception("Number of ground truths cannot be \"zero\" or \"negative\" nunmber")
		if self.TP<0:
			raise Exception("Number of True Positives cannot be a \"negative\" number")
		if self.FP<0:
			raise Exception("Number of False Positives cannot be a \"negative\" number")
		if self.FN<0:
			raise Exception("Number of False Negatives cannot be a \"negative\" number")
		if self.TP+self.FN!=self.GT:
			raise Exception("Total number of labeled TP & FN emitters are not match with the ground truth")

	def jaccardian_index(self):
		TPFNFP = self.TP+self.FN+self.FP
		self.JI = self.TP*100/TPFNFP
		print("Your JI is {} %".format(self.JI))
		return self.JI

	def recall(self):
		TPFN = self.TP+self.FN
		self.RL = self.TP*100/TPFN
		print("Your Recall is {} %".format(self.RL))
		return self.RL
	
	def precision(self):
		TPFP = self.TP+self.FP
		self.PR = self.TP*100/TPFP
		print("Your Precision is {} %".format(self.PR))
		return self.PR

class Localization:
	def __init__(self, res_loc):
		res_loc = pd.DataFrame(res_loc)
		if len(res_loc)==0:
			raise Exception("Localization set is \"empty\".")
		self.xe = res_loc.X_Err
		self.ye = res_loc.Y_Err
		self.ze = res_loc.Z_Err
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

def efficiency_2D(ji,rmse_lat):
	t1 = (1-(ji*0.01))**2
	t2 = ((rmse_lat**2)*2)*(.01**2)
	Eff_lateral = (1-math.sqrt(t1+t2))*100.
	return Eff_lateral

def efficiency_axial(ji,rmse_z):
	t1 = (1-(ji*0.01))**2
	t2 = (rmse_z**2)*(.005**2)
	Eff_axial = (1-math.sqrt(t1+t2))*100.
	return Eff_axial

def efficiency_3D(ji,rmse_lat,rmse_z):
	t1 = efficiency_2D(ji,rmse_lat)
	t2 = efficiency_axial(ji,rmse_z)
	Eff_volume = (t1+t2)/2
	return Eff_volume



		
	