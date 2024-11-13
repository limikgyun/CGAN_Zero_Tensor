import numpy as np
import pickle
from matplotlib import pyplot as plt
import matplotlib
from utils import *
# from sklearn.preprocessing import MinMaxScaler 

def print_csi(exp, t, n_classes, dataset):
	for p in range(n_classes):
		plt.figure()
		plt.title('Real CSI', fontsize=20)
		# plt.title('Real CSI $p_{%d}$'%(p+1) , fontsize=20)
		plt.xlabel('CSI Subcarrier Index', fontsize=16)
		plt.ylabel('CSI Amplitude (Normalized)', fontsize=16)
		plt.axis([0, 256, 0, 1])
		plt.grid(True)

		for i in dataset[p]:
			plt.plot(i)
		font = {'family' : 'DejaVu Sans','weight' : 'normal','size'   : 12}	
		matplotlib.rc('font', **font)
		plt.show()
		plt.savefig('/home/mnetlig/Desktop/CSI 시각화/%s%s-클래스#%d.png'%(exp, t, p+1), dpi=600)
		plt.close() 

def plotting(a, csi_num, file_path):
	dataset1 = data_preproc(np.asarray(pickle.load(open(file_path,'rb'))), scale_range =(0,1))
	X_tra1, X_class, _, _ = dataset1
	print(X_tra1[1])
	X_tra1 = X_tra1[:csi_num]
	X_tra1 = X_tra1.reshape(1,-1,256)
	print(X_tra1.shape)
	[3860, 1, 256, 1]
	print_csi(a, csi_num, 1, X_tra1)

def plotting_1(a, csi_num, file_path):
	dataset1 = np.asarray(pickle.load(open(file_path,'rb')))
	X_tra1 = dataset1
	X_tra1 = single_minmaxscale(X_tra1, (0, 1))

	print(X_tra1[1])
	X_tra1 = X_tra1[:csi_num]
	X_tra1 = X_tra1.reshape(1,-1,256)
	print(X_tra1.shape)
	[3860, 1, 256, 1]
	print_csi(a, csi_num, 1, X_tra1)


if __name__ == '__main__':
	plotting_1('실제CSI, 개수=', 500, '/home/mnetlig/CGAN_Zero/Storage/데이터/aa_001.pickle')
	plotting_1('실제CSI, 개수=', 500, '/home/mnetlig/CGAN_Zero/Storage/데이터/aa_002.pickle')

	