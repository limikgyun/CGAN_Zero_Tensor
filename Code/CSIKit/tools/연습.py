from CSIKit.filters.passband import lowpass
from CSIKit.filters.statistical import running_mean
from CSIKit.util.filters import hampel

from CSIKit.reader import get_reader
from CSIKit.tools.batch_graph import BatchGraph
from CSIKit.util import csitools

import numpy as np
import pickle

#파일경로
paath="/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/x_1.pcap"
p_paath='/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/가시화용피클'

my_reader = get_reader(paath)
csi_data = my_reader.read_file(paath, scaled=True)
csi_matrix, no_frames, no_subcarriers = csitools.get_CSI(csi_data, metric="amplitude")

# CSI matrix is now returned as (no_frames, no_subcarriers, no_rx_ant, no_tx_ant).
# First we'll select the first Rx/Tx antenna pairing.
csi_matrix_first = csi_matrix[:, :, 0, 0]
# Then we'll squeeze it to remove the singleton dimensions.
csi_matrix_squeezed = np.squeeze(csi_matrix_first)

# This example assumes CSI data is sampled at ~100Hz.
# In this example, we apply (sequentially):
#  - a lowpass filter to isolate frequencies below 10Hz (order = 5)
#  - a hampel filter to reduce high frequency noise (window size = 10, significance = 3)
#  - a running mean filter for smoothing (window size = 10)

for x in range(no_frames):
  csi_matrix_squeezed[x] = lowpass(csi_matrix_squeezed[x], 10, 100, 5)
  csi_matrix_squeezed[x] = hampel(csi_matrix_squeezed[x], 10, 3)
  csi_matrix_squeezed[x] = running_mean(csi_matrix_squeezed[x], 10)

# Save
with open(p_paath, 'wb') as f:
    pickle.dump(csi_matrix_squeezed[x], f)

# 저장된 피클 파일 로드 및 확인
with open(p_paath, 'rb') as f:
    loaded_csi_matrix_element = pickle.load(f)

BatchGraph.plot_heatmap(loaded_csi_matrix_element, loaded_csi_matrix_element.timestamps)