import pickle
import numpy as np
from numpy import inf
import os

# with open('/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/case_2/CSV_to_Pickle_0093.pickle','rb') as f:
with open('/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Removed_a_0[0,2].pickle','rb') as f:
    data = pickle.load(f)
    data_array = np.array(data, dtype=object)
    
    # print(data_array[0])
    # print(data_array[1])
    print("-------------------")
    # print(data_array)
    print("\n")
    print("-------------------")
    print(type(data_array[0]))   
    print(np.shape(data_array))
    print("\n")
    print(np.shape(data_array[0]))
    print(np.shape(data_array[1]))
    print(np.shape(data_array[2]))
    print(np.shape(data_array[3]))
    print("-------------------")
    print(data_array[1][:100])
    # print(data_array)

"""

()

### CSI 정보 확인 코드
from get_info import display_info

# 파일 경로 패턴 설정
base_path = "C:/Users/MNET/Desktop/CSI_data/class_a_{}.pcap"

# class_a_1.pcap부터 class_a_4.pcap까지 반복
for i in range(1, 5):
    pcap_file_path = base_path.format(i)
    
    # 데이터의 정보를 출력
    print(f"\n Displaying info for {pcap_file_path}")
    display_info(pcap_file_path)

"""