import os
import random
import pickle
import pandas as pd
import numpy as np
from numpy import inf
from convert_csv import generate_csv
from sklearn.model_selection import train_test_split

# 클래스(레이블) 수 / 클래스(레이블) 마다 파일이 하나씩 있음
class_num = 9
# 각 파일 당 최대 데이터 수
max_per_file = 1100
# 파일 경로 패턴 설정



pcap_base_path = "/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/aa_00{}.pcap" # 수집한 pcap 파일의 경로 입력


""" 아래 경로들은 파일명 변경 불필요 """
csv_base_path = "/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Pcap_to_CSV_{}.csv" 
pickle_base_path = "/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/CSV_to_Pickle_{}.pickle"
p_pickle_base_path = "/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/CSI_Processing_{}.pickle"
base_dir = os.path.dirname(pcap_base_path)

def csv_to_pickle(csv_path: str, pickle_path: str):
    df = pd.read_csv(csv_path)
    with open(pickle_path, 'wb') as f:
        pickle.dump(df, f)
    # print("Pickle file written to: {}".format(pickle_path))

# Class_b_1.pcap부터 Class_b_4.pcap까지 반복
for i in range(1, class_num+1):
    pcap_file_path = pcap_base_path.format(i)
    csv_file_path = csv_base_path.format(i)
    pickle_file_path = pickle_base_path.format(i)
    
    # pcap 파일을 csv 파일로 변환
    generate_csv(pcap_file_path, csv_file_path)
    
    # csv 파일을 pickle 파일로 변환
    csv_to_pickle(csv_file_path, pickle_file_path)



## pickle 파일을 읽어와서 데이터를 정제하고 분할하여 저장
for i in range(1, class_num+1):
    save_path = pickle_base_path.format(i)
    with open(save_path,'rb') as f:
        original_data = pickle.load(f)
        original_data[original_data == -np.inf] = 0

        # 정제되어있지 않은 각 행을 리스트로 변환하여 2차원 리스트 생성
        two_dimensional_list = original_data.values.tolist()

        # 2차원 리스트를 NumPy 배열로 변환
        np_array = np.array(two_dimensional_list)

        # 800개까지만 로드
        np_array = np_array[:max_per_file]
        
        # np_array에서 -inf 값이 있는지 확인
        has_negative_inf = np.isinf(np_array) & (np_array == -np.inf)
        if np.any(has_negative_inf):
            print("np_array contains -inf values.")
        else:
            print("OK")

        # 1. 데이터 로드 및 분할
        X_training, X_test = train_test_split(np_array, test_size=0.2, random_state=42)
        
        # 2. 레이블 생성 및 추가
        label = i - 1
        print(f"레이블 as {label}")
        y_tra = [label for _ in range(len(X_training))]
        y_tst = [label for _ in range(len(X_test))]
        
        # 3. 데이터 저장
        data_to_save = [X_training, y_tra, X_test, y_tst]
        
        # 새로운 파일명 생성
        file_path = p_pickle_base_path.format(i)
        with open(file_path, 'wb') as f:
            pickle.dump(data_to_save, f)

        # print(f"{pickle_file_path} 파일이 변환되어 {file_path}로 저장되었습니다.")





## 피클 파일 병합

# 파일 이름 추출 및 변경
original_name = os.path.basename(pcap_base_path).format('')
# 파일 이름 추출 및 확장자 제거
file_name_without_ext = os.path.splitext(os.path.basename(pcap_base_path))[0]
# 새로운 확장자 추가
output_file_name = f"Processed_{file_name_without_ext}.pickle"
output_file_path = os.path.join(base_dir, output_file_name)  # 병합된 데이터를 저장할 파일 경로

# 병합된 데이터를 저장할 리스트 초기화
merged_data = [[], [], [], []]

# 피클 파일의 데이터를 읽어와 병합
for i in range(1, class_num+1):
    save_path = p_pickle_base_path.format(i)
    with open(save_path, 'rb') as f:
        data = pickle.load(f)
        merged_data[0].extend(data[0])  # 1열 (400, 256) 형태의 2차원 리스트
        merged_data[1].extend(data[1])  # 2열 (400,) 형태의 1차원 리스트
        merged_data[2].extend(data[2])  # 3열 (100, 256) 형태의 2차원 리스트
        merged_data[3].extend(data[3])  # 4열 (100,) 형태의 1차원 리스트

## 데이터 섞기
# 쌍을 만들기 위한 리스트 초기화
pairs_12 = []
pairs_34 = []

# 반복문을 사용하여 쌍 생성
for j in range(len(merged_data[0])):
    pairs_12.append((merged_data[0][j], merged_data[1][j]))
for k in range(len(merged_data[2])):
    pairs_34.append((merged_data[2][k], merged_data[3][k]))

# 순서 랜덤으로 섞기 / 2번 섞음
random.shuffle(pairs_12)
random.shuffle(pairs_12)
random.shuffle(pairs_34)
random.shuffle(pairs_34)

# 섞은 거 다시 리스트로 만들기
shuffled_data = [[], [], [], []]
for j in range(len(pairs_12)):
    shuffled_data[0].append(pairs_12[j][0])
    shuffled_data[1].append(pairs_12[j][1])
for k in range(len(pairs_34)):
    shuffled_data[2].append(pairs_34[k][0])
    shuffled_data[3].append(pairs_34[k][1])

shuffled_data[0] = np.array(shuffled_data[0])
shuffled_data[1] = np.array(shuffled_data[1], dtype=np.int32)
shuffled_data[2] = np.array(shuffled_data[2])
shuffled_data[3] = np.array(shuffled_data[3], dtype=np.int32)

shuffled_data = np.array(shuffled_data, dtype=object)

# 병합된 데이터를 하나의 피클 파일로 저장
with open(output_file_path.format(0), 'wb') as f:
    pickle.dump(shuffled_data, f)
print('\n 피클 파일 병합이 완료되었습니다.')

    
    
# 변환결과 확인
with open(output_file_path.format(0),'rb') as f:
    data = pickle.load(f)
    data_array = np.array(data, dtype=object)
    print("-------------------")
    print(f"type(data_array[0]) = {type(data_array[0])}")
    print(f"np.shape = {np.shape(data_array)} :: {np.shape(data_array[0])} // {np.shape(data_array[1])} // {np.shape(data_array[2])} // {np.shape(data_array[3])}\n")
#     print("-------------------")
#     print(data_array[0][100][-1])
#     print(data_array[0][100][255])
#     # print(data_array[0][100])
#     print(data_array[1])


for i in range(1, class_num+1):
    # os.remove(csv_base_path.format(i))
    os.remove(pickle_base_path.format(i))
    os.remove(p_pickle_base_path.format(i))