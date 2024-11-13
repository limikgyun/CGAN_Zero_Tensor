import os
import pickle
import pandas as pd
import numpy as np
from convert_csv import generate_csv, csv_to_pickle
from sklearn.model_selection import train_test_split

# 클래스(레이블) 수 / 클래스(레이블) 마다 파일이 하나씩 있음
class_num = 9

# 각 파일 당 최대 데이터 수
max_per_file = 500

# 파일 경로 패턴 설정
pcap_base_path = "/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/raw_case_2/cc_00{}.pcap"
output_pickle_path = "/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/cc_00{}.pickle"

## 아래 2개 파일명 변경 불필요
csv_base_path = "/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Pcap_to_CSV_00{}.csv"
pickle_base_path = "/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/CSV_to_Pickle_00{}.pickle"

# 9번 반복
for group in range(1, 10):
    extended_data = []
    # 각 그룹 내의 5개 파일 처리
    for i in range(1, 6):
        file_index = group * 10 + i
        pcap_file_path = pcap_base_path.format(file_index)
        csv_file_path = csv_base_path.format(file_index)
        pickle_file_path = pickle_base_path.format(file_index)

        # pcap 파일을 csv 파일로 변환
        generate_csv(pcap_file_path, csv_file_path)

        # csv 파일을 pickle 파일로 변환
        csv_to_pickle(csv_file_path, pickle_file_path)

        # pickle 파일을 읽어와서 데이터를 리스트에 추가
        with open(pickle_file_path, 'rb') as f:
            original_data = pickle.load(f)
            original_data[original_data == -np.inf] = 0 # -inf 값을 0으로 변환

            two_dimensional_list = original_data.values.tolist()
            np_array = np.array(two_dimensional_list)

            # 모든 값이 0인 튜플의 개수 계산 및 출력
            zero_tuples_count = np.sum(np.all(np_array == 0, axis=1))
            print(f"제거될 튜플 갯수 : {zero_tuples_count}")

            # 모든 값이 0인 튜플 제거
            np_array = np_array[~np.all(np_array == 0, axis=1)]

            # 최대 데이터 수 제한
            np_array = np_array[:max_per_file]
            extended_data.extend(np_array)

    np.random.shuffle(extended_data)
    # 추출된 데이터를 새로운 pickle 파일로 저장
    with open(output_pickle_path.format(group), 'wb') as f:
        pickle.dump(extended_data, f)

    print(f"Group {group}: Saved {len(extended_data)} records to {output_pickle_path}")



# 생성된 CSV 파일과 Pickle 파일 삭제
for group in range(1, 10):
    for i in range(1, 6):
        file_index = group * 10 + i
        csv_file_path = csv_base_path.format(file_index)
        pickle_file_path = pickle_base_path.format(file_index)
        
        if os.path.exists(csv_file_path):
            os.remove(csv_file_path)
            print(f"Deleted {csv_file_path}")
        
        if os.path.exists(pickle_file_path):
            os.remove(pickle_file_path)
            print(f"Deleted {pickle_file_path}")


## pickle 파일을 읽어와서 데이터를 정제하고 분할하여 저장
def pre_processing_data(pickle_path, class_num):
    pickle_data = []
    for i in range(1, class_num+1):
        with open(output_pickle_path.format(i),'rb') as f:
            original_data = pickle.load(f)
            original_data[original_data == -np.inf] = 0

            # 1. 데이터 로드 및 분할
            X_training, X_test = train_test_split(np_array, test_size=0.2, random_state=42)
            
            # 2. 레이블 생성 및 추가
            label = i # 0값을 최대한 안쓰려고함
            print(f"레이블 as {label}")
            y_tra = [label for _ in range(len(X_training))]
            y_tst = [label for _ in range(len(X_test))]
            
            # 3. 데이터 저장
            data_to_save = [X_training, y_tra, X_test, y_tst]
        
            # print(f"{pickle_file_path} 파일이 변환되어 {file_path}로 저장되었습니다.")


    # 병합된 데이터를 저장할 리스트 초기화
    merged_data = [[], [], [], []]

    # 피클 파일의 데이터를 읽어와 병합
    for i in range(1, class_num+1):
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
