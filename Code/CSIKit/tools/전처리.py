import os
import pickle
import pandas as pd
import numpy as np
import random
from tensorflow.keras.models import load_model
from convert_csv import generate_csv, csv_to_pickle
from sklearn.model_selection import train_test_split


def pre_process_pcap(pcap_base_path, class_num=9, max_per_file=500):
    # 파일 경로 패턴 설정
    # pcap_base_path = "/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/raw_case_2/cc_00{}.pcap"
    output_base_dir = "/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/"
    csv_base_path = "/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Pcap_to_CSV_00{}.csv"
    pickle_base_path = "/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/CSV_to_Pickle_00{}.pickle"

    # 9번 반복
    for group in range(1, class_num + 1):
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
                print(f"제거될 튜플 갯수 : {zero_tuples_count} \n")

                # 모든 값이 0인 튜플 제거
                np_array = np_array[~np.all(np_array == 0, axis=1)]

                # 최대 데이터 수 제한
                np_array = np_array[:max_per_file]
                extended_data.extend(np_array)

        np.random.shuffle(extended_data)
        
        # 추출된 데이터를 새로운 pickle 파일로 저장
        # pcap 파일명을 가져와서 확장자를 .pickle로 변경
        pcap_filename = os.path.basename(pcap_base_path.format(group))
        pickle_filename = os.path.splitext(pcap_filename)[0] + '.pickle'
        output_pickle_path = os.path.join(output_base_dir, pickle_filename)

        with open(output_pickle_path.format(group), 'wb') as f:
            pickle.dump(extended_data, f)

        print(f"Group {group}: Saved {len(extended_data)} records to {output_pickle_path}")

    # 생성된 CSV 파일과 Pickle 파일 삭제
    for group in range(1, class_num + 1):
        for i in range(1, 6):
            file_index = group * 10 + i
            csv_file_path = csv_base_path.format(file_index)
            pickle_file_path = pickle_base_path.format(file_index)
            
            if os.path.exists(csv_file_path):
                os.remove(csv_file_path)            
            if os.path.exists(pickle_file_path):
                os.remove(pickle_file_path)

def pre_process_pcap_only1(pcap_base_path, class_num=3, max_per_file=500):
    # 파일 경로 패턴 설정
    output_base_dir = "/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/"
    csv_base_path = "/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Pcap_to_CSV_00{}.csv"
    pickle_base_path = "/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/CSV_to_Pickle_00{}.pickle"

    # 9번 반복
    
    for i in range(1, class_num+1):
        extended_data = []
        file_index = i
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
            print(f"제거될 튜플 갯수 : {zero_tuples_count} \n")

            # 모든 값이 0인 튜플 제거
            np_array = np_array[~np.all(np_array == 0, axis=1)]
            np.random.shuffle(np_array)

            # 최대 데이터 수 제한
            np_array = np_array[:max_per_file]
            extended_data.extend(np_array)        
        
        # 추출된 데이터를 새로운 pickle 파일로 저장
        # pcap 파일명을 가져와서 확장자를 .pickle로 변경
        pcap_filename = os.path.basename(pcap_base_path.format(i))
        pickle_filename = os.path.splitext(pcap_filename)[0] + '.pickle'
        output_pickle_path = os.path.join(output_base_dir, pickle_filename)

        with open(output_pickle_path.format(i), 'wb') as f:
            pickle.dump(extended_data, f)

        print(f"Group {i}: Saved {len(extended_data)} records to {output_pickle_path}")

    # 생성된 CSV 파일과 Pickle 파일 삭제
    for i in range(1, class_num + 1):
        file_index = i
        csv_file_path = csv_base_path.format(i)
        pickle_file_path = pickle_base_path.format(i)
        
        if os.path.exists(csv_file_path):
            os.remove(csv_file_path)            
        if os.path.exists(pickle_file_path):
            os.remove(pickle_file_path)

## pickle 파일을 읽어와서 데이터를 정제하고 분할하여 저장
def pre_process_pickle(pickle_path, class_num):

    # output_file_path 설정
    base_pickle_path = pickle_path.format(0)
    base_filename = os.path.basename(base_pickle_path)
    processed_filename = "Processed_" + base_filename
    output_file_path = os.path.join(os.path.dirname(base_pickle_path), processed_filename)

    merged_data = [[], [], [], []]
    # for i in range(1, class_num+1):
    for i in range(1, class_num+1):
        with open(pickle_path.format(i),'rb') as f:
            print(f"{pickle_path.format(i)} 파일을 읽어옵니다.")
            original_data = pickle.load(f)
            np_array = np.array(original_data, dtype=object)

            # 1. 데이터 로드 및 분할
            X_training, X_test = train_test_split(np_array, test_size=0.2, random_state=42)
            print(f"X_training.shape = {X_training.shape} // X_test.shape = {X_test.shape}")
            # 2. 레이블 생성 및 추가
            label = i-1 # 0값을 최대한 안쓰려고함 // 1~9 까지 가능 // 오류 떠서 그냥 0부터 시작하게 함
            print(f"레이블 as {label}")
            y_tra = [label for _ in range(len(X_training))]
            y_tst = [label for _ in range(len(X_test))]
                        
            # 3. 데이터 저장
            merged_data[0].extend(X_training)
            print(f"merged_data[0].shape = {np.shape(merged_data[0])}")
            merged_data[1].extend(y_tra)
            merged_data[2].extend(X_test)
            print(f"merged_data[2].shape = {np.shape(merged_data[2])}")
            merged_data[3].extend(y_tst)

    # numpy 배열로 초기화
    pairs_12 = np.empty((0, 2), dtype=object)
    pairs_34 = np.empty((0, 2), dtype=object)

    # 반복문을 사용하여 쌍 생성
    for i in range(len(merged_data[0])):
        pairs_12 = np.vstack((pairs_12, [merged_data[0][i], merged_data[1][i]]))
    for i in range(len(merged_data[2])):
        pairs_34 = np.vstack((pairs_34, [merged_data[2][i], merged_data[3][i]]))

    # 순서 랜덤으로 섞기 / 2번 섞음
    np.random.shuffle(pairs_12)
    np.random.shuffle(pairs_34)
    np.random.shuffle(pairs_12)
    np.random.shuffle(pairs_34)

    # 섞은 거 다시 리스트로 만들기
    shuffled_data = [[], [], [], []]

    for i in range(len(pairs_12)):
        shuffled_data[0].append(pairs_12[i][0])
        shuffled_data[1].append(pairs_12[i][1])
    for i in range(len(pairs_34)):
        shuffled_data[2].append(pairs_34[i][0])
        shuffled_data[3].append(pairs_34[i][1])

    shuffled_data[0] = np.array(shuffled_data[0])
    shuffled_data[1] = np.array(shuffled_data[1], dtype=np.int32)
    shuffled_data[2] = np.array(shuffled_data[2])
    shuffled_data[3] = np.array(shuffled_data[3], dtype=np.int32)
    
    shuffled_data = np.array(shuffled_data, dtype=object)
    # 병합된 데이터를 하나의 피클 파일로 저장
    with open(output_file_path, 'wb') as f:
        pickle.dump(shuffled_data, f)
    print('\n 피클 파일 병합이 완료되었습니다.')

        
        
    # 변환결과 확인
    with open(output_file_path,'rb') as f:
        data = pickle.load(f)
        data_array = np.array(data, dtype=object)
        print("-------------------")
        print(f"type(data_array[0]) = {type(data_array[0])}")
        print(f"np.shape(data_array[0][0]) = {np.shape(data_array[0][0])}")
        print(f"np.shape = {np.shape(data_array)} :: {np.shape(data_array[0])} // {np.shape(data_array[1])} // {np.shape(data_array[2])} // {np.shape(data_array[3])}\n")
        print(data_array[1])
        print("-------------------")


def select_pre_process_pickle(selected_coord_list, pickle_path, class_num):#training 데이터에 특정 레이블 값을 제거
    # output_file_path 설정
    base_pickle_path = pickle_path.format(0)
    base_filename = os.path.basename(base_pickle_path)
    processed_filename = "Selected_" + str(selected_coord_list) + "_" + base_filename + '.pickle'
    output_file_path = os.path.join(os.path.dirname(base_pickle_path), processed_filename)

    merged_data = [[], [], [], []]
    # for i in range(1, class_num+1):

    #임시로 label_value 추가 / 레이블을 파일명에 무관하게 0, 1, 2로 설정하기 위해
    label_value = 0

    for i in selected_coord_list:#선택된 위치만 더해짐
        with open(pickle_path.format(i),'rb') as f:
            print(f"{pickle_path.format(i)} 파일을 읽어옵니다.")
            original_data = pickle.load(f)
            np_array = np.array(original_data, dtype=object)

            # 1. 데이터 로드 및 분할
            X_training, X_test = train_test_split(np_array, test_size=0.2, random_state=42)
            print(f"X_training.shape = {X_training.shape} // X_test.shape = {X_test.shape}")
            # 2. 레이블 생성 및 추가

            ## 임시로 주석 처리 / 레이블을 파일명에 무관하게 0, 1, 2로 설정하기 위해
            # label = i-1
            # print(f"레이블 as {label}")
            # y_tra = [label for _ in range(len(X_training))]
            # y_tst = [label for _ in range(len(X_test))]

            print(f"레이블 as {label_value}")
            y_tra = [label_value for _ in range(len(X_training))]
            y_tst = [label_value for _ in range(len(X_test))]
            label_value += 1
                        
            # 3. 데이터 저장
            merged_data[0].extend(X_training)
            print(f"merged_data[0].shape = {np.shape(merged_data[0])}")
            merged_data[1].extend(y_tra)
            merged_data[2].extend(X_test)
            print(f"merged_data[2].shape = {np.shape(merged_data[2])}")
            merged_data[3].extend(y_tst)

    # numpy 배열로 초기화
    pairs_12 = np.empty((0, 2), dtype=object)
    pairs_34 = np.empty((0, 2), dtype=object)

    # 반복문을 사용하여 쌍 생성
    for i in range(len(merged_data[0])):
        pairs_12 = np.vstack((pairs_12, [merged_data[0][i], merged_data[1][i]]))
    for i in range(len(merged_data[2])):
        pairs_34 = np.vstack((pairs_34, [merged_data[2][i], merged_data[3][i]]))

    # 순서 랜덤으로 섞기 / 2번 섞음
    np.random.shuffle(pairs_12)
    np.random.shuffle(pairs_34)
    np.random.shuffle(pairs_12)
    np.random.shuffle(pairs_34)

    # 섞은 거 다시 리스트로 만들기
    shuffled_data = [[], [], [], []]

    for i in range(len(pairs_12)):
        shuffled_data[0].append(pairs_12[i][0])
        shuffled_data[1].append(pairs_12[i][1])
    for i in range(len(pairs_34)):
        shuffled_data[2].append(pairs_34[i][0])
        shuffled_data[3].append(pairs_34[i][1])

    shuffled_data[0] = np.array(shuffled_data[0])
    shuffled_data[1] = np.array(shuffled_data[1], dtype=np.int32)
    shuffled_data[2] = np.array(shuffled_data[2])
    shuffled_data[3] = np.array(shuffled_data[3], dtype=np.int32)
    
    shuffled_data = np.array(shuffled_data, dtype=object)
    # 병합된 데이터를 하나의 피클 파일로 저장
    with open(output_file_path, 'wb') as f:
        pickle.dump(shuffled_data, f)
    print('\n 피클 파일 병합이 완료되었습니다.')

        
        
    # 변환결과 확인
    with open(output_file_path,'rb') as f:
        data = pickle.load(f)
        data_array = np.array(data, dtype=object)
        print("-------------------")
        print(f"type(data_array[0]) = {type(data_array[0])}")
        print(f"np.shape(data_array[0][0]) = {np.shape(data_array[0][0])}")
        print(f"np.shape = {np.shape(data_array)} :: {np.shape(data_array[0])} // {np.shape(data_array[1])} // {np.shape(data_array[2])} // {np.shape(data_array[3])}\n")
        print(data_array[1])
        print(data_array[3])
        print("-------------------")

def remove_specific_training_data(pickle_file_path, target_label):
    # pickle 파일 로드
    with open(pickle_file_path, 'rb') as file:
        ori_data = pickle.load(file)

    # 특정 레이블 값을 가진 idx 값을 기록
    idx_to_remove = []

    for idx, label in enumerate(ori_data[1]):
        if label == target_label:
            idx_to_remove.append(idx)

    # 역순으로 정렬하여 삭제 (인덱스 오류 방지)
    idx_to_remove.sort(reverse=True)

    print(f"Original data: {len(ori_data[0])} samples")
    print(f"Original labels: {len(ori_data[1])} labels")

    # idx 값을 사용하여 데이터 삭제
    for idx in idx_to_remove:
        ori_data[0] = np.delete(ori_data[0], idx, axis=0)
        ori_data[1] = np.delete(ori_data[1], idx, axis=0)

    # 결과 확인
    print(f"Remaining data: {len(ori_data[0])} samples")
    print(f"Remaining labels: {len(ori_data[1])} labels")

    print(ori_data[1])
    print(ori_data[3])

    # 저장 경로 생성
    dir_name, base_name = os.path.split(pickle_file_path)
    base_name = base_name.replace('Processed_', '')
    new_file_name = f"Removed_[{target_label}]_{base_name}"
    new_pickle_file_path = os.path.join(dir_name, new_file_name)

    # 결과를 pickle 파일로 저장
    with open(new_pickle_file_path, 'wb') as file:
        pickle.dump(ori_data, file)

    print(f"Data saved to {new_pickle_file_path}")

def generate_data_by_label(generator, latent_dim, n_samples, label_value):
    latent_vectors = np.random.randn(n_samples, latent_dim)
    labels = np.full((n_samples, 1), label_value)
    generated_data = generator.predict([latent_vectors, labels])
    return generated_data
def generate_csi_by_CGAN(pickle_file_path, generator_model_path):
    data_file_name_with_ext = os.path.basename(pickle_file_path)
    data_name, _ = os.path.splitext(data_file_name_with_ext)
    data_name = data_name.replace('Removed_[1]_Processed_', '')
    data_name = data_name.replace('Removed_[1]_', '')

    # 저장된 generator 모델 로드
    generator = load_model(generator_model_path)

    # 전처리 완료된 Pickle 파일 경로
    processed_file_path = pickle_file_path

    # shuffled_data를 피클 파일로 저장
    shuffled_output_file_path = '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Generated_[1]_{}.pickle'.format(data_name)  # save 저장경로

    # 잠재 공간 벡터 생성 (latent_dim은 모델 정의 시 사용된 값과 동일해야 함)
    latent_dim = 100
    n_samples = 1000  # 각 레이블에 대해 생성할 샘플 수

    pairs = []
    # 레이블 0, 2, 4에 대한 데이터 생성
    labels = [1]
    for label_value in labels:
        generated_data = generate_data_by_label(generator, latent_dim, n_samples, label_value)
        generated_data = generated_data.reshape(n_samples, -1)    
        for i in range(len(generated_data)):
            pairs.append((generated_data[i], label_value))

    # Pickle 파일에서 데이터 로드
    with open(processed_file_path, 'rb') as f:
        loaded_data = pickle.load(f)

    # 레이블 0, 2, 4, 6, 8에 대한 데이터 추가
    additional_labels = [0, 2]
    for label_value in additional_labels:
        count = 0
        for i in range(len(loaded_data[1])):
            if loaded_data[1][i] == label_value:
                pairs.append((loaded_data[0][i], label_value))
                count += 1
                if count >= n_samples:
                    break

    # 평가데이터는 기존거 그대로 사용
    original_test_pairs = []
    for j in range(len(loaded_data[2])):
        original_test_pairs.append((loaded_data[2][j], loaded_data[3][j]))

    random.shuffle(pairs)
    random.shuffle(pairs)

    shuffled_data = [[], [], [], []]

    for i in range(len(pairs)):
        shuffled_data[0].append(pairs[i][0])
        shuffled_data[1].append(pairs[i][1])
    for j in range(len(original_test_pairs)):
        shuffled_data[2].append(original_test_pairs[j][0])
        shuffled_data[3].append(original_test_pairs[j][1])
        
    shuffled_data[0] = np.array(shuffled_data[0])
    shuffled_data[1] = np.array(shuffled_data[1], dtype=np.int32)
    shuffled_data[2] = np.array(shuffled_data[2])
    shuffled_data[3] = np.array(shuffled_data[3], dtype=np.int32)

    shuffled_data = np.array(shuffled_data, dtype=object)

    with open(shuffled_output_file_path, 'wb') as f:
        pickle.dump(shuffled_data, f)

if __name__ == "__main__":
    ## 240414 재실험 전처리 순서 for x
    # pre_process_pcap_only1("/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/재실험/x_0{}.pcap", 3, 500)
    # pre_process_pcap_only1("/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/재실험/x++_0{}.pcap", 3, 500)
    # pre_process_pickle("/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/x_0{}.pickle", 3)
    # pre_process_pickle("/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/x++_0{}.pickle", 3)
    # for path in ["/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Processed_x_00.pickle",
    #              "/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Processed_x++_00.pickle"]:
    #     remove_specific_training_data(path, 1)
    # generate_csi_by_CGAN('/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Removed_[1]_x++_00.pickle',
    #                      '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Trans-C간-Removed_[1]_Processed_x++_00-g-1000샘플-9에폭.h5')
    # generate_csi_by_CGAN('/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Removed_[1]_x_00.pickle',
    #                      '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Trans-C간-Removed_[1]_x_00-g-1000샘플-9에폭.h5')
    # 240414 재실험 전처리 순서 for z
    # pre_process_pcap_only1("/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/재실험/z_0{}.pcap", 3, 500)
    # pre_process_pcap_only1("/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/재실험/z++_0{}.pcap", 3, 500)
    # pre_process_pickle("/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/z_0{}.pickle", 3)
    # pre_process_pickle("/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/z++_0{}.pickle", 3)
    # for path in ["/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Processed_z_00.pickle",
    #              "/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Processed_z++_00.pickle"]:
    #     remove_specific_training_data(path, 1)
    generate_csi_by_CGAN('/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Removed_[1]_z++_00.pickle',
                         '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Trans-C간-Removed_[1]_Processed_z++_00-g-1000샘플-9에폭.h5')
    generate_csi_by_CGAN('/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Removed_[1]_z_00.pickle',
                         '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Trans-C간-Removed_[1]_z_00-g-1000샘플-9에폭.h5')









    # class_num = 9
    # max_per_file = 500
    # pre_process_pcap("/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/raw_case_2/cc_00{}.pcap", class_num, max_per_file)
    # pre_process_pickle("/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/aa_00{}.pickle", class_num)
    # pre_process_pickle("/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/bb_00{}.pickle", class_num)
    # pre_process_pickle("/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/cc_00{}.pickle", class_num)

    # select_pre_process_pickle([1, 2, 3], "/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/cc_00{}.pickle", class_num)
    # select_pre_process_pickle([4, 5, 6], "/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/cc_00{}.pickle", class_num)
    # select_pre_process_pickle([7, 8, 9], "/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/cc_00{}.pickle", class_num)
    # select_pre_process_pickle([1, 4, 7], "/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/cc_00{}.pickle", class_num)
    # select_pre_process_pickle([2, 5, 8], "/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/cc_00{}.pickle", class_num)
    # select_pre_process_pickle([3, 6, 9], "/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/cc_00{}.pickle", class_num)

#     remove_specific_training_data("/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Selected_bb_[1, 2, 3].pickle", 1)
#     remove_specific_training_data("/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Selected_bb_[1, 4, 7].pickle", 1)
#     remove_specific_training_data("/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Selected_bb_[2, 5, 8].pickle", 1)
#     remove_specific_training_data("/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Selected_bb_[3, 6, 9].pickle", 1)
#     remove_specific_training_data("/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Selected_bb_[4, 5, 6].pickle", 1)
#     remove_specific_training_data("/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Selected_bb_[7, 8, 9].pickle", 1)
#     remove_specific_training_data("/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Selected_cc_[1, 2, 3].pickle", 1)
#     remove_specific_training_data("/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Selected_cc_[1, 4, 7].pickle", 1)
#     remove_specific_training_data("/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Selected_cc_[2, 5, 8].pickle", 1)
#     remove_specific_training_data("/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Selected_cc_[3, 6, 9].pickle", 1)
#     remove_specific_training_data("/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Selected_cc_[4, 5, 6].pickle", 1)
#     remove_specific_training_data("/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Selected_cc_[7, 8, 9].pickle", 1)


#     generate_csi_by_CGAN(
# '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Selected_bb_[1, 2, 3].pickle',
# '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Trans-C간-Removed_[1]_Selected_bb_[1, 2, 3]-g-4000샘플-8에폭-64.h5')
#     generate_csi_by_CGAN(
# '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Selected_bb_[1, 4, 7].pickle',
# '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Trans-C간-Removed_[1]_Selected_bb_[1, 4, 7]-g-4000샘플-8에폭-63.h5')
#     generate_csi_by_CGAN(
# '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Selected_bb_[2, 5, 8].pickle',
# '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Trans-C간-Removed_[1]_Selected_bb_[2, 5, 8]-g-4000샘플-8에폭-62.h5')
#     generate_csi_by_CGAN(
# '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Selected_bb_[3, 6, 9].pickle',
# '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Trans-C간-Removed_[1]_Selected_bb_[3, 6, 9]-g-4000샘플-8에폭-59.h5')
#     generate_csi_by_CGAN(
# '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Selected_bb_[4, 5, 6].pickle',
# '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Trans-C간-Removed_[1]_Selected_bb_[4, 5, 6]-g-4000샘플-8에폭-58.h5')
#     generate_csi_by_CGAN(
# '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Selected_bb_[7, 8, 9].pickle',
# '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Trans-C간-Removed_[1]_Selected_bb_[7, 8, 9]-g-4000샘플-8에폭-51.h5')
#     generate_csi_by_CGAN(
# '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Selected_cc_[1, 2, 3].pickle',
# '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Trans-C간-Removed_[1]_Selected_cc_[1, 2, 3]-g-4000샘플-8에폭-68.h5')
#     generate_csi_by_CGAN(
# '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Selected_cc_[1, 4, 7].pickle',
# '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Trans-C간-Removed_[1]_Selected_cc_[1, 4, 7]-g-4000샘플-8에폭-57.h5')
#     generate_csi_by_CGAN(
# '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Selected_cc_[2, 5, 8].pickle',
# '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Trans-C간-Removed_[1]_Selected_cc_[2, 5, 8]-g-4000샘플-8에폭-52.h5')
#     generate_csi_by_CGAN(
# '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Selected_cc_[3, 6, 9].pickle',
# '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Trans-C간-Removed_[1]_Selected_cc_[3, 6, 9]-g-4000샘플-8에폭-69.h5')
#     generate_csi_by_CGAN(
# '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Selected_cc_[4, 5, 6].pickle',
# '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Trans-C간-Removed_[1]_Selected_cc_[4, 5, 6]-g-4000샘플-8에폭-55.h5')
#     generate_csi_by_CGAN(
# '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Selected_cc_[7, 8, 9].pickle',
# '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Trans-C간-Removed_[1]_Selected_cc_[7, 8, 9]-g-4000샘플-8에폭-64.h5')
    

#     generate_csi_by_CGAN(
# '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Selected_cc_[4, 5, 6].pickle',
# '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Trans-C간-Removed_[1]_Selected_cc_[4, 5, 6]-g-4000샘플-8에폭-55.h5')