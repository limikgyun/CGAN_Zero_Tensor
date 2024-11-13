import random
import numpy as np
import os
from tensorflow.keras.models import load_model
import pickle

def generate_data_by_label(generator, latent_dim, n_samples, label_value):
    latent_vectors = np.random.randn(n_samples, latent_dim)
    labels = np.full((n_samples, 1), label_value)
    generated_data = generator.predict([latent_vectors, labels])
    return generated_data

def generate_csi_by_CGAN(pickle_file_path, generator_model_path):
    data_file_name_with_ext = os.path.basename(pickle_file_path)
    data_name, _ = os.path.splitext(data_file_name_with_ext)
        
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

if __name__ == '__main__':
    generate_csi_by_CGAN(
'/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Selected_bb_[1, 2, 3].pickle',
'/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Trans-C간-Removed_[1]_Selected_bb_[1, 2, 3]-g-4000샘플-8에폭-64.h5')
    generate_csi_by_CGAN(
'/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Selected_bb_[1, 4, 7].pickle',
'/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Trans-C간-Removed_[1]_Selected_bb_[1, 4, 7]-g-4000샘플-8에폭-63.h5')
    generate_csi_by_CGAN(
'/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Selected_bb_[2, 5, 8].pickle',
'/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Trans-C간-Removed_[1]_Selected_bb_[2, 5, 8]-g-4000샘플-8에폭-62.h5')
    generate_csi_by_CGAN(
'/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Selected_bb_[3, 6, 9].pickle',
'/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Trans-C간-Removed_[1]_Selected_bb_[3, 6, 9]-g-4000샘플-8에폭-59.h5')
    generate_csi_by_CGAN(
'/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Selected_bb_[4, 5, 6].pickle',
'/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Trans-C간-Removed_[1]_Selected_bb_[4, 5, 6]-g-4000샘플-8에폭-58.h5')
    generate_csi_by_CGAN(
'/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Selected_bb_[7, 8, 9].pickle',
'/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Trans-C간-Removed_[1]_Selected_bb_[7, 8, 9]-g-4000샘플-8에폭-51.h5')
    generate_csi_by_CGAN(
'/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Selected_cc_[1, 2, 3].pickle',
'/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Trans-C간-Removed_[1]_Selected_cc_[1, 2, 3]-g-4000샘플-8에폭-68.h5')
    generate_csi_by_CGAN(
'/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Selected_cc_[1, 4, 7].pickle',
'/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Trans-C간-Removed_[1]_Selected_cc_[1, 4, 7]-g-4000샘플-8에폭-57.h5')
    generate_csi_by_CGAN(
'/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Selected_cc_[2, 5, 8].pickle',
'/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Trans-C간-Removed_[1]_Selected_cc_[2, 5, 8]-g-4000샘플-8에폭-52.h5')
    generate_csi_by_CGAN(
'/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Selected_cc_[3, 6, 9].pickle',
'/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Trans-C간-Removed_[1]_Selected_cc_[3, 6, 9]-g-4000샘플-8에폭-69.h5')
    generate_csi_by_CGAN(
'/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Selected_cc_[4, 5, 6].pickle',
'/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Trans-C간-Removed_[1]_Selected_cc_[4, 5, 6]-g-4000샘플-8에폭-55.h5')
    generate_csi_by_CGAN(
'/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Selected_cc_[7, 8, 9].pickle',
'/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Trans-C간-Removed_[1]_Selected_cc_[7, 8, 9]-g-4000샘플-8에폭-64.h5')