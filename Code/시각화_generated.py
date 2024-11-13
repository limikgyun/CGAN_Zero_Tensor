import matplotlib.pyplot as plt
from keras.models import load_model
from utils import *
from main import generate_latent_points

# 모델 로드
# generator = load_model('/home/limikgyun/CGAN_Zero_Tensor/Storage/모델/Base간-Processed_x_00-g-3000샘플-100에폭.h5')
generator = load_model('/home/limikgyun/CGAN_Zero_Tensor/Storage/모델/c to b/Trans-Base간-240924-1408-g-1000샘플-80에폭-86.h5')

# 잠재 공간 벡터의 차원 (예시)
latent_dim = 100

# 10번 반복하여 서로 다른 파일 이름의 그림을 생성
for i in range(10):
    # 잠재 공간 벡터 생성
    n_CSI = 1  # 각 반복에서 1개의 CSI를 생성
    z_inputs = generate_latent_points(latent_dim, n_CSI)

    # 데이터 생성
    generated_samples = generator.predict(z_inputs)
    generated_samples = generated_samples.reshape(-1, 256)
    generated_samples = single_minmaxscale(generated_samples, scale_range=(0, 1))

    # 데이터 플롯 및 저장
    plt.figure()
    for x_g in generated_samples:
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42        
        plt.title('Generated Data (fake)', fontsize=20)
        plt.xlabel('CSI Index', fontsize=18)
        plt.ylabel('CSI Amplitude (Normalized)', fontsize=18)
        plt.axis([0, 256, 0, 1])
        plt.grid(True)
        plt.plot(x_g)
    plt.savefig(f'/home/limikgyun/CGAN_Zero_Tensor/Storage/시각화/CSI의 형상_{i+1}.png', dpi=600)
    plt.close()

# import time
# import numpy as np
# import pickle
# from numpy.random import seed
# from matplotlib import pyplot as plt
# from tensorflow.keras.models import load_model
# from utils import *
# from main import *

# def generate_data_by_label(generator, latent_dim, n_samples, label_value):
#     latent_vectors = np.random.randn(n_samples, latent_dim)
#     labels = np.full((n_samples, 1), label_value)
#     generated_data = generator.predict([latent_vectors, labels])
#     return generated_data

# def exp1(label):
#     generator = load_model('/home/limikgyun/CGAN_Zero_Tensor/Storage/모델/C간-a-240924-0132-g-3000샘플-200에폭-100.h5')
    
#     # use generate_data_by_label to produce random samples
#     latent_dim = 100
#     n_samples = 70
#     generated_samples = generate_data_by_label(generator, latent_dim, n_samples, label)

#     generated_samples = generated_samples.reshape(-1, 256)
    
#     print(generated_samples.shape)
#     generated_samples = single_minmaxscale(generated_samples, scale_range=(0, 1))

#     plt.figure()
#     for x_g in generated_samples:
#         plt.rcParams['pdf.fonttype'] = 42
#         plt.rcParams['ps.fonttype'] = 42
        
#         plt.title('Generated CSI', fontsize=20)
#         plt.xlabel('CSI Subcarrier Index', fontsize=16)
#         plt.ylabel('CSI Amplitude (Normalized)', fontsize=16)       
#         plt.axis([0, 256, 0, 1])
#         plt.grid(True)
#         plt.plot(x_g)

#     plt.savefig('/home/limikgyun/CGAN_Zero_Tensor/Storage/시각화/Generated CSI%d.png' % label, dpi=600)
#     plt.close()

# if __name__ == '__main__':
#     seed(0)
#     exp1(1)
#     # exp1(1)
#     # exp1(2)