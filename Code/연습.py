# from tensorflow.python.client import device_lib
# device_lib.list_local_devices()

import tensorflow as tf

print("TensorFlow Version: ", tf.__version__)
print("CUDA Version: ", tf.sysconfig.get_build_info()["cuda_version"])
print("cuDNN Version: ", tf.sysconfig.get_build_info()["cudnn_version"])

# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# import numpy as np

# # 간단한 모델 정의
# model = Sequential([
#     Dense(10, activation='relu', input_shape=(32,)),
#     Dense(1, activation='sigmoid')
# ])

# model.compile(optimizer='adam', loss='binary_crossentropy')

# # 가짜 데이터 생성
# x = np.random.random((32, 32))
# y = np.random.randint(2, size=(32, 1))

# # 한 배치 단위로 훈련
# loss = model.train_on_batch(x, y)
# print('손실 값:', loss)                                  