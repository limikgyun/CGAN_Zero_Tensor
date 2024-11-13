# from tensorflow.python.client import device_lib
# import tensorflow as tf
# device_lib.list_local_devices()
# print("TensorFlow Version: ", tf.__version__)
# print("CUDA Version: ", tf.sysconfig.get_build_info()["cuda_version"])
# print("cuDNN Version: ", tf.sysconfig.get_build_info()["cudnn_version"])

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 모델 정의
model = Sequential([
    Dense(10, activation='relu', input_shape=(32,)),
    Dense(1, activation='sigmoid')
])

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 더미 데이터 생성
import numpy as np
x_train = np.random.random((100, 32))
y_train = np.random.randint(2, size=(100, 1))

# 배치 단위로 훈련
batch_size = 32
for i in range(0, len(x_train), batch_size):
    x_batch = x_train[i:i+batch_size]
    y_batch = y_train[i:i+batch_size]
    print(f'Batch {i//batch_size + 1} - x_batch shape: {x_batch.shape}, y_batch shape: {y_batch.shape}')
    loss, accuracy = model.train_on_batch(x_batch, y_batch)
    print(f'Batch {i//batch_size + 1} - Loss: {loss}, Accuracy: {accuracy}')