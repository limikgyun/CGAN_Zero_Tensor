from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, f1_score
import pickle
import numpy as np
from numpy import inf
import os
from models import *
from utils import *

def ttest(model_paths, data_path):
    # 1. 저장된 c_model을 로드합니다.
    # model_paths = [
    # '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/c to b/Trans-Base간-240924-1408-c-1000샘플-80에폭-86.h5',
    # '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/c to b/Trans-Base간-240924-1606-c-1000샘플-80에폭-86.h5'

    #     # '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/y/240905-1425-GAN-c-3840samples-100.h5', # 대조군
    #     # '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/t-240920-1958-c-1000샘플-50에폭-19.h5' # #2 case 로 전이학습한 g를 사용한 모델! 바로 위의 대조군과 비교 시 미량의 정확도 상승을 보임........ 이정도론 부족함

    #     # '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/t-240920-1953-c-640샘플-60에폭-26.h5',
    #     # '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/t-240920-1954-c-640샘플-60에폭-26.h5',
    #     # '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/t-240920-1955-c-640샘플-60에폭-27.h5',

    #     # '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/transfer-240919-2112-c-3840samples-70.h5' #***35epoch 샘플개수 3840개 전이학습 완료된 cGAN으로 생성한 데이터로 전이학습된 GAN모델***
    #     # '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/transfer-240919-2030-c-3840samples-69.h5',#100epoch 샘플개수 3840개 전이학습 완료된 cGAN으로 생성한 데이터로 전이학습된 GAN모델
    # ]
    models = [load_model(path) for path in model_paths]

    # 2. 테스트 데이터 로드
    dataset = data_preproc(np.asarray(pickle.load(open(data_path,'rb'))))
    # dataset = data_preproc(np.asarray(pickle.load(open('/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Processed_b_0.pickle','rb'))))
    X_tra, y_tra, X_tst, y_tst = dataset


    # # 3. 뱉어낸 클래스 출력
    # for i, model in enumerate(models, 1):
    #     predictions = model.predict(X_tst)
    #     predicted_classes = np.argmax(predictions, axis=1)
    #     print(f'Model c{i}_model - Predicted Classes: {predicted_classes}')

    # # 3. 클래스별 확률 출력
    # for i, model in enumerate(models, 1):
    #     predictions = model.predict(X_tst)
    #     predictions_rounded = np.round(predictions, 3)
    #     print(f'Model c{i}_model - Predictions: {predictions_rounded}')

    # 각 모델에 대한 클래스별 정확도 평가
    for i, model in enumerate(models, 1):
        predictions = model.predict(X_tst)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # 혼동 행렬 생성
        cm = confusion_matrix(y_tst, predicted_classes)
        
        # 클래스별 정확도 계산
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        
        print(f'Model c{i}_model - Class-wise Accuracies:')
        for class_idx, accuracy in enumerate(class_accuracies):
            print(f'  Class {class_idx}: {accuracy * 100:.2f}%')

        # F1 스코어 계산
        f1 = f1_score(y_tst, predicted_classes, average='weighted')
        print(f'Model c{i}_model - F1 Score: {f1:.2f}')

    # 각 모델에 대한 분류 정확도 평가
    for i, model in enumerate(models, 1):
        loss, accuracy = model.evaluate(X_tst, y_tst, verbose=0)
        print(f'Model c{i}_model - Test Accuracy: {accuracy * 100:.2f}%')

if __name__ == '__main__':
## 240414 실험~!!
    # ttest(['/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Base간-Processed_x_00-c-3000샘플-100에폭.h5',
    #        '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Base간-Processed_x++_00-c-3000샘플-100에폭.h5'],
    #        '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Processed_x_00.pickle')
    # ttest(['/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Base간-Processed_x_00-c-3000샘플-100에폭.h5',
    #        '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Base간-Processed_x++_00-c-3000샘플-100에폭.h5'],
    #        '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Processed_x++_00.pickle')
    # ttest(['/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Trans-Base간-Generated_[1]_x_00-c-1000샘플-90에폭.h5',
    #        '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Trans-Base간-Generated_[1]_x++_00-c-1000샘플-90에폭.h5'],
    #         '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Processed_x_00.pickle')
    # ttest(['/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Trans-Base간-Generated_[1]_x_00-c-1000샘플-90에폭.h5',
    #        '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Trans-Base간-Generated_[1]_x++_00-c-1000샘플-90에폭.h5'],
    #         '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Processed_x++_00.pickle')
    ## for z
    ttest(['/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Base간-Processed_z_00-c-3000샘플-100에폭.h5',
           '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Base간-Processed_z++_00-c-3000샘플-100에폭.h5',
           '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Trans-Base간-Generated_[1]_z_00-c-1000샘플-7에폭.h5',],
           '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Processed_z_00.pickle')
    # ttest(['/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Base간-Processed_z_00-c-3000샘플-100에폭.h5',
    #        '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Base간-Processed_z++_00-c-3000샘플-100에폭.h5'],
    #        '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Processed_z++_00.pickle')
#     ttest(['/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Trans-Base간-Generated_[1]_x_00-c-1000샘플-90에폭.h5',
#            '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Trans-Base간-Generated_[1]_x++_00-c-1000샘플-90에폭.h5'],
#             '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Processed_x_00.pickle')
#     ttest(['/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Trans-Base간-Generated_[1]_x_00-c-1000샘플-90에폭.h5',
#            '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Trans-Base간-Generated_[1]_x++_00-c-1000샘플-90에폭.h5'],
#             '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Processed_x++_00.pickle')
    
           

#     ttest([
# '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Tuned_T_Base/b2a_Trans-Base간-241002-1203-c-1000샘플-70에폭-92.h5'],
#         '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Processed_a_0.pickle')
#     ttest([
# '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Tuned_T_Base/c2a_Trans-Base간-240924-1322-c-1000샘플-30에폭-86.h5',
# '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Tuned_T_Base/c2a_Trans-Base간-240924-1330-c-1000샘플-80에폭-87.h5'],
#         '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Processed_a_0.pickle')
#     ttest([
# '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Tuned_T_Base/a2b_Trans-Base간-240924-1029-c-1000샘플-80에폭-98.h5',],
#         '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Processed_b_0.pickle')
#     ttest([
# '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Tuned_T_Base/c2b_Trans-Base간-240924-1408-c-1000샘플-80에폭-86.h5',
# '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Tuned_T_Base/c2b_Trans-Base간-240924-1606-c-1000샘플-80에폭-86.h5'],
#         '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Processed_b_0.pickle')
#     ttest([
# '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Tuned_T_Base/a2c_Trans-Base간-241002-1407-c-1000샘플-53에폭-92.h5'],
#         '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Processed_c_0.pickle')
#     ttest([
# '/home/mnetlig/Desktop/CSI-SemiGAN-master/models/Tuned_T_Base/b2c_Trans-Base간-241002-1445-c-1000샘플-90에폭-86.h5'],
#         '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/Processed_c_0.pickle')
