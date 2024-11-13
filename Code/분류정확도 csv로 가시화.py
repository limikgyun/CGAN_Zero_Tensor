import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# CSV 파일 읽기
df = pd.read_csv('/home/mnetlig/Desktop/CSI 시각화/CSV_d_model_predictions.csv')
# 그래프 제목 설정
title = '정확도그래프'
save_path = '/home/mnetlig/Desktop/CSI 시각화'

# 한글 폰트 경로 설정
font_path = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
fontprop = fm.FontProperties(fname=font_path)

# 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(df['Step'], df['Real Data Accuracy'], label='Real Data Accuracy')
plt.plot(df['Step'], df['Fake Data Accuracy'], label='Fake Data Accuracy')

# y축 범위 설정
plt.ylim(0, 1)

# 그래프 제목 및 레이블 설정
plt.title(title, fontproperties=fontprop)
plt.xlabel('Step', fontproperties=fontprop)
plt.ylabel('Accuracy', fontproperties=fontprop)
plt.legend(prop=fontprop)

# 그리드 추가
plt.grid(True)

# 그래프 저장
plt.savefig(f'{save_path}/{title}.png', bbox_inches='tight')

# 그래프 표시
plt.show()