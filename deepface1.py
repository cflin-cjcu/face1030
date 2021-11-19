# pip install deepface
# 匯入模組 deepface 會新增模組資料夾
from deepface import DeepFace
import cv2
# 用cv2顯示圖片
imgpath = "aaa.jpg"
# img_sr = cv2.imread(imgpath)
# cv2.imshow('aaa', img_sr)
# 人臉偵測
image = DeepFace.detectFace(
    img_path=imgpath,
    # detector_backend='opencv',  # 以不同模型進行偵測臉部
    enforce_detection=False
)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
# 基本驗證
face1 = 'cflin.jpg'
face2 = 'cflin3.jpg'
result = DeepFace.verify(
    face1, face2, model_name='VGG-Face', enforce_detection=False)
print(result)
if result['verified'] == True:
    print("這二張圖片是同一個人")
else:
    print("這二張圖片不是同一個人")
# 尋找相同的臉
# face1 = "./dataset/img13.jpg"
# df = DeepFace.find(img_path=face1, db_path='./dataset',
#                    model_name='Facenet', enforce_detection=False)
# #condition = df['Facenet_cosine'] < 0.2
# print(df)
# 人臉屬性分析
obj = DeepFace.analyze(
    img_path=imgpath,
    actions=['age', 'gender', 'race', 'emotion'],
    enforce_detection=False)
print(obj)

cv2.imshow('bbb', image)
# plt.imshow(image)
cv2.waitKey(0)
cv2.destroyAllWindows()
