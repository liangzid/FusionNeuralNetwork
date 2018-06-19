import numpy as np
import cv2
import model.hand_on_feature.LOMO as LOMO
import os

path='/home/liangzi/文档/test_dataset'


imgs_path=[os.path.join(path,img) for img in os.listdir(path)]
feature=np.zeros((len(imgs_path),26960))
for i in range(len(imgs_path)):
    img=cv2.imread(imgs_path[i])

    feature[i,:]=LOMO.LOMO(img)
#print(feature)
#np.savetxt('./feature1.txt',feature[1,:])
def normalizee(feature1):
    lens=len(feature1)
    max=feature1[0]
    min=feature1[0]
    for i in range(lens):
        if feature1[i]>max:
            max=feature1[i]
        if feature1[i]<min:
            min=feature1[i]
    for i in range(lens):
        feature1[i]=int((feature1[i]-min)*255/(max-min))

    return feature1

feature1=normalizee(feature[1,:])[0:168*96].reshape(168,96)
cv2.imshow('pa',feature1)
cv2.waitKey(0)
cv2.destroyAllWindows()











