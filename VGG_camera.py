import numpy as np 
import json 
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import models,transforms
import cv2

use_pretrained = True 
net = models.vgg16(use_pretrained)
net.eval()

class BaseTransform():

    def __init__(self,resize,mean,std):

        self.base_transform = transforms.Compose([ 
            transforms.Resize(resize), 
            transforms.CenterCrop(resize),
            transforms.ToTensor(), 
            transforms.Normalize(mean,std) 

        ])

    def __call__(self,img):

        return self.base_transform(img)

resize = 224
mean = (0.485,0.456,0.406)
std = (0.229,0.224,0.225)

ILSVRC_class_index = json.load(open('C:\\imagenet_class_index.json','r'))
ILSVRC_class_index

class ILSVRCPredictor():

    def __init__(self,class_index):

        self.class_index = class_index

    def predict_max(self,out):

        maxid = np.argmax(out.detach().numpy()) 
        predicted_label_name = self.class_index[str(maxid)][1]
        return predicted_label_name


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)

transform = BaseTransform(resize,mean,std)
predictor = ILSVRCPredictor(ILSVRC_class_index)
while (cap.isOpened()):
    ret,frame = cap.read()
    frame = Image.fromarray(frame)
    img_transformed = transform(frame) 
    inputs = img_transformed.unsqueeze_(0)  
    out =net(inputs)
    result = predictor.predict_max(out)

    print("入力画像の予測結果:", result)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
