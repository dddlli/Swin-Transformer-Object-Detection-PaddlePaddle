import paddle.vision.transforms as transforms
import numpy as np
import cv2
import paddle

class Extractor(object):    #特征提取器的定义：
    def __init__(self, Net): #已经训练好的model_加载进来。
        self.net = Net
        #self.net.eval()
        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.58666682, 0.58484647, 0.57418193], [0.20736474, 0.19249499, 0.1870952]),
        ])

    def _preprocess(self, im_crops):    #私有的preprocess函数，完成对roi区域的resize

        def _resize(im, size):  #私有的_resize
            return cv2.resize(im.astype(np.float32)/255., size)
        
        im_batch = paddle.concat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], axis=0) #.float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with paddle.no_grad():
            features = self.net(im_batch) #将roi区域送到特征提取器中去提取特征，从而获得特征。
        return features.cpu().numpy()
