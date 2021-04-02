import argparse
import time
from pathlib import Path
import random
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import os

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from torchvision import transforms


class V5:
    def __init__(self , weights):
        # Initialize
        set_logging()
        device = select_device('')
        half = device.type != 'cpu'  # half precision only supported on CUDA
        
        # Load model
        self.model = attempt_load(weights, map_location=device)  # load FP32 model

    def detect(self , img1):
        
        # Get names and colors
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        trans = transforms.ToTensor()   
        # while True:
        # Run inference
        t0 = time.time()

        # img1 = cv2.imread('data/images/1.jpg')
        img1 = cv2.cvtColor(img1 , cv2.COLOR_BGR2RGB)
        img  = trans(img1)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = self.model(img, augment=False)[0]
        # Apply NMS
        pred = non_max_suppression(pred, 0.3, 0.45, classes=None, agnostic=False)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  
            print(det)
            gn = torch.tensor(img1.shape)[[1, 0, 1, 0]]  # norma

            display_str_list = []
            display_str_dict={}
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img1.shape).round()
            

                for *xyxy, conf, cls in reversed(det):
                    # print('xyxy = ',xyxy , 'conf = ',conf , 'cls = ',cls)
                    label = f'{names[int(cls)]} {conf:.2f}'
                    x1=int(xyxy[0].item())
                    y1=int(xyxy[1].item())
                    x2=int(xyxy[2].item())
                    y2=int(xyxy[3].item())

                    plot_one_box(xyxy, img1, label=label, color=colors[int(cls)], line_thickness=3)
            cv2.imshow(str('a'), img1)
            cv2.waitKey(0)
            cv2.destroyAllWindows()



if __name__ == '__main__':
    model = V5('best.pt')
    test_dir = './test_images/'
    test_imgs = []
    for path, subdirs, files in os.walk(test_dir):
        for name in files:
            test_imgs.append(os.path.join(path, name))
    random_test_image = random.choice(test_imgs)
    image = cv2.imread(random_test_image)
    image = cv2.resize(image, (416, 416))
    cv2.imshow(str('a'), image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    model.detect(image)
