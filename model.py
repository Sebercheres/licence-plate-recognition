from paddleocr import PaddleOCR
from ultralytics import YOLO
import cv2
import numpy as np
import paddle
paddle.set_device('cpu')


class ALPR():
    def __init__(self):
        self.model_detect = YOLO('./models/license_plate_detector.pt')
        self.model_recog = PaddleOCR(use_angle_cls=False, det=False, lang='en',
                                     ocr_version='PP-OCRv3', vis_font_path='./simfang.ttf', 
                                     rec_char_dict_path='./plate_dict.txt',  
                                     rec_model_dir='./models/plate_rec/', det_model_dir='models/en_PP-OCRv3_det_infer',
                                     show_log=False, cls_model_dir='./models/ch_ppocr_mobile_v2.0_cls_infer')
    
    def should_flip_colors(self, image):
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Count the number of black and white pixels
        black_pixels = np.sum(gray_image == 0)
        white_pixels = np.sum(gray_image == 255)

        # Decide whether to flip colors based on pixel count
        return black_pixels > white_pixels

    def flip_colors(self, image):
        if self.should_flip_colors(image):
            # Invert the colors using bitwise NOT operation
            inverted_image = cv2.bitwise_not(image)
            return inverted_image
        else:
            return image
    
    def detect(self, image):
        results = self.model_detect(image, verbose=False)[0]
        try:
            x1, y1, x2, y2, score, class_id = results.boxes.data.tolist()[0]
        except:
            return None

        return x1, y1, x2, y2
    
    def read(self, image):
        img = self.flip_colors(image)
        res = self.model_recog.ocr(img, det=False, cls=False)
        return res

    def forward(self, image):
        try:
            x1, y1, x2, y2 = self.detect(image)
        except:
            return None, None, None  # No detection

        # Crop the image
        img_crop = image[int(y1):int(y2), int(x1):int(x2)]

        ocr_result = self.read(img_crop)

        # Return bounding box coordinates, cropped image, and OCR result
        return (x1, y1, x2, y2), img_crop, ocr_result