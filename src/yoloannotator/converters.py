import cv2
import numpy as np
from yoloannotator.models import DefaultBBoxConfidenceModel, OpenCVBBoxConfidenceModel
from pprint import PrettyPrinter

class YoloToBBoxConfidenceConverter:
    """Converts YOLO row output to DefaultBBoxConfidenceModel
    
    YOLO row output:

    <x_center> <y_center> <width> <height> <object_confidence> <class1> <class2>
    [5.6381125e-02 4.4504557e-02 3.2851920e-01 3.1377596e-01 1.0989135e-08 0
            0.6381125e-02]

    Rest of the elements of the list contains confidence level for each and
    every class of the network
    """

    def __init__(self, confidence_threshold):
        self.confidence_threshold = confidence_threshold

    def convert(self, outputs):
        """ Returns DefaultBBoxConfidenceModel from YOLO row output
        @param outputs: 
            YOLO network has 3 output layers
            Output passes 3 element array that contains all outputs from 3
            output layers
        """
        boxes = []

        for output in outputs:
            for detection in output:
                confidence_list = detection[5:]
                max_confidence_index = np.argmax(confidence_list)
                max_confidence = confidence_list[max_confidence_index]

                if max_confidence < self.confidence_threshold:
                    continue

                x, y, width, height = detection[:4]

                boxes.append(DefaultBBoxConfidenceModel(
                    x, y, width, height, max_confidence_index, max_confidence))

        return boxes


class YoloToOpenCVBBoxConfidenceConverter:
    """Converts Yolo type bbox to opencv supported bbox

    YOLO output is different from opencv supported rectangle type
    YoloToOpenCVBBoxConfidenceConverter convertes data to opencv supported
    data
    """

    def __init__(self, confidence_threshold, nms_threshold):
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

    def convert(self, outputs, img_pixel_width, img_pixel_height):
        # convert yolo output to DefaultBBoxConfidenceModel
        bbox_confidence_model_list = YoloToBBoxConfidenceConverter(
            self.confidence_threshold).convert(outputs)


        # create seperate lists to normalize bounding bboxes
        listby_index = {}

        for box in bbox_confidence_model_list:
            cindex = box.confidence_index

            if listby_index.get(cindex) == None:
                listby_index[cindex] = {
                    'bbox_list': [],
                    'confidence_list': []
                }

            obj = listby_index[cindex]

            obj['bbox_list'].append(self.get_pixel_bbox(
                box.x, box.y,
                box.width, box.height,
                img_pixel_width, img_pixel_height))

            obj['confidence_list'].append(float(box.confidence))

        # normalize bboxes using 
        index_list_by_confidence_index = {}

        for key, obj in listby_index.items():
            index_list_by_confidence_index[key] = self.nms_boxes(
                obj['bbox_list'],
                obj['confidence_list'])

        filterd_boxes = []

        for key, indexlist in index_list_by_confidence_index.items():
            for index in indexlist:
                index = index[0]

                confidence_index = key
                confidence = listby_index[key]['confidence_list'][index]
                x, y, width, height = listby_index[key]['bbox_list'][index]

                filterd_boxes.append(OpenCVBBoxConfidenceModel(
                    # opencv expecting top left and bottom right points
                    # right now it's top left x, y with width and height
                    (x, y), (width + x, height + y), confidence_index , confidence))

        return filterd_boxes

    def nms_boxes(self, bbox_list, confidence_list):
        return cv2.dnn.NMSBoxes(
            bbox_list, confidence_list,
            self.confidence_threshold, self.nms_threshold
        )
        
    def get_pixel_bbox(self, x, y, width, height, img_pixel_width, img_pixel_height):
        width, height = int(width * img_pixel_width), int(height * img_pixel_height)
        x, y = int((x * img_pixel_width) - (width / 2)), int((y * img_pixel_height) - (height / 2))

        return (x, y, width, height)
