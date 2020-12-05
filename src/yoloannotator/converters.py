import cv2
import numpy as np
from yoloannotator.models import DefaultBBoxConfidenceModel, OpenCVBBoxConfidenceModel


class YoloToBBoxConfidenceConverter:
    """Converts yolo row output to DefaultBBoxConfidenceModel"""

    def __init__(self, confidence_threshold):
        self.confidence_threshold = confidence_threshold

    def convert(self, outputs):
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

    def __init__(self, confidence_threshold, nms_threshold):
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

    def convert(self, outputs, pixel_width, pixel_height):
        bbox_confidence_model_list = YoloToBBoxConfidenceConverter(
            self.confidence_threshold).convert(outputs)

        # shape, confidence and index has to be added seperatly
        # seperate lists will be used to resolve overlapping bboxes
        boxes = []
        confidence_list = []
        confidence_index_list = []

        for box in bbox_confidence_model_list:
            # row output format should be converted to pixels
            # row output has x, y, width, height in persentage to original image resolution
            width, height = int(
                box.width * pixel_width), int(box.height * pixel_height)

            # row yolo output x, y is the center point of the bbox
            # converting it to mix_x and mix_y
            x, y = int((box.x * pixel_width) - width /
                       2), int((box.y * pixel_height) - height / 2)

            boxes.append([x, y, width, height])
            confidence_index_list.append(box.confidence_index)
            '''@todo -  do we need float conversion?'''
            confidence_list.append(float(box.confidence))

        # remove overlapping bboxes
        # contains indexes of bboxes from original boxes list after filtering
        indexes = cv2.dnn.NMSBoxes(
            boxes, confidence_list,
            self.confidence_threshold, self.nms_threshold
        )

        filterd_boxes = []

        # indexes looks something like:: [ [10], [20] ]
        for i in indexes:
            # box at index "i" is the one we should select
            # this is the one opencv selected after filtering overlaps in bboxes list
            i = i[0]
            x, y, width, height = boxes[i]

            filterd_boxes.append(OpenCVBBoxConfidenceModel(
                # opencv expecting top left and bottom right points
                # right now it's top left x, y with width and height
                (x, y), (width + x, height + y), confidence_index_list[i], confidence_list[i]))

        return filterd_boxes
