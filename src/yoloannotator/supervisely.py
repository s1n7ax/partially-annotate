import json
import re
import datetime
import time
import random
from shutil import copyfile

from pathlib import Path
from os import listdir, path
from os.path import join
from yoloannotator.models import OpenCVBBoxConfidenceModel
from yoloannotator.image import Image
from typing import List


def get_image_json_dummy(width, height):
    return {
        "description": "",
        "tags": [],
        "size": {
            "width": width,
            "height": height
        },
        "objects": []
    }


def get_bbox_json_dummy(classname: str, min_point: List[int], max_point: List[int]):
    return {
        "description": "",
        "geometryType": "rectangle",
        "tags": [],
        "classTitle": classname,
        "points": {
            "exterior": [
                min_point,
                max_point
            ],
            "interior": []
        }
    }


def get_meta_json_dummy(clssnames: List[str]):
    c = lambda: random.randint(0, 255)

    meta = {}
    meta['classes'] = []
    meta['tags'] = []
    meta['projectType'] = 'images'
    for classname in clssnames:
        color = '#%02X%02X%02X' % (c(), c(), c())
        meta['classes'].append({
            "title": classname,
            "shape": "rectangle",
            "color": color,
            "geometry_config": {},
            "id": 1,
            "hotkey": ""
        })
    return meta


class ImageData:

    def __init__(self, image: Image, classnames: List[str], bbox_list: List[OpenCVBBoxConfidenceModel]):
        self.image = image
        self.classnames = classnames
        self.bbox_list = bbox_list

    def get_file_path(self):
        return self.image.get_file_path()

    def get_size(self):
        return self.image.get_size()

    def get_image_data(self):
        width, height = self.get_size()
        image_json = get_image_json_dummy(width, height)

        for bbox in self.bbox_list:
            classname = self.classnames[bbox.confidence_index]
            bbox_json = get_bbox_json_dummy(
                classname, list(bbox.min_point), list(bbox.max_point))
            image_json["objects"].append(bbox_json)

        return image_json


class Project:
    def __init__(self, network, classname_list, training_data_dir, outputdir_root='resources'):
        self.network = network
        self.classname_list = classname_list
        self.training_data_dir = training_data_dir
        self.outputdir_root = outputdir_root

        # set outputdir using timestamp
        st = datetime.datetime.fromtimestamp(
            time.time()).strftime('%Y_%m_%d %H_%M_%S')
        self.outputdir = path.join(outputdir_root, st)
        self.output_img_dir = path.join(self.outputdir, 'data', 'img')
        self.output_ann_dir = path.join(self.outputdir, 'data', 'ann')

        # create dirs
        Path(self.output_img_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_ann_dir).mkdir(parents=True, exist_ok=True)

    def generate(self):
        meta_file = path.join(self.outputdir, 'meta.json')
        self.write_json(meta_file, get_meta_json_dummy(self.classname_list))

        for f in listdir(self.training_data_dir):
            if self.isimage(join(self.training_data_dir, f)):
                image_json_file = path.join(self.output_ann_dir, f + '.json')
                image_path = path.join(self.training_data_dir, f)

                image = Image(image_path)
                boxes = self.network.get_image_bboxes(image)

                imagedata = ImageData(
                    image, self.classname_list, boxes).get_image_data()

                copyfile(image_path, path.join(self.output_img_dir, f))
                self.write_json(image_json_file, imagedata)

    # @todo let the user deside what types of images to filter
    def isimage(self, file: str) -> bool:
        return re.match("(.*.png$|.*.jpeg$|.*.jpg$)", file) is not None

    def write_json(self, path, jsondata):
        with open(path, 'w') as file:
            json.dump(jsondata, file)
