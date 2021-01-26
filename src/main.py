#!/usr/bin/env python

from sys import argv
from yoloannotator.network import NetworkManagerBuilder
from yoloannotator.converters import YoloToOpenCVBBoxConfidenceConverter
from yoloannotator.supervisely import Project


def main():
    classnames = get_classes()
    network = NetworkManagerBuilder.init().with_network_type(
        'darknet').with_bbox_converter(
                YoloToOpenCVBBoxConfidenceConverter(0.5, 0.3)).build()

    '''
    If you just want to display the bounding boxes in opencv
    Uncomment following commeted lines
    It's just taking one image from the resources dir
    Good way to make sure besic stuff are working before proceeding to
    generating the project
    '''
    # image = Image('resources/image.png')
    # boxes = network.get_image_bboxes(image)
    # image_data = ImageData(image, classnames, boxes)

    # print(image_data.get_image_json())

    # for box in boxes:
    #     cv2.rectangle(image.get_image(), box.min_point,
    #                   box.max_point, (255, 0, 255), 2)

    # cv2.imshow('image', image.get_image())
    # cv2.waitKey(0)


    project = Project(network, classnames, './resources/training', './resources')
    project.generate()


def get_classes():
    with open('resources/obj.names', 'rt') as file:
        return file.read().splitlines()


if __name__ == '__main__':
    main()

