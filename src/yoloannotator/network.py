import cv2
from yoloannotator.models import DefaultBBoxConfidenceModel, OpenCVBBoxConfidenceModel


class NetworkManager:
    """Represents a nural network in openvc

    Provides a common interface for processes like object detection from given network
    Use CVNetworkBuilder to build new CVNetworks
    """

    def set_net_configurer(self, configurer):
        self.configurer = configurer
        cv2.dnn.readNet

    def set_network(self, network):
        self.network = network

    def set_bbox_converter(self, converter):
        self.converter = converter

    def get_image_bboxes(self, image):
        """Returns bounding boxes for objects matched in image"""
        output = self.network.get_layer_output(image)
        width, height = image.get_size()
        return self.converter.convert(output, width, height)


class NetworkManagerBuilder:

    def init():
        return NetworkManagerBuilder()

    def with_network_type(self, cv_net_type):
        self.net_type = cv_net_type
        return self

    def with_network_factory(self, factory):
        self.net_factory = factory
        return self

    def with_net_configurer(self, configurer):
        self.net_configurer = configurer
        return self

    def with_bbox_converter(self, modifier):
        self.bbox_converter = modifier
        return self

    def build(self):
        if self.net_type is None:
            raise "Set the type of network needs to be built"

        if not hasattr(self, 'net_factory'):
            self.net_factory = DefaultNetoworkFactory()

        if not hasattr(self, 'net_configurer '):
            self.net_configurer = CudaConfigurer()

        network = NetworkManager()
        network.set_network(self.net_factory.get(self.net_type))
        network.set_net_configurer(self.net_configurer)
        if hasattr(self, 'bbox_converter'):
            network.set_bbox_converter(self.bbox_converter)

        return network


class DefaultNetoworkFactory:
    """Returns new network"""

    def get(self, type):
        if type == 'darknet':
            return DarknetYoloNetwork()


class DarknetYoloNetwork:
    """Network that is compatible with Darknet"""

    def __init__(self, cfg_path='resources/obj.cfg', weights_path='resources/obj.weights'):
        self.cfg_path = cfg_path
        self.weights_path = weights_path
        self.net = cv2.dnn.readNetFromDarknet(self.cfg_path, self.weights_path)

    def get_net(self):
        """Returns opencv network object"""
        return self.net

    def get_layer_output(self, image):
        """Returns row output from yolo output layers"""

        width, height = image.get_size()
        image_blob = image.get_blob(width, height)

        self.net.setInput(image_blob)
        return self.forward_layers()

    def get_output_layers(self):
        """Returns names of output layers of yolo network"""
        layer_names = self.net.getLayerNames()
        output_layers = self.net.getUnconnectedOutLayers()
        return [layer_names[i[0] - 1] for i in output_layers]

    def forward_layers(self):
        """Returns the output from output layers of yolo network"""
        layers = self.get_output_layers()
        return self.net.forward(layers)


class CudaConfigurer:

    def configure(self, net):
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        return net
