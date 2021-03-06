class DefaultBBoxConfidenceModel:
    def __init__(self, x, y, width, height, confidence_index, confidence):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.confidence_index = confidence_index
        self.confidence = confidence

    def __str__(self):
        return ("class index::" + str(self.confidence_index) +
        ", confidence::"+ str(self.confidence))

    def __repr__(self):
        return ("class index::" + str(self.confidence_index) +
        ", confidence::"+ str(self.confidence))


class OpenCVBBoxConfidenceModel:
    def __init__(self, min_point, max_point, confidence_index, confidence):
        self.min_point = min_point
        self.max_point = max_point
        self.confidence_index = confidence_index
        self.confidence = confidence

