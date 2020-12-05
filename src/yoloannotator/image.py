import cv2


class Image:

    def __init__(self, img_path):
        self.img_path = img_path
        self.img = None

    def get_path(self):
        return self.img_path

    def get_size(self):
        h, w, _ = self.get_image().shape
        return [w, h]

    def get_blob(self, blob_width=None, blob_height=None):
        img = self.get_image()

        if blob_width is None:
            blob_width = img.shape[1]

        if blob_height is None:
            blob_height = img.shape[0]

        return cv2.dnn.blobFromImage(
            img, 1/225,
            (blob_width, blob_height),
            [0, 0, 0], 1,
            crop=False
        )

    def get_image(self):
        if self.img is None:
            print(self.img_path)
            self.img = cv2.imread(self.img_path)
        return self.img

