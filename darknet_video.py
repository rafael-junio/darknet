import os
import logging
import cv2
from . import darknet


class DarknetProcessor:

    def __init__(self, configPath="./core/detector/darknet/cfg/yolov4-custom-semalo-classes-complete.cfg",
                 weightPath="./core/detector/darknet/backup/yolov4-custom-semalo-classes-complete_best.weights",
                 metaPath="./core/detector/darknet/data/obj-semalo-classes.data"):

        self.netMain = None
        self.metaMain = None
        self.altNames = None
        self.darknet_image = None

        self.configPath = configPath
        self.weightPath = weightPath
        self.metaPath = metaPath

        if not os.path.exists(self.configPath):
            raise ValueError("Invalid config path `" +
                             os.path.abspath(self.configPath) + "`")
        if not os.path.exists(self.weightPath):
            raise ValueError("Invalid weight path `" +
                             os.path.abspath(self.weightPath) + "`")
        if not os.path.exists(self.metaPath):
            raise ValueError("Invalid data file path `" +
                             os.path.abspath(self.metaPath) + "`")
        if self.netMain is None:
            self.netMain = darknet.load_net_custom(self.configPath.encode(
                "ascii"), self.weightPath.encode("ascii"), 0, 1)  # batch size = 1
        if self.metaMain is None:
            self.metaMain = darknet.load_meta(self.metaPath.encode("ascii"))
        if self.altNames is None:
            try:
                with open(self.metaPath) as metaFH:
                    metaContents = metaFH.read()
                    import re
                    match = re.search("names *= *(.*)$", metaContents,
                                      re.IGNORECASE | re.MULTILINE)
                    if match:
                        result = match.group(1)
                    else:
                        result = None
                    try:
                        if os.path.exists(result):
                            with open(result) as namesFH:
                                namesList = namesFH.read().strip().split("\n")
                                self.altNames = [x.strip() for x in namesList]
                    except TypeError:
                        pass
            except TypeError:
                pass

        # Create an image we reuse for each detect
        self.darknet_image = darknet.make_image(darknet.network_width(self.netMain),
                                                darknet.network_height(self.netMain), 3)
        logging.info("YoloV4 STARTED...")

    def process_Yolov4_detection(self, frame_read):
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(self.netMain),
                                    darknet.network_height(self.netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(self.darknet_image, frame_resized.tobytes())

        detections = darknet.detect_image(self.netMain, self.metaMain, self.darknet_image, thresh=0.1)
        return detections
