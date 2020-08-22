

class PhysicalObject:
    """
    PhysicalObject
    """
    def __init__(self):
        self.name = ""
        self.image_path = ""
        self.image = None
        self.annotations = []


class Annotation:
    """
    Annotation
    """
    def __init__(self):
        self.type = ""
        self.position = [0.0, 0.0]
        self.text = ""
        self.start = [0.0, 0.0]
        self.end = [0.0, 0.0]
        self.video_path = ""
        self.audio_path = ""
        self.image_path = ""
        self.color = ""
        self.radius = 0
        self.thickness = 0
        self.width = 0
        self.height = 0
        self.update_orientation = False


class PoseEstimationOutput:
    """
    PoseEstimationOutput
    """
    def __init__(self, name, homography, error):
        self.object_name = name
        self.homography = homography
        self.error = error


class PoseEstimationInput:
    """
    PoseEstimationInput
    """
    def __init__(self, object_name, template_image, target_image, best_homography):
        self.object_name = object_name
        self.template_image = template_image
        self.target_image = target_image
        self.best_homography = best_homography


class ObjectDetectionResult:
    """
    ObjectDetectionResult
    """
    def __init__(self, lable, confidence, top_left, bottom_right):
        self.lable = lable
        self.confidnece = confidence
        self.top_left = top_left
        self.bottom_right = bottom_right
        self.image = None

