

class PoseEstimationOutput:
    def __init__(self, name, homography, error):
        self.object_name = name
        self.homography = homography
        self.error = error


class PoseEstimationInput:
    def __init__(self, object_name, template_image, target_image, best_homography):
        self.object_name = object_name
        self.template_image = template_image
        self.target_image = target_image
        self.best_homography = best_homography

