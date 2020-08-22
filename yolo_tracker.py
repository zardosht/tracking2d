import traceback

import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
import logging
from yolo_pose_estimator import PoseEstimator
import multiprocessing as mp
from yolo_tracker_classes import PhysicalObject, Annotation, \
        ObjectDetectionResult, PoseEstimationInput, PoseEstimationOutput



options = {
    'model': 'model/tools/tools_v2.cfg',
    'load': 'model/tools/tools_v2_12600.weights',
    'labels': 'model/tools/labels.txt',
    'threshold': 0.5,
    'gpu': 1.0
}


model_loaded = False
tfnet = None

croppedImagesPath = "./tmp/croppedImages/"
cropped_image_extension = ".jpg"
physical_objects_data_path = "./physical_objects/tools/"

logger = logging.getLogger('tracking2d.yolo_tracker')
debug = False


homography_error_drawing_threshold = 200


def load_yolo_model():
    global tfnet
    tfnet = TFNet(options)


def get_predictions(frame):
    prediction_results = tfnet.return_predict(frame)
    predictions = []
    for pred_result in prediction_results:
        tl = (pred_result['topleft']['x'], pred_result['topleft']['y'])
        br = (pred_result['bottomright']['x'], pred_result['bottomright']['y'])
        label = pred_result['label']
        confidnece = pred_result['confidence']
        predictions.append(ObjectDetectionResult(label, confidnece, tl, br))
    return predictions


def draw_results_on_frame(frame, colors, objectDetectionResults):
    for color, result in zip(colors, objectDetectionResults):
        text = '{}: {:.0f}%'.format(result.lable, result.confidnece * 100)
        # frame = cv2.rectangle(frame, result.top_left, result.bottom_right, color, 5)
        # frame = cv2.putText(frame, text, result.top_left, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

        frame = cv2.rectangle(frame, result.top_left, result.bottom_right, color, 2)
        frame = cv2.putText(frame, text, result.top_left, cv2.FONT_HERSHEY_COMPLEX, .5, (0, 0, 0), 1)


def cropPredictedObjects(frame, results):
    for result in results:
        # if result.confidnece * 100 > 60:
        result.image = frame[result.top_left[1]:result.bottom_right[1], result.top_left[0]:result.bottom_right[0]]
        if debug: cv2.imwrite(croppedImagesPath + result.lable + cropped_image_extension, result.image)

def compute_new_position(position, homography):
    try:
        # homogenous_position = np.array((position[0], position[1], 1)).reshape((3, 1))
        # transformed_position = np.dot(homography, homogenous_position)
        # transformed_position = np.sum(transformed_position, 1)
        # new_x = int(round(transformed_position[0] / transformed_position[2]))
        # new_y = int(round(transformed_position[1] / transformed_position[2]))

        homogenous_position = np.array((position[0], position[1], 1)).reshape((3, 1))
        new_position = np.dot(homography, homogenous_position)
        new_x = new_position[0]
        new_y = new_position[1]

        return new_x, new_y

    except Exception:
        logger.error("Exception in transforming new annotation position. Homography: %s", homography)
        traceback.print_exc()
        return position


def draw_annotations(frame, present_phys_objs, homographies):
    for physical_object, object_detection_result in present_phys_objs:
        if not physical_object.name in homographies.keys() or homographies[physical_object.name] is None:
            logger.warning("physicalObject.name is not in homographies.keys() or homographies[physicalObject.name] is None: %s", physical_object.name)
            continue
        if homographies[physical_object.name].error > homography_error_drawing_threshold:
            logger.warning("Homography has a very large error: %s", physical_object.name)
            continue
        for annotation in physical_object.annotations:
            if annotation.type == "CircleAnnotation":
                new_position = compute_new_position(annotation.position, homographies[physical_object.name].homography)
                # new_position = annotation.position
                newAbsolutePosition = (object_detection_result.top_left[0] + new_position[0], object_detection_result.top_left[1] + new_position[1])

                logger.debug("new_position: %s", newAbsolutePosition)
                frame = cv2.circle(frame, newAbsolutePosition, annotation.radius, annotation.color, annotation.thickness)


def create_dummy_physical_objects():
    phs = []
    pincers = PhysicalObject()
    pincers.name = "Pincers"
    pincers.image_path = "Pincers.jpg"
    pincers.image = cv2.imread(physical_objects_data_path + pincers.image_path)
    textAnnotation = Annotation()
    textAnnotation.type = "TextAnnotation"
    textAnnotation.text = "This is a pincers"
    textAnnotation.position = [2, 2]
    pincers.annotations.append(textAnnotation)
    arrow = Annotation()
    arrow.type = "Arrow"
    arrow.start = [10.5, 20.9]
    arrow.end = [20.5, 30.5]
    pincers.annotations.append(arrow)
    videoAnnotation = Annotation()
    videoAnnotation.type = "VideoAnnotation"
    videoAnnotation.position = [40.5, 50.9]
    videoAnnotation.video_path = "pincers_video.mpg"
    pincers.annotations.append(videoAnnotation)
    circle_annotation = Annotation()
    circle_annotation.type = "CircleAnnotation"
    circle_annotation.radius = 20
    circle_annotation.position = [72, 70]
    circle_annotation.color = [0, 0, 255]
    circle_annotation.thickness = 3
    pincers.annotations.append(circle_annotation)
    circle_annotation = Annotation()
    circle_annotation.type = "CircleAnnotation"
    circle_annotation.radius = 20
    circle_annotation.position = [122, 330]
    circle_annotation.color = [0, 255, 123]
    circle_annotation.thickness = 3
    pincers.annotations.append(circle_annotation)
    phs.append(pincers)

    adjSpanner = PhysicalObject()
    adjSpanner.name = "Adjustable Spanner"
    adjSpanner.image_path = "Adjustable Spanner.jpg"
    adjSpanner.image = cv2.imread(physical_objects_data_path + adjSpanner.image_path)
    textAnnotation = Annotation()
    textAnnotation.type = "TextAnnotation"
    textAnnotation.text = "This is an Adjustable Spanner"
    textAnnotation.position = [2, 2]
    adjSpanner.annotations.append(textAnnotation)
    arrow = Annotation()
    arrow.type = "Arrow"
    arrow.start = [10.5, 20.9]
    arrow.end = [20.5, 30.5]
    adjSpanner.annotations.append(arrow)
    videoAnnotation = Annotation()
    videoAnnotation.type = "VideoAnnotation"
    videoAnnotation.position = [40.5, 50.9]
    videoAnnotation.video_path = "Adjustable_Spanner_video.mpg"
    adjSpanner.annotations.append(videoAnnotation)
    circle_annotation = Annotation()
    circle_annotation.type = "CircleAnnotation"
    circle_annotation.radius = 20
    circle_annotation.position = [74, 380]
    circle_annotation.color = [0, 0, 255]
    circle_annotation.thickness = 3
    adjSpanner.annotations.append(circle_annotation)
    phs.append(adjSpanner)

    pump_pliers = PhysicalObject()
    pump_pliers.name = "Pump Pliers"
    pump_pliers.image_path = "Pump Pliers.jpg"
    pump_pliers.image = cv2.imread(physical_objects_data_path + pump_pliers.image_path)
    textAnnotation = Annotation()
    textAnnotation.type = "TextAnnotation"
    textAnnotation.text = "This is an Pump Pliers"
    textAnnotation.position = [2, 2]
    pump_pliers.annotations.append(textAnnotation)
    arrow = Annotation()
    arrow.type = "Arrow"
    arrow.start = [10.5, 20.9]
    arrow.end = [20.5, 30.5]
    pump_pliers.annotations.append(arrow)
    videoAnnotation = Annotation()
    videoAnnotation.type = "VideoAnnotation"
    videoAnnotation.position = [40.5, 50.9]
    videoAnnotation.video_path = "Pump_Pliers_video.mpg"
    pump_pliers.annotations.append(videoAnnotation)
    circle_annotation = Annotation()
    circle_annotation.type = "CircleAnnotation"
    circle_annotation.radius = 20
    circle_annotation.position = [112, 78]
    circle_annotation.color = [0, 0, 255]
    circle_annotation.thickness = 3
    pump_pliers.annotations.append(circle_annotation)
    circle_annotation = Annotation()
    circle_annotation.type = "CircleAnnotation"
    circle_annotation.radius = 20
    circle_annotation.position = [48, 464]
    circle_annotation.color = [123, 255, 0]
    circle_annotation.thickness = 3
    pump_pliers.annotations.append(circle_annotation)
    circle_annotation = Annotation()
    circle_annotation.type = "CircleAnnotation"
    circle_annotation.radius = 20
    circle_annotation.position = [136, 474]
    circle_annotation.color = [255, 123, 0]
    circle_annotation.thickness = 3
    pump_pliers.annotations.append(circle_annotation)
    phs.append(pump_pliers)

    linemans_pliers = PhysicalObject()
    linemans_pliers.name = "Linemans Pliers"
    linemans_pliers.image_path = "Linemans Pliers.jpg"
    linemans_pliers.image = cv2.imread(physical_objects_data_path + linemans_pliers.image_path)
    textAnnotation = Annotation()
    textAnnotation.type = "TextAnnotation"
    textAnnotation.text = "This is a Linemans Pliers"
    textAnnotation.position = [2, 2]
    linemans_pliers.annotations.append(textAnnotation)
    arrow = Annotation()
    arrow.type = "Arrow"
    arrow.start = [10.5, 20.9]
    arrow.end = [20.5, 30.5]
    linemans_pliers.annotations.append(arrow)
    videoAnnotation = Annotation()
    videoAnnotation.type = "VideoAnnotation"
    videoAnnotation.position = [40.5, 50.9]
    videoAnnotation.video_path = "linemans_pliers_video.mpg"
    linemans_pliers.annotations.append(videoAnnotation)
    circle_annotation = Annotation()
    circle_annotation.type = "CircleAnnotation"
    circle_annotation.radius = 20
    circle_annotation.position = [84, 65]
    circle_annotation.color = [0, 255, 255]
    circle_annotation.thickness = 3
    linemans_pliers.annotations.append(circle_annotation)
    circle_annotation = Annotation()
    circle_annotation.type = "CircleAnnotation"
    circle_annotation.radius = 20
    circle_annotation.position = [47, 353]
    circle_annotation.color = [255, 255, 0]
    circle_annotation.thickness = 3
    linemans_pliers.annotations.append(circle_annotation)
    circle_annotation = Annotation()
    circle_annotation.type = "CircleAnnotation"
    circle_annotation.radius = 20
    circle_annotation.position = [125, 353]
    circle_annotation.color = [255, 0, 255]
    circle_annotation.thickness = 3
    linemans_pliers.annotations.append(circle_annotation)
    phs.append(linemans_pliers)

    needle_nose_pliers = PhysicalObject()
    needle_nose_pliers.name = "Needle Nose Pliers"
    needle_nose_pliers.image_path = "Needle Nose Pliers.jpg"
    needle_nose_pliers.image = cv2.imread(physical_objects_data_path + needle_nose_pliers.image_path)
    textAnnotation = Annotation()
    textAnnotation.type = "TextAnnotation"
    textAnnotation.text = "This is a Needle Nose Pliers"
    textAnnotation.position = [2, 2]
    needle_nose_pliers.annotations.append(textAnnotation)
    arrow = Annotation()
    arrow.type = "Arrow"
    arrow.start = [10.5, 20.9]
    arrow.end = [20.5, 30.5]
    needle_nose_pliers.annotations.append(arrow)
    videoAnnotation = Annotation()
    videoAnnotation.type = "VideoAnnotation"
    videoAnnotation.position = [40.5, 50.9]
    videoAnnotation.video_path = "needle_nose_pliers_video.mpg"
    needle_nose_pliers.annotations.append(videoAnnotation)
    circle_annotation = Annotation()
    circle_annotation.type = "CircleAnnotation"
    circle_annotation.radius = 20
    circle_annotation.position = [81, 95]
    circle_annotation.color = [255, 123, 123]
    circle_annotation.thickness = 3
    needle_nose_pliers.annotations.append(circle_annotation)
    circle_annotation = Annotation()
    circle_annotation.type = "CircleAnnotation"
    circle_annotation.radius = 20
    circle_annotation.position = [82, 16]
    circle_annotation.color = [123, 123, 123]
    circle_annotation.thickness = 3
    needle_nose_pliers.annotations.append(circle_annotation)
    phs.append(needle_nose_pliers)

    # frozen = jsonpickle.encode(phs)
    # thawed = jsonpickle.decode(frozen)

    return phs


def find_present_physical_objects(physical_objects, object_detection_results):
    present_objects = []
    for physical_object in physical_objects:
        for object_detection_result in object_detection_results:
            if physical_object.name == object_detection_result.lable:
                present_objects.append((physical_object, object_detection_result))
    return present_objects




def main():
    # feed the video/camera_feed into Yolo and get the bounding boxes of detected objects
    # crop the detected objects from Yolo output frame
    # find the corresponding object in the list of physical objects of the scene
    # for each scene physical object find features in the reference image and in the matching cropped image
    # use matcher to match the features
    # compute the transformation (affine, homography) from the set of matched features
    # apply transformation to the position of the annotations.


    physical_objects = create_dummy_physical_objects()
    logger.info("Created predefined physical objects.")
    load_yolo_model()
    logger.info("YOLO model loaded.")


    # predict every n-th frames
    # prediction_rate = 5
    prediction_rate = 2
    
    video_path = './test/tools_960x540.avi'
    # video_path = './test/tools.mp4'
    capture = cv2.VideoCapture(video_path)
    # capture = cv2.VideoCapture(0)

    # capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    colors = [tuple(255 * np.random.rand(3)) for i in range(10)]
    object_detection_results = []
    present_physical_objects = []
    best_homographies = {}

    num_processes = 10
    task_queue = mp.JoinableQueue()
    results_queue = mp.Queue()
    processes = [PoseEstimator(task_queue, results_queue) for i in range(num_processes)]
    for p in processes:
        p.start()

    logger.info("All pose estimator processes started. Starting OpenCV capture loop.")

    frame_number = 0
    while capture.isOpened():
        stime = time.time()
        ret, frame = capture.read()
        frame = cv2.resize(frame, (1920, 1080))
        if ret:
            try:
                frame_number += 1

                # frame = imutils.resize(frame, width=500)
                # frame = imutils.resize(frame, width=960)

                if int(frame_number % prediction_rate) == 0:
                    t1 = time.time()
                    object_detection_results = get_predictions(frame)
                    present_physical_objects = find_present_physical_objects(physical_objects, object_detection_results)
                    cropPredictedObjects(frame, object_detection_results)

                    t1 = time.time()

                    physical_object_names = [i[0].name for i in present_physical_objects]
                    # remove from best_homographies those objects that are no more present on the table
                    remove_from_best_homographies = list(set(best_homographies.keys()) - set(physical_object_names))
                    for name in remove_from_best_homographies:
                        del best_homographies[name]

                    for template, target in present_physical_objects:
                        best_homography = None
                        if template.name in best_homographies and \
                                best_homographies[template.name] is not None:
                            best_homography = best_homographies[template.name]

                        pe_input = PoseEstimationInput(template.name, template.image, target.image, best_homography)
                        task_queue.put(pe_input)

                    task_queue.join()
                    for i in range(len(physical_objects)):
                        if not results_queue.empty():
                            pe_output = results_queue.get()
                            if pe_output is None:
                                logger.info("pe_output is None.")
                                continue

                            best_homographies[pe_output.object_name] = pe_output

                    logger.debug("Computing homographies took %s", time.time() - t1)

                draw_results_on_frame(frame, colors, object_detection_results)
                draw_annotations(frame, present_physical_objects, best_homographies)
                cv2.imshow('frame', frame)
                logger.info('FPS {:.1f}'.format(1 / (time.time() - stime)))
            except Exception as exp:
                logger.info(str(exp))
                continue

        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    capture.release()


if __name__ == "__main__":
    logger = logging.getLogger('tracking2d')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('log/tracking2d.log')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s:%(lineno)d - %(funcName)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info('Logging is configured.')

    main()


