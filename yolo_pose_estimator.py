import logging
import multiprocessing as mp
import random

import cv2
import numpy as np
import time
from yolo_tracker_classes import PoseEstimationOutput


logger = logging.getLogger('tracking2d.yolo_pose_estimator')
DEBUG = True
LOG_IMAGES_PATH = "./log/"

class PoseEstimator(mp.Process):
    """
    PoseEstimator
    """

    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = 0.90
    ransac_reprojection_threshold = 5
    recompute_homography_using_only_inliers = False

    # recompute_homography_using_ECC = False
    recompute_homography_using_ECC = True
    recompute_homography_using_ECC_threshold_min = 40
    recompute_homography_using_ECC_threshold_max = 70

    def __init__(self, task_queue, result_queue):
        mp.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        while True:
            pe_input = self.task_queue.get()

            if pe_input is None:
                logger.info("Received None pose_estimation_input. Shutting down.")
                self.task_queue.task_done()
                break

            # compute pose from template and target images
            # put a PoseEstimationOutput instance int the results queue
            t1 = time.time()
            estimated_pose = None
            try:
                estimated_pose = self.find_best_homography(pe_input.template_image, pe_input.target_image, pe_input.best_homography)
                estimated_pose.object_name = pe_input.object_name
            except Exception as exp:
                logger.info(str(exp))

            self.task_queue.task_done()
            self.result_queue.put(estimated_pose)
            print("Finding best homograpy for {} took {}".format(pe_input.object_name, time.time() - t1))

        return

    def find_best_homography(self, physical_object_image, cropped_image, best_pe):
        pe_result = self.compute_homography(physical_object_image, cropped_image)
        pe_result.error = self.compute_error(pe_result.homography, physical_object_image, cropped_image)
        # Keep track of the best pe_result. If the newly computed one is better, replace the best with it.
        # See compute_quality_score(). I think a better approach would be compute the quality of a pe_result using reprojection error.
        if best_pe is not None:
            best_pe.error = self.compute_error(best_pe.homography, physical_object_image, cropped_image, "_best")

        if self.recompute_homography_using_ECC:
            better_pe = best_pe if best_pe is not None and best_pe.error < pe_result.error else pe_result
            if self.recompute_homography_using_ECC_threshold_min < better_pe.error < self.recompute_homography_using_ECC_threshold_max:
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-6)
                physical_object_image_gray, cropped_image_gray = self.convert_to_gray_scale(physical_object_image, cropped_image)
                better_homography_float32 = better_pe.homography.astype(np.float32)

                # (cc, h_ECC) = cv2.findTransformECC(physical_object_image_gray, cropped_image_gray, better_homography_float32,
                #                                    cv2.MOTION_HOMOGRAPHY, criteria)

                (cc, h_ECC) = cv2.findTransformECC(physical_object_image_gray, cropped_image_gray, better_homography_float32,
                                                   cv2.MOTION_AFFINE, criteria)

                error_ecc = self.compute_error(h_ECC, physical_object_image, cropped_image, "_ecc")
                if h_ECC is not None and error_ecc < better_pe.error:
                    pe_result.homography = h_ECC
                    pe_result.error = error_ecc

        epsilon = 5e0
        if best_pe is None or pe_result.error < best_pe.error:
            return pe_result
        elif np.abs(pe_result.error - best_pe.error) < epsilon:
            return random.choice((pe_result, best_pe))
        else:
            return best_pe

    def compute_homography(self, physical_object_image, cropped_image):
        # physicalObjectImage, croppedImage = convertToGrayScale(physicalObjectImage, croppedImage)

        # phys_obj_keypoints, phys_obj_descriptors, cropped_image_keypoints, cropped_image_descriptors \
        #     = extractFeaturePoints(physicalObjectImage, croppedImage, algorithm='SURF')
        # phys_obj_keypoints, phys_obj_descriptors, cropped_image_keypoints, cropped_image_descriptors \
        #     = extractFeaturePoints(physicalObjectImage, croppedImage, algorithm='SIFT')
        # phys_obj_keypoints, phys_obj_descriptors, cropped_image_keypoints, cropped_image_descriptors \
        #     = extractFeaturePoints(physicalObjectImage, croppedImage, algorithm='ORB')
        phys_obj_keypoints, phys_obj_descriptors, cropped_image_keypoints, cropped_image_descriptors \
            = self.extract_feature_points(physical_object_image, cropped_image, algorithm='AKAZE')

        # matches = findMatches(phys_obj_descriptors, cropped_image_descriptors, algorithm='BRUTE_FORCE_L1')
        matches = self.find_matches(phys_obj_descriptors, cropped_image_descriptors, algorithm='BRUTE_FORCE_HAMMING', ratio_test=False)
        # matches = findMatches(phys_obj_descriptors, cropped_image_descriptors, algorithm='BRUTE_FORCE_HAMMING', ratioTest=True)
        # matches = findMatches(phys_obj_descriptors, cropped_image_descriptors, algorithm='FLANN')

        # Draw top matches
        im_matches = cv2.drawMatches(physical_object_image, phys_obj_keypoints, cropped_image, cropped_image_keypoints, matches, None)
        if DEBUG: cv2.imwrite(LOG_IMAGES_PATH + "matches.jpg", im_matches)

        points1, points2 = self.find_points_from_matches(phys_obj_keypoints, cropped_image_keypoints, matches)

        # h1, mask = cv2.findHomography(points1, points2, cv2.RANSAC, ransac_reprojection_threshold)

        h1, mask = cv2.estimateAffine2D(points1, points2, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_reprojection_threshold)

        # error = compute_error(h1, physical_object_image, cropped_image)
        homography_result = PoseEstimationOutput(None, h1, None)

        if self.recompute_homography_using_only_inliers:
            # get the inlier matches (list of tuples of points)
            # run find homography again using only the inliers this time instead of LEMDS or RHO instead of RANSAC
            boolean_inliers_mask = (mask > 0)
            inlier_points1 = points1[boolean_inliers_mask.repeat(2, axis =1)].reshape((-1, 2))
            inlier_points2 = points2[boolean_inliers_mask.repeat(2, axis =1)].reshape((-1, 2))
            # h2, mask = cv2.findHomography(inlier_points1, inlier_points2, cv2.LMEDS)
            h2, mask = cv2.estimateAffine2D(inlier_points1, inlier_points2, cv2.LMEDS)
            # prosac_reprojection_error = 2.5
            # h2, mask = cv2.findHomography(inlier_points1, inlier_points2, cv2.RHO, prosac_reprojection_error)
            if h2 is not None:
                homography_result.homography = h2
                # homography_result.error = compute_error(h2, physical_object_image, cropped_image)

        logger.debug("Homography: %s", homography_result.homography)
        return homography_result

    @staticmethod
    def convert_to_gray_scale(*images):
        result = []
        for image in images:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = gray.astype(np.float32)
            gray = gray / 255
            result.append(gray)
        return tuple(result)

    def extract_feature_points(self, physical_object_image, cropped_image, algorithm='SURF'):
        feature_extractor = None
        if algorithm == 'SURF':
            feature_extractor = cv2.xfeatures2d.SURF_create()
        elif algorithm == 'SIFT':
            feature_extractor = cv2.xfeatures2d.SIFT_create()
        elif algorithm == 'ORB':
            feature_extractor = cv2.ORB_create(self.MAX_FEATURES)
        elif algorithm == 'AKAZE':
            feature_extractor = cv2.AKAZE_create(threshold=1e-4)

        physical_object_key_points, physical_object_descriptors = feature_extractor.detectAndCompute(physical_object_image, None)
        cropped_image_key_points, cropped_image_descriptors = feature_extractor.detectAndCompute(cropped_image, None)

        return physical_object_key_points, physical_object_descriptors, cropped_image_key_points, cropped_image_descriptors

    def find_matches(self, physical_object_descriptors, cropped_image_descriptors, algorithm=None, ratio_test=False):
        matcher = None
        if algorithm == 'FLANN':
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        elif algorithm == 'BRUTE_FORCE_HAMMING':
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            # matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        elif algorithm == 'BRUTE_FORCE_L1':
            matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_L1)

        good_matches = []
        if ratio_test:
            matches = matcher.knnMatch(physical_object_descriptors, cropped_image_descriptors, k=2)
            for m, n in matches:
                if m.distance < 0.5 * n.distance:
                    good_matches.append(m)
        else:
            matches = matcher.match(physical_object_descriptors, cropped_image_descriptors)
            matches.sort(key=lambda x: x.distance, reverse=False)
            num_good_matches = int(len(matches) * self.GOOD_MATCH_PERCENT)
            good_matches = matches[:num_good_matches]

        logger.debug("Number of good matches: %s", len(good_matches))
        return good_matches

    @staticmethod
    def find_points_from_matches(physical_object_key_points, cropped_image_key_points, matches):
        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = physical_object_key_points[match.queryIdx].pt
            points2[i, :] = cropped_image_key_points[match.trainIdx].pt

        return points1, points2

    @staticmethod
    def get_gradient(im):
        # Calculate the x and y gradients using Sobel operator
        grad_x = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3)

        # Combine the two gradients
        grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
        return grad

    @staticmethod
    def compute_error(homography, physical_object_image, cropped_image, debug_postfix=""):
        height, width, _ = cropped_image.shape

        # warpped_original = cv2.warpPerspective(physical_object_image, homography, (width, height))
        warpped_original = cv2.warpAffine(physical_object_image, homography, (width, height), flags = cv2.INTER_LINEAR)

        warpped_original[warpped_original == 0] = 255
        if DEBUG: cv2.imwrite(LOG_IMAGES_PATH + "warpped_original" + debug_postfix + ".jpg", warpped_original)

        difference_image = np.abs(cropped_image.astype("float32") - warpped_original.astype("float32"))
        difference_image = cv2.cvtColor(difference_image, cv2.COLOR_BGR2GRAY)

        # difference_image = cv2.normalize(difference_image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

        # # difference_image = get_gradient(difference_image)
        # # difference_image = cv2.erode(difference_image, np.ones((3, 3), np.uint8), iterations=1)
        # # difference_image = cv2.dilate(difference_image, np.ones((3,3),np.uint8), iterations=1)
        #
        # morph_elem = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        # # difference_image = cv2.morphologyEx(difference_image, cv2.MORPH_CLOSE, morph_elem)
        # # difference_image = cv2.morphologyEx(difference_image, cv2.MORPH_TOPHAT, morph_elem)
        # difference_image = cv2.morphologyEx(difference_image, cv2.MORPH_BLACKHAT, morph_elem)
        # # ret, difference_image = cv2.threshold(difference_image, 100, 255, cv2.THRESH_BINARY)
        # difference_image = cv2.adaptiveThreshold(difference_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        #
        # morph_elem = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # difference_image = cv2.morphologyEx(difference_image, cv2.MORPH_OPEN, morph_elem)
        # # difference_image = cv2.morphologyEx(difference_image, cv2.MORPH_TOPHAT, morph_elem)

        if DEBUG: cv2.imwrite(LOG_IMAGES_PATH + "difference_image" + debug_postfix + ".jpg", difference_image)

        # error = (1 / (height * width)) * np.sum(difference_image ** 2)
        error = np.sqrt((1 / (height * width)) * np.sum(difference_image ** 2))

        # cropped_image_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        # warpped_original_gray = cv2.cvtColor(warpped_original, cv2.COLOR_BGR2GRAY)
        # similarity_score = ssim(cropped_image_gray, warpped_original_gray, gradient=False)
        # error = (1 - similarity_score) / 2

        return error
