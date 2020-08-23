# Markerless Tracking of Objects on a Tabletop

As part of [ISAR](https://github.com/zardosht/isar), I wrote code for tracking of the objects put on the tabletop. The tracking code uses Yolo V2 for object detection. Pose estimation is done using feature matching (AKAZE features) and OpenCV's estimateAffineTransform. Pose refinement is done through image alignment using Enhanced Correlation Coefficient (ECC).

![Demo](./demo.gif)

Video: https://www.youtube.com/watch?v=FpB5MsbWenY

## Steps
Following steps are performed at each frames: 

1. Send frame to object detection (Yolo) and get the bounding box of the objects 
2. Crop the object images at bounding boxes
3. For each cropped image, extract features from cropped image and the corresponding template image of the object
4. Do feature matching to find correspondence points
5. Calculate the transformation (homography, affine transform) from matches
6. Check if the newly calculated transformation is better than the best transformation so far. If yes, return the newly calculated transformation, otherwise return the best transformation. 
   
Pose estimation is done in parallel for each pair of physical object images (cropped image, template image), using processes. Multiprocessing Queue as used to communicate between the main thread and pose estimation processes. 


