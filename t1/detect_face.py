
import time
import cv2
import numpy as np
import imutils
import os



def save_the_face(file_path):
  frame = cv2.imread(file_path)

  protoPath = "face_detector/deploy.prototxt"
  modelPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
  net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
  frame = imutils.resize(frame, width=600)
  FACE_IMAGE_PART = None
  # grab the frame dimensions and convert it to a blob
  (h, w) = frame.shape[:2]
  blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
      (300, 300), (104.0, 177.0, 123.0))
  net.setInput(blob)
  detections = net.forward()

  # loop over the detections
  for i in range(0, detections.shape[2]):
      # extract the confidence (i.e., probability) associated with the
      # prediction
      confidence = detections[0, 0, i, 2]
      #if confidence > .4: print(confidence)
      if confidence > .6: #args["confidence"]
          # compute the (x, y)-coordinates of the bounding box for
          # the face and extract the face ROI
          box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
          (startX, startY, endX, endY) = box.astype("int")

          # ensure the detected bounding box does fall outside the
          # dimensions of the frame
          startX = max(0, startX)
          startY = max(0, startY)
          endX = min(w, endX)
          endY = min(h, endY)

          # extract the face ROI and then preproces it in the exact
          # same manner as our training data
          face = frame[startY:endY, startX:endX]
          try:
              FACE_IMAGE_PART = cv2.resize(face, (256, 256))
              #face = cv2.resize(face, (256, 256))
              os.makedirs('face') if not os.path.exists('face') else ''
              timestamp = int(time.time())  # Generate a unique timestamp
              image_filename = f"face/face_{timestamp}_{i}.jpg"  # Create a filename with timestamp
              cv2.imwrite(image_filename, FACE_IMAGE_PART)  # Save the frame as an image
          except:
              break