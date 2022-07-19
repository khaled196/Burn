import numpy as np
import cv2
import sys
import os
import tqdm

os.chdir(sys.argv[2])
i=0
path=sys.argv[1]
for img in os.listdir(path):
  try:
    i+=1
    image = cv2.imread(os.path.join(path,img))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 110, 0])
    upper = np.array([179, 240, 255])
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(image, image, mask=mask)
    #cv2.imshow('result', result)
    cv2.imwrite('extracted_burn_skin'+str(i)+'.png', result)
    cv2.waitKey(0)
  except Exception as e:  # in the interest in keeping the output clean...
    pass

