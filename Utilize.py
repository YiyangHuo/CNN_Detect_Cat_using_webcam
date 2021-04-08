#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 4/7/21 12:33 AM
#@Author: Yiyang Huo
#@File  : Utilize.py

import cv2
from PIL import Image
import numpy as np
from keras.models import Sequential, load_model


IMAGE_SIZE = 256
font = cv2.FONT_HERSHEY_SIMPLEX

# org
org = (50, 50)

# fontScale
fontScale = 4

# Blue color in BGR
color = (255, 0, 0)

# Line thickness of 2 px
thickness = 3

# Using cv2.putText() method



if __name__ == "__main__":
    # define a video capture object
    vid = cv2.VideoCapture(0)
    model = load_model("model2.h5")


    while (True):

        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        (h, w) = frame.shape[:2]
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(frame)
        pil_im = pil_im.convert('L')
        pil_im = pil_im.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
        sample_to_predict = np.array(pil_im).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
        predictions = model.predict(np.array(sample_to_predict))
        # [1,0] means that it is cat
        if predictions[0][0] >0.9 and predictions[0][0] < 1.1 and predictions[0][1] < 0.1 and predictions[0][1] > -0.1:

            frame = cv2.putText(frame, 'There is a cat', (w//16, h//2), font,
                                fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        print(predictions)
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()