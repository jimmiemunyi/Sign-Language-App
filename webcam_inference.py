from collections import deque, Counter

import cv2
from fastai.vision.all import *

print('Loading our Inference model...')
# load our inference model
inf_model = load_learner('model/sign_language.pkl')
print('Model Loaded')


# define a deque to get rolling average of predictions
# I go with the last 20 predictions
rolling_predictions = deque([], maxlen=10)

# get the most common item in the deque
def most_common(D):
    data = Counter(D)
    return data.most_common(1)[0][0]


def hand_area(img):
    # img = cv2.flip(img, flipCode = 1)
    # specify where hand should go
    hand = img[50:324, 50:324]
    # the images in the model were trainind on 200x200 pixels
    hand = cv2.resize(hand, (200,200))
    return hand

# capture video on the webcam
cap = cv2.VideoCapture(0)


# get the dimensions on the frame
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# define codec and create our VideoWriter to save the video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output/sign-language.mp4', fourcc, 12, (frame_width, frame_height))


# read video
while True:
    # capture each frame of the video
    ret, frame = cap.read()

    # flip frame to feel more 'natural' to webcam
    frame = cv2.flip(frame, flipCode = 1)

    # draw a blue rectangle where to place hand
    cv2.rectangle(frame, (50, 50), (324, 324), (255, 0, 0), 2)

    # get the image
    inference_image = hand_area(frame)

    # get the current prediction on the hand
    pred = inf_model.predict(inference_image)
    # append the current prediction to our rolling predictions
    rolling_predictions.append(pred[0])

    # our prediction is going to be the most common letter
    # in our rolling predictions
    prediction_output = f'The predicted letter is {most_common(rolling_predictions)}'

    # show predicted text
    cv2.putText(frame, prediction_output, (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    # show the frame
    cv2.imshow('frame', frame)
    # save the frames to out file
    out.write(frame)


    # press `q` to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release VideoCapture()
cap.release()
# release out file
out.release()
# close all frames and video windows
cv2.destroyAllWindows()
