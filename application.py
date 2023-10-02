import copy
import argparse
import itertools
from collections import deque

from flask import Flask, render_template, Response
import cv2

import numpy as np
import mediapipe as mp

from model import KeyPointClassifier

app = Flask(__name__, template_folder='templates', static_folder='static')
camera = cv2.VideoCapture(0)


def get_args():
    """
    :return:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def generate_frames():
    """
    generate opencv frames
    """

    # Argument parsing #################################################################
    args = get_args()

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    while True:

        # Camera capture #####################################################
        success, frame = camera.read()
        frame = cv2.flip(frame, 1)  # Mirror display
        if not success:
            break
        # Copying the image to a different variable for processing
        debug_image = copy.deepcopy(frame)
        image = copy.deepcopy(frame)
        # Detection implementation #############################################################
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        # Landmark Detection and Info Determination ####################################################################
        if results.multi_hand_landmarks is not None:

            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)

                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                hand_info = handedness.classification[0].label[0:]
                gesture = "Not Assigned"

                # Setting gesture IDs for the detected gesture for display
                if (hand_info == "Right"):
                    if hand_sign_id == 0:  # open palm
                        gesture = "open palm"
                    if hand_sign_id == 1:  # fist
                        gesture = "fist"
                    if hand_sign_id == 2:  # index finger
                        gesture = "index finger"
                    if hand_sign_id == 3:  # index thumb joined and rest up i.e ok
                        gesture = "ok"
                if (hand_info == "Left"):
                    if hand_sign_id == 0:  # open palm
                        gesture = "open palm"
                    if hand_sign_id == 1:  # fist
                        gesture = "fist"
                    if hand_sign_id == 2:  # index finger
                        gesture = "index finger"
                    if hand_sign_id == 3:  # index thumb joined and rest up i.e ok
                        gesture = "ok"

                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(debug_image, brect, handedness, gesture)

        ret, buffer = cv2.imencode('.jpg', debug_image)
        debug_image = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + debug_image + b'\r\n')


def calc_bounding_rect(image, landmarks):
    """
    :param image:
    :param landmarks:
    :return:
    """
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    """
    :param image:
    :param landmarks:
    :return:
    """
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    """
    :param landmark_list:
    :return:
    """
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        """
        :param n:
        :return:
        """
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def draw_landmarks(image, landmark_point):
    """
    :param image:
    :param landmark_point:
    :return:
    """
    if len(landmark_point) > 0:
        # Thumb
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                 (255, 255, 255), 2)

        # Index finger
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                 (255, 255, 255), 2)

        # Middle finger
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                 (255, 255, 255), 2)

        # Ring finger
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                 (255, 255, 255), 2)

        # Little finger
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                 (255, 255, 255), 2)

        # Palm
        cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                 (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                 (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                 (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # wrist 1
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                       -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # wrist 2
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                       -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # thumb: base
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                       -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # Thumb: 1st joint
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                       -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # thumb: fingertip
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                       -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # index finger: base
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                       -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # index finger: second joint
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                       -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # Index finger: 1st joint
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                       -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # index finger: fingertip
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                       -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # middle finger: base
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                       -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # middle finger: second joint
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                       -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # Middle finger: 1st joint
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                       -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # middle finger: fingertip
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                       -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # ring finger: base
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                       -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # ring finger: second joint
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                       -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # Ring finger: 1st joint
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                       -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # ring finger: fingertip
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                       -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # little finger: base
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                       -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # little finger: 2nd joint
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                       -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # little finger: 1st joint
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                       -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # little finger: fingertip
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                       -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    """
    :param use_brect:
    :param image:
    :param brect:
    :return:
    """
    if use_brect:
        # Outer rectangle
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                      (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text):
    """
    :param image:
    :param brect:
    :param handedness:
    :param hand_sign_text:
    :return:
    """
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                  (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return image


@app.route('/')
def index():
    """
    route traffic
    :return:
    """
    return render_template('index.html')


@app.route('/video')
def video():
    """
    video output
    :return:
    """
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)

