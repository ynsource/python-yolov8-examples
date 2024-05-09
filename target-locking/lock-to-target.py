import cv2
import math
import cvzone
import logging
from ultralytics import YOLO

# GLOBAL SETTINGS
CAMERA_INDEX = 0                        # use Nth camera as capture device
FLIP_CAMERA = True                      # flip camera on x axis (useful for front cameras)
DETECTION_CLASS = "person"              # lock to this object
USE_YOLOV8_MODEL = "yolov8n.pt"         # use this pretrained YOLOv8 model

# CAMERA FEATURES
H_FOV = 75.5                            # horizontal angle of view
V_FOV = (1080/1920) * H_FOV             # vertical angle of view

# supress yolov8 info logs
logging.getLogger("ultralytics").setLevel(logging.ERROR)

def calculate_angles(dx, dy, iw, ih):
    # add or remove (-) to match servo angles
    dpan = (dx/iw)*H_FOV                    # angle on horizontal axis
    dtilt = -(dy/ih)*V_FOV                  # angle on vertical axis
    return (round(dpan), round(dtilt))


def process_image(img, model=YOLO(USE_YOLOV8_MODEL), stream=True):
    # find objects in image by using model
    results = model(img, stream)

    for result in results:
        for box in result.boxes:
            # object box
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            
            # object center
            cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
            
            # image width and height
            iw, ih = img.shape[:2][::-1]
            
            # center coordinates of image
            icx, icy = iw // 2, ih // 2
            
            # center distances (Euclidian distance)
            dx, dy = (cx - icx), (cy - icy)
            cdist = int(math.sqrt(dx ** 2 + dy ** 2))
            
            # confidency
            conf = math.ceil((box.conf[0] * 100)) / 100
            
            # detected/predicted class name
            cls = result.names[int(box.cls[0])]

            # if class detected with at least %50 accuracy
            if cls == DETECTION_CLASS and conf >= 0.5:
                # calculate angles
                dpan, dtilt = calculate_angles(dx, dy, iw, ih)

                # draw object box
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 1)

                # put class and confidency on object in image
                cvzone.putTextRect(img, f'{cls} {conf}', (max(0, x1+5), max(0, y1 - 15)), scale=1, thickness=1)

                # draw a circle in the center of image
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                # put angles bottom of the object box
                cvzone.putTextRect(img, f'{dpan}, {dtilt} deg', (max(0, x1+5), min(ih - 25, y2 + 25)), scale=1, thickness=1)

                # draw a circle in the center of object
                cv2.circle(img, (icx, icy), 5, (255, 0, 255), cv2.FILLED)

                # draw borders with %10 inset
                cv2.rectangle(img, (int(iw * 0.1), int(ih * 0.1)), (int(iw * 0.9), int(ih * 0.9)), (255, 0, 255), 1)

                # draw a line from image center to object center
                cv2.line(img, (icx, icy), (cx, cy), (255, 0, 255), 1)

videoCapture = cv2.VideoCapture(0)

while videoCapture.isOpened():
    success, img = videoCapture.read()
    if success:
        img = cv2.flip(img, 1)
        process_image(img)

        # show result in opencv window
        cv2.imshow("Image", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    else:
        break


# destroy opencv window
cv2.destroyAllWindows()
