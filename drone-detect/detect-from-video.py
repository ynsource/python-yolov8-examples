import cv2
import math
import cvzone
from ultralytics import YOLO

SAVE_RESULT_AS_VIDEO = False
RESULT_FILE_NAME = "result.mp4"

def process_image(img, model=YOLO('model.pt'), stream=True):
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
            cdist = int(math.sqrt((icx - cx) ** 2 + (icy - cy) ** 2))
            
            # confidency
            conf = math.ceil((box.conf[0] * 100)) / 100
            
            # detected/predicted class name
            cls = result.names[int(box.cls[0])]

            # draw object box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 1)

            # put class and confidency on object in image
            cvzone.putTextRect(img, f'{cls} {conf}', (max(0, x1+5), max(0, y1 - 15)), scale=1, thickness=1)

            # draw a circle in the center of image
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            # put distance bottom of the object box
            cvzone.putTextRect(img, f'{cdist}', (max(0, x1+5), min(ih - 25, y2 + 25)), scale=1, thickness=1)

            # draw a circle in the center of object
            cv2.circle(img, (icx, icy), 5, (255, 0, 255), cv2.FILLED)

            # draw borders with %10 inset
            cv2.rectangle(img, (int(iw * 0.1), int(ih * 0.1)), (int(iw * 0.9), int(ih * 0.9)), (255, 0, 255), 1)

            # draw a line from image center to object center
            cv2.line(img, (icx, icy), (cx, cy), (255, 0, 255), 1)


videoCapture = cv2.VideoCapture('example-videos/01.mp4')
videoWriter = None

while videoCapture.isOpened():
    success, img = videoCapture.read()
    if success:
        process_image(img)

        if SAVE_RESULT_AS_VIDEO:
            if videoWriter is None:
                cv2_fourcc = cv2.VideoWriter.fourcc(*'mp4v')
                videoWriter = cv2.VideoWriter(RESULT_FILE_NAME, cv2_fourcc, 24, img.shape[:2][::-1])

            videoWriter.write(img)
        else:
            # show result in opencv window
            cv2.imshow("Image", img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    else:
        break

if SAVE_RESULT_AS_VIDEO:
    videoWriter.release()
else:    
    # destroy opencv window
    cv2.destroyAllWindows()

print("Video processing completed!")
