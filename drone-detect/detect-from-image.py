import math
import cv2
import cvzone
from ultralytics import YOLO


def process_image(img, model=YOLO('model.pt'), stream=False):
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


if __name__ == '__main__':
    
    # read image
    img = cv2.imread("example-images/01.jpg")

    # process image and try to find object
    process_image(img)

    # show result in opencv window
    cv2.imshow("Image", img)

    # wait to press any key
    cv2.waitKey(0)
    
    # destroy opencv window
    cv2.destroyAllWindows() 
