# YOLOv8 Examples in Python

This repo contains YOLOv8 examples such as object detection, object tracking etc.

See also: https://github.com/ultralytics/ultralytics

# Installing YOLOv8
To install YOLOv8 Python packages and CLI tool open a terminal and run:
```
pip install ultralytics
```

# Double Check PATH
To use YOLOv8 CLI Tool Python Scripts folder should be added to PATH.
For Windows (Python Version is 3.11): `%APPDATA%\Python\Python311\Scripts`

# YOLOv8 Pretrained Models

Instead of model.pt that trained for drones only you can type any YOLOv8 model
```
yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt or yolov8x.pt
```

See https://docs.ultralytics.com/tasks/detect/#models

You don't need to manually download this model files they will be downloaded automatically from YOLOv8.

But this examples uses trained model for drone detection.

# CLI Predict Images
```
yolo predict model=model.pt source="example-images/01.jpg"
```
or multiple
```
yolo predict model=model.pt source="example-images/\*.jpg"
```

# CLI Predict Videos
```
yolo predict model=model.pt source="example-videos/01.mp4"
```
or multiple
```
yolo predict model=model.pt source="example-videos/\*.mp4"
```

# CLI Traning Model
After you select and prepare datasets (e.g. upload any dataset and then download for YOLOv8 from RoboFlow)
you can train the model with this command.

```
yolo task=detect mode=train model=yolov8n.pt data=dataset-folder/data.yaml epochs=20 imgsz=640
```

It should fail if you don't edit .yaml file which has relative file paths. If so you should change
test: ..., train: ... paths with absolute paths. E.g. for Google Colab it should be `/content/My-Dataset/test`
for test folder instead of `./My-Dataset/test` or `My-Dataset/test`. 

After this task completed you can find trained models `best.pt` and `last.pt` in `runs/detect/weights` folder.

It is recommended to run training tasks in Google Colab or another service that ensures TPU or GPU. If your
computer doesn't have CUDA with NVIDIA GPU or any supported TPU training task will be run at CPU and process
will be too much slow.

# Results
You can see results in `runs/detect/predictX` folders after CLI command completed.

# Python Examples
You can found python examples in folders next to this file.

# License
The Unlicense. Feel free to use or change it how you need.
But third party sources like pictures, videos etc. may have some limitations.
If you have doubts, please check out the links we attributed.
