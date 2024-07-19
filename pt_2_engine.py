from ultralytics import RTDETR as YOLO
import argparse
from pathlib import Path 
import os
import cv2

parser = argparse.ArgumentParser(description="input import arguments for pytorch model to trt model")
parser.add_argument("--model", type=str, required=True, help="input pytorch model to convert")
parser.add_argument("--dynamic", type=bool, required=False, default=False, help="dynamic input size")
parser.add_argument("--batch", type=int, required=False, default=1, help="batch size")
parser.add_argument("--workspace", type=int, required=False, default=4, help="workspace")
parser.add_argument("--int8", type=bool, required=False, default=False)
parser.add_argument("--fp16", type=bool, required=False, default=True)
parser.add_argument("--data", type=str, required=False, default=None, help="data is the yaml file which is used to calibrate the int8 model")
parser.add_argument("--image", type=str, required=True)
parser.add_argument("--imgsz", type=int, required=True, default="model_input_size")

args = parser.parse_args()
print(args)
# exit(0)
model = YOLO(args.model)

coco_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# 示例检测结果

model.export(format="engine", dynamic=args.dynamic, batch=args.batch, workspace=args.workspace, 
        int8=args.int8, half=args.fp16, data=args.data, imgsz=args.imgsz)

ptfile = str(Path(args.model).name)
print("Training:",model.training)
print(ptfile)
engine_file = ptfile[:ptfile.rfind('.')]
origin_engine_file = engine_file + ".engine"
if args.int8:
    engine_file = engine_file + "_int8"
elif args.fp16:
    engine_file = engine_file + "_fp16"
else:
    engine_file = engine_file + "_fp32"

engine_file = engine_file + f"_{args.imgsz}" +  ".engine"

# rename
if os.path.isfile(origin_engine_file):
    os.rename(origin_engine_file, engine_file)

if os.path.isfile(engine_file):
    print(f"engine file {engine_file} on the way.")
    # model = YOLO(engine_file)

    # Run inference
    results = model.predict(args.image, imgsz=args.imgsz, conf=0.01)
    results = [result.cpu() for result in results]
    image = cv2.imread(args.image)
    print("image width x heihgt:", image.shape[1], "x", image.shape[0])
    for i, result in enumerate(results):
        boxes = result.boxes
        cls = boxes.cls
        conf = boxes.conf
        xyxy = boxes.xyxy
        for j in range(len(cls)):
            if conf[j] > 0.2:
                x1, y1, x2, y2 = xyxy[j]
                x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2) 

                label = f"{coco_classes[int(cls[j])]}: {conf[j]:.2f}"
                # print("label:", label)
                print(j+1, int(cls[j]), float(conf[j]), x1, y1, x2, y1, x2,y2, x1, y2, (x1+x2)/2.0, (y1+y2)/2.0)

                cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    cv2.imwrite("result.jpg", image)
else:
    print(f"Can not find {engine_file}")
