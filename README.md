# ? YOLOv7-onnx-inference

### ? Installation
```bash
git clone https://github.com/iwannamaltese/YOLOv7-onnx-inference.git
cd YOLOv7-onnx-inference

conda create -n yolov7_inf python=3.9
conda activate

pip install -r requirements.txt
```

### ? Usage
1. Object Detection
```bash
python det_inf.py --input "input_image_path" --weights "yolov7-tiny.onnx" --size 640 --line --text

python det_inf.py -i "input_image_path" -w "yolov7-tiny.onnx" -s 640 -l -t
```

2. Segmentation
```bash
python seg_inf.py --input "input_image_path" --weights yolov7-seg.onnx" --size 640 --data "data.yaml" --device 0

python seg_inf.py -i "input_image_path" -w "yolov7-seg.onnx" -s 640 -dt "data.yaml" -dv 0
```

추가 작성 예정