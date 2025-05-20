import os
import cv2
import time
import math
import argparse
import datetime
import warnings
import numpy as np
import onnxruntime as ort
from PIL import Image
from tqdm import tqdm
import random
warnings.filterwarnings('ignore')

def letterBox(im, new_shape, auto=True, scaleup=True, stride=32):
    shape = im.shape[:2]

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)

    # divide padding into 2 sides
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=None)

    return im, r, (dw, dh)


def xyxy2xywh(box, img_size):
    x1, y1, x2, y2 = box
    img_width, img_height = img_size

    x = ((x1 + x2) / 2) / img_width
    y = ((y1 + y2) / 2) / img_height
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height

    result = [x,y,w,h]

    return result


def agnosticNms(predictions, conf_thres, iou_thres):
    """
    Agnostic NMS: Class-agnostic Non-Maximum Suppression
    :param predictions: ndarray of shape (N, 7) [batch_id, x0, y0, x1, y1, cls_id, score]
    :param conf_thres: Confidence threshold
    :param iou_thres: IoU threshold for NMS
    :return: Filtered predictions after NMS
    """
    predictions = predictions[predictions[:, 6] > conf_thres]
    if len(predictions) == 0:
        return []
    
    predictions = predictions[np.argsort(-predictions[:, 6])]

    selected_boxes = []
    
    while len(predictions) > 0:
        chosen_box = predictions[0]
        selected_boxes.append(chosen_box)

        ious = calculateIou(chosen_box[1:5], predictions[:, 1:5])
        predictions = predictions[ious < iou_thres]

    return np.array(selected_boxes)


def calculateIou(box1, boxes):
    """
    Calculate Intersection over Union (IoU)
    :param box1: (x0, y0, x1, y1) coordinates of the first box
    :param boxes: ndarray of shape (N, 4) for other boxes
    :return: IoU values
    """

    inter_x0 = np.maximum(box1[0], boxes[:, 0])
    inter_y0 = np.maximum(box1[1], boxes[:, 1])
    inter_x1 = np.minimum(box1[2], boxes[:, 2])
    inter_y1 = np.minimum(box1[3], boxes[:, 3])
    
    inter_area = np.maximum(inter_x1 - inter_x0, 0) * np.maximum(inter_y1 - inter_y0, 0)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    union_area = area1 + areas - inter_area
    
    return inter_area / np.maximum(union_area, 1e-6)


def onnxRuntime():
    path = args.input
    img_size = args.size
    conf = args.conf
    iou = args.iou
    w = args.weights
    inf_time = args.inference_time
    MYTIME = 0

    cuda = True

    # 경로가 없다면 생성
    createOSDir(path)
        
    print("Inference Model : ", w)

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
    session = ort.InferenceSession(w, providers=providers)
    print("Onnxruntime Session Providers", session.get_providers())
    print("Selected Graphic Card: ", ort.get_device())

    # 폴더 탐색을 위한 이미지 파일 리스트 생성
    img_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):  # 이미지 확장자 추가
                img_files.append(os.path.join(root, file))

    names = ['Rectangle_Lead', 'Polygonal)Lead', 'TR_Lead (Big)', 'TR_Lead (Small)']
    colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}
    start = time.time()
    missing_count = 0

    for idx in tqdm(img_files, desc="Processing..", unit="file", colour="#4caf50"):
        pre_start = time.time()
        img = cv2.imread(idx)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = img.copy()
        image, ratio, dwdh = letterBox(image, (img_size, img_size), auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)

        im = image.astype(np.float32)
        im /= 255
        pre_end = time.time()

        outname = [i.name for i in session.get_outputs()]
        inname = [i.name for i in session.get_inputs()]

        inp = {inname[0]:im}
        inference_start = time.time()
        outputs = session.run(outname, inp)[0]
        inference_end = time.time()

        MYTIME += inference_end - inference_start

        if len(outputs) == 0:
            missing_count += 1

        if inf_time:
            print("Speed: {}ms pre-process, {}ms inference".format(pre_end - pre_start, inference_end - inference_start))

        agnostic = args.agnostic_nms
        line = args.line
        text = args.text
        if agnostic == True:
            # Apply agnostic NMS
            outputs = agnosticNms(outputs, conf, iou)

        ori_images = [img.copy()]

        for _, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(outputs):
            box = np.array([x0, y0, x1, y1])
            box -= np.array(dwdh * 2)
            box /= ratio
            class_id = int(cls_id)
            name = names[class_id]
            color = colors[name]
            yolo_box = xyxy2xywh(box, (img_size, img_size))
            yolo_box = np.round(yolo_box, 6)
            yolo_box = list(map(str, yolo_box))

            if line == True:
                box = box.round().astype(np.int32).tolist()
                image = ori_images[int(batch_id)]
                cv2.rectangle(image, box[:2], box[2:], (255, 0, 0), 1, cv2.LINE_AA)

            if text == True:
                cv2.putText(image, name, (box[0], box[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), thickness=1)
                cv2.putText(image, "{:.2f}".format(score), (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), thickness=1)


        Image.fromarray(ori_images[0])
        ori_images[0] = cv2.cvtColor(ori_images[0], cv2.COLOR_BGR2RGB)

        # 현재 이미지의 상위 폴더에 _ai 폴더 생성
        parent_dir = os.path.dirname(idx)  # 상위 폴더 경로
        output_folder = parent_dir + "_result"  # _ai 폴더 경로
        createOSDir(output_folder)  # _ai 폴더가 없으면 생성

        # 이미지 저장
        cv2.imwrite(os.path.join(output_folder, os.path.basename(idx)), ori_images[0])

    
    end = time.time()
    process_time = end - start

    print("Missing Count : {}".format(missing_count))
    print("Total Inference Time : {} seconds".format(round(process_time, 3)))
    print("Complete!\n")
    # print("Saved As {}".format(output_path))

    print("MYTIME", MYTIME)


def createOSDir(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)
    except OSError as e:
        print(e)


def rotationPoint(x, y, angle, cx, cy):
    ''' 회전된 좌표에 대한 x, y 값을 반환 '''
    theta = math.radians(angle)     
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    nx = cos_theta * (x - cx) - sin_theta * (y - cy) + cx
    ny = sin_theta * (x - cx) + cos_theta * (y - cy) + cy

    return nx, ny


def utcNow():
    ''' 현재 시간을 반환 (KR) '''
    dt_kst = datetime.datetime.utcnow() + datetime.timedelta(hours=9)
    date_time = dt_kst.strftime('%m%d%H%M%S')

    return date_time


def main(args):
    onnxRuntime()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create OCR Rotation Data with Coordinate txt File")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input Directory Path")
    parser.add_argument("--weights", "-w", type=str, required=False, help="Input Custom Weight Model")
    parser.add_argument("--size", "-s", type=int, required=False, default=1024, help="Input Image Size")
    parser.add_argument("--conf", "-c", type=float, required=False, default=0.15, help="Insert Confidence-Threshold Score")
    parser.add_argument("--iou", "-u", type=float, required=False, default=0.35, help="Insert IoU-Threshold Score")
    parser.add_argument("--agnostic_nms", "-a", action="store_true", required=False, help="class agnostic NMS")
    parser.add_argument("--line", "-l", action="store_true", required=False, help="Save as BBOX Line Image")
    parser.add_argument("--text", "-t", action="store_true", required=False, help="Save as Class Name Image")
    parser.add_argument("--inference_time", "-it", action="store_true", required=False, help="Show Inference Time per Image")

    args = parser.parse_args()

    main(args)