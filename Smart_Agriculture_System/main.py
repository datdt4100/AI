import argparse
import time
from pathlib import Path
from threading import Thread
from datetime import datetime, date
import sys

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, LoadWebcam
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from utils.door_control import *
import keyboard

import sys
from threading import Thread
from Adafruit_IO import MQTTClient
import time
import serial.tools.list_ports
import keyboard

import utils.read_sensors as read_sensors
import utils.relay_control as relay_control
import utils.port as port


PRED = None
MELTAL_PLANT_MODEL = './mentalplant.pt'
CONF = 0.1
IOU = 0.45
IMG_SIZE = 416
ACCESS_CAM = '0'
DETECT_CAM = '1'
SAVE_TXT = True #LOG file
PROJECT = 'Mental_Plant_Detection'
VIEW_IMG = True

#Constants
AIO_FEED_IDs = ["sensor1", "sensor2", "sensor3", "sensor4", "relay1", "relay2", "AI"]
AIO_USERNAME = "Fusioz"
AIO_KEY = "aio_HtYs87cVQ6F4VtA4wZkIlaHbl0lv"
PUBLISH_INTERVAL = 50
DETECTION_INTERVAL = 100

#Variables
count = 1
count_detect = 1
isRelay1Signal = False
isRelay1 = False
isRelay2Signal = False
isRelay2 = False
analysis_light = None
day_of_growth = 0

ser = None

#Connect to serial port
portName = port.getPort()
print("Port: " + portName)
if portName != "None":
    ser = serial.Serial(port=portName, baudrate=9600)

#MQTt callbacks
def connected(client):
    print("Ket noi thanh cong ...")
    for topic in AIO_FEED_IDs:
        client.subscribe(topic)

def subscribe(client , userdata , mid , granted_qos):
    print("Subscribe thanh cong...")

def disconnected(client):
    print("Ngat ket noi ...")
    sys.exit (1)

def message(client , feed_id , payload):
    #Update relay states based on MQTT messages
    
    if feed_id == "relay1":
        global isRealay1Signal, isRelay1
        isRealay1Signal = True
        if payload == "ON":
            isRelay1 = True
            print("Relay 1: ON")
        else:
            isRelay1 = False
            print("Relay 1: OFF")
    elif feed_id == "relay2":
        global isRelay2Signal, isRelay2
        isRelay2Signal = True
        if payload == "ON":
            isRelay2 = True
            print("Relay 2: ON")
        else:
            isRelay2 = False
            print("Relay 2: OFF")

#Connect to Adafruit IO
client = MQTTClient(AIO_USERNAME , AIO_KEY)
client.on_connect = connected
client.on_disconnect = disconnected
client.on_message = message
client.on_subscribe = subscribe
client.connect()
client.loop_background()


def show_result(names, colors, img, im0s, save_dir, view_img):
    global PRED
    if PRED == None:
        return False
    
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    
    detected = False
    # Process detections
    for i, det in enumerate(PRED):  # detections per image
        s, im0 = '%s: ' % current_time, im0s[i].copy()
        
        today = date.today().strftime("%d_%m_%Y")
        # p = Path(p)  # to Path
        txt_path = str(save_dir / 'log' / str(today) )  # img.txt
        #gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                #xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                #line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                with open(txt_path + '.txt', 'a') as f:
                    f.write(s + '\n')
                    detected = True

                if view_img:  # Add bbox to image
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

        # Print time (inference + NMS)
        # print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

        # Stream results
        if view_img:
            cv2.imshow('DISEASE DETECTION', im0)
            cv2.waitKey(1)  # 1 millisecond
            
    return detected

def run_model(img, model):
    global PRED

    # Inference
    #t1 = time_synchronized()
    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        PRED = model(img, augment=False)[0]
    #t2 = time_synchronized()

    # Apply NMS
    PRED = non_max_suppression(PRED, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    #t3 = time_synchronized()

    # Apply Classifier
    #pred = apply_classifier(pred, classifier, img, im0s)
    


def detect(save_img=False):
    weights = MELTAL_PLANT_MODEL
    source, view_img, save_txt, imgsz, trace = DETECT_CAM, VIEW_IMG, SAVE_TXT, IMG_SIZE, False
    webcam = True

    # Directories
    save_dir = Path(increment_path(Path(PROJECT), exist_ok=True))  # increment run
    (save_dir / 'log' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device()
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, IMG_SIZE)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    # modelc = load_classifier(name='resnet101', n=2)  # initialize
    # modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader   
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    
    t0 = time.time()
    t1 = 0

    door_access = None
    plant = None
    # dd/mm/YY
    
    
    global count
    global isRelay1Signal
    global isRelay1
    global isRelay2Signal
    global isRelay2
    global analysis_light
    global day_of_growth
    update_light_counter = 60
    
    
    for path, img, im0s, vid_cap in dataset:
        if count == PUBLISH_INTERVAL:
            air_temp_value = read_sensors.readTemperature(ser, port.serial_read_data)/10
            print(f"Air Temperature: {air_temp_value}°C")
            client.publish("sensor2", air_temp_value)
            air_humi_value = read_sensors.readMoisture(ser, port.serial_read_data)/10
            print(f"Air Humidity: {air_humi_value}%")
            client.publish("sensor1", air_humi_value)

            soil_temp_value = read_sensors.readSoilTemp(ser, port.serial_read_data)/100
            print(f"Soil Temperature: {soil_temp_value}°C")
            client.publish("sensor3", soil_temp_value)
            soil_humi_value = read_sensors.readSoilMoisture(ser, port.serial_read_data)/100
            print(f"Soil Humidity: {soil_humi_value}%")
            client.publish("sensor4", soil_humi_value)
            count = 0
        if isRelay1Signal:
            if isRelay1:
                relay_control.setDevice1(True, ser)
            else:
                relay_control.setDevice1(False, ser)
            isRelay1Signal = False
        if isRelay2Signal:
            if isRelay2:
                relay_control.setDevice2(True, ser)
            else:
                relay_control.setDevice2(False, ser)
            isRelay2Signal = False
        if keyboard.is_pressed('u'):
            try:
                from data_analysis import light
                analysis_light = light
            except ImportError:
                print("There is no function !")
        
        if analysis_light != None :
            update_light_counter -= 1
            if update_light_counter == 0:
                day = analysis_light(1)
                if day != None:
                    day_of_growth += day
                print(day_of_growth)
                update_light_counter = 60
                
        count += 1
        time.sleep(0.5)
        
        if keyboard.is_pressed('d') and (door_access == None or not door_access.is_alive()):
            door_access = Thread(target= recognition, args=(ACCESS_CAM,), daemon= True)
            door_access.start()
        
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        if plant == None or not plant.is_alive():
            plant = Thread(target=run_model, args=(img, model, ))
            plant.start()
        
        if not show_result(names, colors, img, im0s, save_dir, view_img):
            t1 += 1
            if t1 == 60:
                today = date.today().strftime("%d_%m_%Y")
                with open(str(save_dir / 'log' / str(today)) + '.txt' , 'a') as f:
                    now = datetime.now()
                    current_time = now.strftime("%H:%M:%S")
                    s = '%s: No detection!\n' % current_time
                    f.write(s)

    
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('log/*.txt')))}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    #physical_device = Thread(target=physical.physical_device_loop, args=(), daemon=True)
    #physical_device.start()

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--view-img', action='store_true', help='display results')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()
    sys.exit()
