#Import standard module
#### System module ###
import argparse
import time
from pathlib import Path

#### Functional module ####
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import seaborn as sns
from pylab import rcParams
import copy
from efficientnet_pytorch import EfficientNet

#Import custom module
from models.experimental import*
from utils.general import *
from utils.torch_utils import *
from utils.dataloader import *
from utils.plots import *
from raft import *

# set defualts
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
rcParams['figure.figsize'] = 16, 10
np.random.seed(42)

TASK = ['DetectTrafficSign', 'DetectVehicle']

def inference_detection_in_files(dataset, save_dir, image_size, opt, device, model, task):
    #Set dataloader
    vid_path, vid_writer = None, None
    
    old_img_w = old_img_h = image_size
    old_img_b = 1
    
    # Get names and colors for traffic sign
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if device != 'cpu' else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        # if classify:
        #    pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name) + task  # img.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if not opt.nosave:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            '''# Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond'''

            # Save results (image with detections)
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
                print(f" The image with the result is saved in: {save_path}")
            else:  # 'video' or 'stream'
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)
    
        #s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    print(f"Results saved to {save_dir}{s}")
    del s
    print(f'Done. ({time.time() - t0:.3f}s)')
    return save_dir, vid_path


def velocity_estimation_in_files(dataset, device, model_optical_flow, model_velocity_estimation):
    _ , prev_data, _, _ = next(dataset)
    prev_data = torch.from_numpy(prev_data).permute(2, 0, 1)
    prev_data = prev_data.half() if device != 'cpu' else prev_data.float()
    prev_data = prev_data[None].to(device)
    
    padder = InputPadder(prev_data.shape)
    
    prev_data = padder.pad(prev_data)
    avg_v = 0
    nframes = 0
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        t1 = time_synchronized()
        im0s = torch.from_numpy(im0s).permute(2, 0, 1)
        im0s = im0s.half() if device != 'cpu' else im0s.float()
        im0s = im0s[None].to(device)
        
        im0s = padder.pad(im0s)
        
        flow_low, flow_up = model_optical_flow(prev_data, im0s, iters=20, test_mode=True)
        velocity = model_velocity_estimation(flow_up).item()
        velocity = round(velocity,2)
        
        avg_v += velocity
        prev_data = im0s
        nframes += 1
        t2 = time_synchronized()
        print(f'Speed is {velocity}. Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference')
    print(f'Done. ({time.time() - t0:.3f}s)')
    return avg_v/nframes
    
        
        

def main(opt):
    ####### INITIALIZATION #######
    
    source, image_size, weights_speed, weights_distance, weights_optical_flow, weights_velocity_estimation = \
        opt.source, opt.image_size, opt.weights_speed, opt.weights_distance, opt.weights_optical_flow, opt.weights_velocity_estimation
    camera = source.isnumeric()
    save_img = not opt.nosave #save inference image option
    
    #Check saved directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir
    
    #Check device
    device = select_device(opt.device)
    half = device != 'cpu' # half precision only supported on CUDA
    
    #Load model of speed limitation
    model_speed = attempt_load(weights_speed, map_location=device)  # load FP32 model
    model_distance = attempt_load(weights_distance, map_location=device)  # load FP32 model
    
    stride = int(model_speed.stride.max())  # model stride
    image_size_spd = check_img_size(image_size, s=stride)  # check img_size
    image_size_dst = check_img_size(640, s=int(model_distance.stride.max()))  # check img_size
    
    model_speed = TracedModel(model_speed, device, image_size_spd, "speed_traced.pt")
    model_distance = TracedModel(model_distance, device, image_size_dst, "distance_traced.pt")
    #model_distance = DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=half)
    model_optical_flow = torch.nn.DataParallel(RAFT())
    model_optical_flow.load_state_dict(torch.load(weights_optical_flow))
    model_optical_flow = model_optical_flow.module
    model_optical_flow.to(device)
    model_optical_flow.eval()
    
    model_velocity_estimation = EfficientNet.from_pretrained('efficientnet-b0', in_channels=2, num_classes=1)
    model_velocity_estimation.load_state_dict(torch.load(weights_velocity_estimation))
    model_velocity_estimation.to(device)
    model_velocity_estimation.eval()

    
    if half:
        model_speed.half()  # to FP16
        model_distance.half()
        model_optical_flow.half()
        model_velocity_estimation.half()
        
    #Check show requiments
    view_image = check_imshow() & opt.view_image
    cudnn.benchmark = True # set True to speed up constant image size inference
    
    ######## RUN INFERENCE ##########
    
    if device.type != 'cpu':
        model_speed(torch.zeros(1, 3, image_size_spd, image_size_spd).to(device).type_as(next(model_speed.parameters())))  # run once
        model_distance(torch.zeros(1, 3, image_size_dst, image_size_dst).to(device).type_as(next(model_distance.parameters())))  # run once
    
    dataset0 = LoadImages(source, img_size=image_size_spd, stride=stride)
    dataset1 = LoadImages(source, img_size=image_size_dst, stride=stride)
    dataset2 = LoadImages(source, img_size=image_size_spd, stride=stride)
    
    if not camera:
        dir_path, vid_path = inference_detection_in_files(dataset0, save_dir, image_size_spd, opt, device, model_speed, TASK[0])
        dir_path, vid_path = inference_detection_in_files(dataset1, save_dir, image_size_dst, opt, device, model_distance, TASK[1])
        avg_spd = velocity_estimation_in_files(dataset, device, model_optical_flow, model_velocity_estimation)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-speed', nargs='+', type=str, default='weights/yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--weights-distance', nargs='+', type=str, default='weights/yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--weights-optical-flow', nargs='+', type=str, default='weights/yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--weights-velocity-estimation', nargs='+', type=str, default='weights/yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--image-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-image', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    
    with torch.no_grad():
        main(opt)