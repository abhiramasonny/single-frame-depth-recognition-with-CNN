import cv2
import torch
from src.neuralnetwork import ResnetUnetHybrid
from typing import Optional, Tuple
import numpy as np
import math
from torchvision import transforms
import tkinter as tk
from tkinter import filedialog, ttk
import tkinter.simpledialog as simpledialog

HEIGHT: int = 256
WIDTH: int = 320

class ImageProcessor:
    @staticmethod
    def scale_image(image: np.ndarray, scale: Optional[float] = None) -> np.ndarray:
        if scale is None:
            scale = max(WIDTH / image.shape[1], HEIGHT / image.shape[0])

        new_size: Tuple[int, int] = (math.ceil(image.shape[1] * scale), math.ceil(image.shape[0] * scale))
        scaled_image: np.ndarray = cv2.resize(image, new_size, interpolation=cv2.INTER_NEAREST)
        return scaled_image

    @staticmethod
    def center_crop(image: np.ndarray) -> np.ndarray:
        corner: Tuple[int, int] = ((image.shape[0] - HEIGHT) // 2, (image.shape[1] - WIDTH) // 2)
        cropped_image: np.ndarray = image[corner[0]:corner[0] + HEIGHT, corner[1]:corner[1] + WIDTH]
        return cropped_image

    @staticmethod
    def img_transform(img):
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        normalized_image: np.ndarray = data_transform(img)
        return normalized_image

    @staticmethod
    def depth_to_color(depth_map: np.ndarray, max_distance: float = 10.0) -> np.ndarray:
        depth_map = np.transpose(depth_map, (1, 2, 0))
        depth_map[depth_map > max_distance] = max_distance
        depth_map = depth_map / max_distance

        depth_map = depth_map  # Invert depth map
        depth_map_hsv = np.zeros((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.uint8)

        # Color mapping: Closer objects (green/blue), further objects (yellow)
        depth_map_hsv[:, :, 0] = ((1.0 - depth_map[:, :, 0]) * 120).astype(np.uint8)  # Hue
        depth_map_hsv[:, :, 1] = 255  # Saturation
        depth_map_hsv[:, :, 2] = 255  # Value

        depth_map_rgb = cv2.cvtColor(depth_map_hsv, cv2.COLOR_HSV2BGR)
        return depth_map_rgb

    @staticmethod
    def predict_img(img: np.ndarray ,model: ResnetUnetHybrid = None,device = None):
        if model is None:
            try:
                device = torch.device('mps')
            except:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            print('Use GPU: {}'.format(str(device) != 'cpu'))
            print('Loading model')
            model = ResnetUnetHybrid.load_pretrained(device=device)
        
        img = ImageProcessor.scale_image(img)
        img = ImageProcessor.center_crop(img)
        inp = ImageProcessor.img_transform(img)
        inp = inp[None, :, :, :].to(device)

        print('Running the image through the network...')
        output = model(inp)

        output = output.cpu()[0].data.numpy()

        return output

    @staticmethod
    def predict_video(video_path):
        try:
            device = torch.device('mps')
        except:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print('Use GPU: {}'.format(str(device) != 'cpu'))
        print('Loading model')
        model = ResnetUnetHybrid.load_pretrained(device=device)
        model.eval()

        cap = cv2.VideoCapture(video_path)
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                frame = ImageProcessor.scale_image(frame)
                frame = ImageProcessor.center_crop(frame)
                inp = ImageProcessor.img_transform(frame)
                inp = inp[None, :, :, :].to(device)

                output = model(inp)

                output = output.cpu()[0].data.numpy()
                img_display = np.copy(frame)

                img_display = cv2.resize(img_display, (1920, 1080))
                cv2.imshow('org vid', img_display)

                depth_map = ImageProcessor.depth_to_color(output)
                depth_map = cv2.resize(depth_map, (1920, 1080))
                cv2.imshow('depth', depth_map)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def predict_webcam(camera_id):
        try:
            device = torch.device('mps')
        except:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print('Use GPU: {}'.format(str(device) != 'cpu'))
        print('Loading model')
        model = ResnetUnetHybrid.load_pretrained(device=device)
        model.eval()

        cap = cv2.VideoCapture(camera_id)
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                frame = ImageProcessor.scale_image(frame)
                frame = ImageProcessor.center_crop(frame)
                inp = ImageProcessor.img_transform(frame)
                inp = inp[None, :, :, :].to(device)

                output = model(inp)

                output = output.cpu()[0].data.numpy()
                img_display = np.copy(frame)

                img_display = cv2.resize(img_display, (1920, 1080))
                cv2.imshow('org vid', img_display)

                depth_map = ImageProcessor.depth_to_color(output)
                depth_map = cv2.resize(depth_map, (1920, 1080))
                cv2.imshow('depth', depth_map)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

def select_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        ImageProcessor.predict_video(file_path)