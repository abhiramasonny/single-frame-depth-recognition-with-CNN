import argparse
import cv2
import torch
from src.network import ResnetUnetHybrid
from typing import Optional, Tuple
import numpy as np
import cv2
import math
from torchvision import transforms

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
    def depth_to_grayscale(depth_map: np.ndarray, max_distance: float = 10.0) -> np.ndarray:
        depth_map: np.ndarray = np.transpose(depth_map, (1, 2, 0))
        depth_map[depth_map > max_distance] = max_distance
        depth_map: np.ndarray = depth_map / max_distance

        depth_map: np.ndarray = np.array(depth_map * 255.0, dtype=np.uint8)
        depth_map: np.ndarray = cv2.resize(depth_map, (WIDTH, HEIGHT))

        grayscale_depth_image: np.ndarray = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR)
        grayscale_depth_image: np.ndarray = np.clip(grayscale_depth_image, 0, 255)
        return grayscale_depth_image

    @staticmethod
    def predict_img(img_path):
        """Inference a single image and display."""
        try:
            device = torch.device('mps')
        except:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Use GPU: {}'.format(str(device) != 'cpu'))

        print('Loading model...')
        model = ResnetUnetHybrid.load_pretrained(device=device)
        model.eval()

        # load image
        img = cv2.imread(img_path)[..., ::-1]
        img = ImageProcessor.scale_image(img)
        img = ImageProcessor.center_crop(img)
        inp = ImageProcessor.img_transform(img)
        inp = inp[None, :, :, :].to(device)

        # inference
        print('Running the image through the network...')
        output = model(inp)

        # transform the results
        output = output.cpu()[0].data.numpy()
        img_display = np.copy(img)

        # Display original image
        cv2.imshow('Original Image', img_display)

        # Display predicted depth map
        depth_map = ImageProcessor.depth_to_grayscale(output)
        cv2.imshow('Predicted Depth Map', depth_map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def predict_video(video_path):
        """Inference a video and display."""
        try:
            device = torch.device('mps')
        except:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Use GPU: {}'.format(str(device) != 'cpu'))

        print('Loading model...')
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

                # inference
                output = model(inp)

                # transform the results
                output = output.cpu()[0].data.numpy()
                img_display = np.copy(frame)

                # Display original frame
                cv2.imshow('Original Video', img_display)

                # Display predicted depth map
                depth_map = ImageProcessor.depth_to_grayscale(output)
                cv2.imshow('Predicted Depth Map', depth_map)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

def get_arguments():
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img_path', required=False, type=str, help='Path to the input image.')
    parser.add_argument('-v', '--video_path', required=False, type=str, help='Path to the input video.')
    return parser.parse_args()

def main():
    args = get_arguments()
    if args.img_path:
        ImageProcessor.predict_img(args.img_path)
    elif args.video_path:
        video_path = args.video_path
        if video_path.isdigit():  # Check if video_path is an integer
            video_path = int(video_path)
        ImageProcessor.predict_video(video_path)
    else:
        print("Please provide either an image path or a video path.")


if __name__ == '__main__':
    main()
