import argparse
import cv2
from src.neuralnetwork import ResnetUnetHybrid
import src.predictions as predictions

def get_arguments():
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video_path', required=False, type=str, help='Path to the input video.')
    return parser.parse_args()

def main(frame=None, model=None, device=None):
    if frame is None:
        args = get_arguments()
        if args.video_path:
            video_path = args.video_path
            if video_path.isdigit():
                video_path = int(video_path)
            predictions.ImageProcessor.predict_video(video_path)
        else:
            print("video path???")
    else:
        predictions.ImageProcessor.predict_img(frame, model=model, device=device)
    
if __name__ == '__main__':
    main()
