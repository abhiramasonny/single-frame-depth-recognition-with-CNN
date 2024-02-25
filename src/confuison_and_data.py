import os
import sys
import cv2
import tarfile
import numpy as np
import torch
import torch.nn.functional as F
from src.neuralnetwork import ResnetUnetHybrid
from typing import Optional, Tuple
import math
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

HEIGHT: int = 256
WIDTH: int = 320

def scale_image(image: np.ndarray, scale: Optional[float] = None) -> np.ndarray:
    if scale is None:
        scale = max(WIDTH / image.shape[1], HEIGHT / image.shape[0])

    new_size: Tuple[int, int] = (math.ceil(image.shape[1] * scale), math.ceil(image.shape[0] * scale))
    scaled_image: np.ndarray = cv2.resize(image, new_size, interpolation=cv2.INTER_NEAREST)
    return scaled_image

def center_crop(image: np.ndarray) -> np.ndarray:
    corner: Tuple[int, int] = ((image.shape[0] - HEIGHT) // 2, (image.shape[1] - WIDTH) // 2)
    cropped_image: np.ndarray = image[corner[0]:corner[0] + HEIGHT, corner[1]:corner[1] + WIDTH]
    return cropped_image

def img_transform(img):
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    normalized_image: np.ndarray = data_transform(img)
    return normalized_image

def depth_to_grayscale(depth_map: np.ndarray, max_distance: float = 10.0) -> np.ndarray:
    depth_map: np.ndarray = np.transpose(depth_map, (1, 2, 0))
    depth_map[depth_map > max_distance] = max_distance
    depth_map: np.ndarray = depth_map / max_distance

    depth_map: np.ndarray = np.array(depth_map * 255.0, dtype=np.uint8)
    depth_map: np.ndarray = cv2.resize(depth_map, (WIDTH, HEIGHT))

    grayscale_depth_image: np.ndarray = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR)
    grayscale_depth_image: np.ndarray = np.clip(grayscale_depth_image, 0, 255)
    return grayscale_depth_image

def collect_test_files(download_path='./NYU_depth_v2_test_set.tar.gz'):
    if not os.path.exists(download_path):
        print('Downloading test set...')
        os.system('wget https://www.dropbox.com/s/zq0kf40bs3gl50t/NYU_depth_v2_test_set.tar.gz')

    test_dir = './NYU_depth_v2_test_set'

    if not os.path.exists(test_dir):
        print('Extracting test set...')
        tar = tarfile.open(download_path)
        tar.extractall(path='.')
        tar.close()

    test_img_paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('_rgb.png')]
    test_label_paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('_dpth.npy')]
    test_img_paths.sort()
    test_label_paths.sort()

    return test_img_paths, test_label_paths

def compute_errors():
    try:
        device = torch.device('mps')
    except:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Use GPU: {}'.format(str(device) != 'cpu'))

    print('Loading model...')
    model = ResnetUnetHybrid.load_pretrained(device=device)
    model.eval()

    preds = np.zeros((466, 582, 654), dtype=np.float32)
    labels = np.zeros((466, 582, 654), dtype=np.float32)

    test_img_paths, test_label_paths = collect_test_files()

    print('\nRunning evaluation:')
    for idx, (img_path, label_path) in enumerate(zip(test_img_paths, test_label_paths)):
        sys.stdout.write('\r{} / {}'.format(idx+1, len(test_img_paths)))
        sys.stdout.flush()

        img = cv2.imread(img_path)[..., ::-1]

        img = scale_image(img, 0.55)
        img = center_crop(img)
        img = img_transform(img)
        img = img[None, :, :, :].to(device)

        pred = model(img)

        pred = F.interpolate(pred, size=(466, 582), mode='bilinear', align_corners=False)
        pred = pred.cpu().data.numpy()

        label = np.load(label_path)
        label = label[7:label.shape[0]-7, 29:label.shape[1]-29]

        labels[:, :, idx] = label
        preds[:, :, idx] = pred[0, 0, :, :]

    rel_error = np.mean(np.abs(preds - labels)/labels)
    print('\nMean Absolute Relative Error: {:.6f}'.format(rel_error))

    rmse = np.sqrt(np.mean((preds - labels)**2))
    print('Root Mean Squared Error: {:.6f}'.format(rmse))

    log10 = np.mean(np.abs(np.log10(preds) - np.log10(labels)))
    print('Mean Log10 Error: {:.6f}'.format(log10))

    acc = np.maximum(preds/labels, labels/preds)
    delta1 = np.mean(acc < 1.25)
    print('Delta1: {:.6f}'.format(delta1))

    delta2 = np.mean(acc < 1.25**2)
    print('Delta2: {:.6f}'.format(delta2))

    delta3 = np.mean(acc < 1.25**3)
    print('Delta3: {:.6f}'.format(delta3))

    print("flattining")
    preds_flat = preds.flatten()
    labels_flat = labels.flatten()
    # Discretize predictions and labels
    preds_discrete = np.round(preds_flat).astype(int)
    labels_discrete = np.round(labels_flat).astype(int)

    # Compute confusion matrix
    # Compute confusion matrix
    conf_matrix = confusion_matrix(labels_discrete, preds_discrete)

    # Calculate percentages
    conf_matrix_percent = conf_matrix / np.sum(conf_matrix, axis=1)[:, np.newaxis] * 100

    # Plot confusion matrix with percentages
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix_percent, annot=True, fmt='.2f', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix (%)')
    plt.show()


def main():
    compute_errors()

if __name__ == '__main__':
    main()
