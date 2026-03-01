import numpy as np
import cv2
import os

def split_horizontal(image, n_strips):
    h, w, c = image.shape
    sh = h // n_strips
    strips = []

    for i in range(n_strips):
        strips.append(image[i*sh:(i+1)*sh, :])

    return strips

def split_blocks(strip):
    h, w, c = strip.shape
    blocks = []

    bh = h // 3
    bw = w // 5

    for r in range(3):
        for col in range(5):
            block = strip[r*bh:(r+1)*bh, col*bw:(col+1)*bw]
            blocks.append(block)

    return blocks

def rgb_stats(block):
    m = np.mean(block, axis=(0,1))
    v = np.var(block, axis=(0,1))
    return np.concatenate((m, v))

def extract_image_features(path):
    img = cv2.imread(path)
    if img is None:
        return np.array([])

    strips = split_horizontal(img, 5)
    features = []

    for s in strips:
        blocks = split_blocks(s)
        vec = []
        for b in blocks:
            vec.extend(rgb_stats(b))
        features.append(vec)

    return np.array(features)

def build_dataset(folder):
    X = []
    y = []
    label = 0

    for person in os.listdir(folder):
        ppath = os.path.join(folder, person)
        if not os.path.isdir(ppath):
            continue

        for imgname in os.listdir(ppath):
            ipath = os.path.join(ppath, imgname)
            feats = extract_image_features(ipath)

            if feats.size == 0:
                continue

            for f in feats:
                X.append(f)
                y.append(label)

        label += 1

    return np.array(X), np.array(y)