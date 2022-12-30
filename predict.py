import cv2
import keras
import pickle
import argparse
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img
from model import inception_model, yolo4_model
from tensorflow.keras.utils import img_to_array
from keras.applications.inception_v3 import preprocess_input

from utils import combine_feature, k_beam_search


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

def predict(path, weight='weights/model.h5', k_beam=9, log=False, mode='single'):
    """
    path: path to image
    weight: path to weight file
    k_beam: k for k-beam search
    log: print log or not
    mode: single (just use inceptionNet) or dual (use inceptionNet and YOLO) to extract feature
    """

    if not Path(weight).exists():
        print('Cannot find weight file, please run train.py or download pretrain model from https://drive.google.com/drive/u/2/folders/1q-COeg-nEMOnJIAfuJcKvcJl7sPQfLOn')
        return

    if mode not in ['single', 'dual']:
        raise ValueError('mode must be `single` or `dual`')

    model = keras.models.load_model(weight, compile=False)

    pic = load_img(path, target_size=(299,299))
    image = img_to_array(pic)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)

    extrator = inception_model()
    feature = extrator.predict(image, verbose=0)

    yolo_feature = None
    if mode == 'dual':
        yolo_model = yolo4_model()
        # frame = cv2.imread(path)
        frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bboxes = yolo_model.predict(frame, prob_thresh=0.8)
        bboxes = bboxes.tolist()
        n = len(bboxes)
        # for each bounding box, append (area * confidence)
        for i in range(n):
            bboxes[i].append(bboxes[i][2] * bboxes[i][3] * bboxes[i][5])
        bboxes = np.array(bboxes)

        yolo_feature = np.array(bboxes.flatten())
        
    try:
        with open('process_data/word_to_id.pkl','rb') as f:
            word_to_id = pickle.load(f)
        with open('process_data/id_to_word.pkl','rb') as f:
            id_to_word = pickle.load(f)
        with open('process_data/max_length.pkl','rb') as f:
            max_length = pickle.load(f)
    except FileNotFoundError:
        print('Please run preprocess.py first')
    if yolo_feature is not None:
        fe = combine_feature(feature, yolo_feature)
    else:
        fe = feature.reshape((1,2048))
    
    caption = k_beam_search(model, fe, word_to_id, id_to_word, max_length, k_beam, log, mode)

    return caption


def run(path, weight='weights/model_28.h5', k_beam=5, log=False, mode='single'):
    caption = predict(path, weight, k_beam, log, mode)
    if caption:
        img = plt.imread(path)
        plt.imshow(img)

        plt.suptitle(f'K-beam:{k_beam}', fontsize=14, fontweight=1, y=0.95, color='blue')
        plt.title(f'{caption}', fontsize=12, fontweight=0, y=-0.1 )

        plt.axis('off')
        plt.show()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='image path')
    parser.add_argument('--weight', type=str, default= ROOT / 'weights' / 'model.h5', help='weigths path(s)')
    parser.add_argument('--k-beam', type=int, default=5, help='beam size')
    parser.add_argument("--log", action=argparse.BooleanOptionalAction)
    parser.add_argument("--mode", type=str, default='single', help='single or dual')
    opt = parser.parse_args()
    return opt


def main():
    opt = parse_opt()
    run(opt.image, opt.weight, opt.k_beam, opt.log, opt.mode)


if __name__ == '__main__':
    main()
