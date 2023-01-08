import keras
import pickle
import argparse
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
from constants import PRE_TRAINED_WEIGHTS_URL
from models import inception_model, ssd_300_model
from keras.applications.inception_v3 import preprocess_input

from utils import _extract_inception_feature_one_image, _extract_ssd_feature_one_image, combine_feature, k_beam_search


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

def predict(path, weight=None, k_beam=9, log=False, mode='single'):
    """
    path: path to image
    weight: path to weight file
    k_beam: k for k-beam search
    log: print log or not
    mode: single (just use inceptionNet) or dual (use inceptionNet and SSD 300) to extract feature
    """

    if not Path(weight).exists():
        print(f'Cannot find weight file, please run train.py or download pretrain model from {PRE_TRAINED_WEIGHTS_URL}')
        return

    if mode not in ['single', 'dual']:
        raise ValueError('mode must be `single` or `dual`')
    
    if not weight:
        raise ValueError('weight must be provided')

    ssd_feature = None
    if mode == 'dual':

        # Extract ssd feature
        ssd300 = ssd_300_model()
        ssd_feature = _extract_ssd_feature_one_image(path, ssd300)

    # Extract inception feature
    _inception_model = inception_model()
    inception_feature = _extract_inception_feature_one_image(path, _inception_model)

    model = keras.models.load_model(weight, compile=False)
        
    try:
        with open('process_data/word_to_id.pkl','rb') as f:
            word_to_id = pickle.load(f)
        with open('process_data/id_to_word.pkl','rb') as f:
            id_to_word = pickle.load(f)
        with open('process_data/max_length.pkl','rb') as f:
            max_length = pickle.load(f)
    except FileNotFoundError:
        print('Please run preprocess.py first')

    if ssd_feature is not None:
        # Combine inception feature and ssd feature
        fe = combine_feature(inception_feature, ssd_feature)
    else:
        fe = inception_feature.reshape((1,2048))
    
    caption = k_beam_search(model, fe, word_to_id, id_to_word, max_length, k_beam, log, mode)

    return caption


def run(path, weight=None, k_beam=5, log=False, mode='single'):
    caption = predict(path, weight, k_beam, log, mode)
    if caption:
        img = plt.imread(path)
        plt.imshow(img)

        plt.suptitle(f'K-beam:{k_beam}', fontsize=14, fontweight=1, y=0.95, color='blue')
        plt.title(f'{caption[0].upper()}{caption[1:]}', fontsize=12, fontweight=0, y=-0.1 )

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
