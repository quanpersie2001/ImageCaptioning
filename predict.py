import argparse
import pickle
import keras
from pathlib import Path

import matplotlib.pyplot as plt
from model import inception_model
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.applications.inception_v3 import preprocess_input

from utils import k_beam_search


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

def predict(path, weight='weights/model_0.h5', k_beam=9, log=False):
    model = keras.models.load_model(weight, compile=False)

    pic = load_img(path, target_size=(299,299))
    image = img_to_array(pic)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)

    extrator = inception_model()
    feature = extrator.predict(image, verbose=0)

    try:
        with open('process_data/word_to_id.pkl','rb') as f:
            word_to_id = pickle.load(f)
        with open('process_data/id_to_word.pkl','rb') as f:
            id_to_word = pickle.load(f)
        with open('process_data/max_length.pkl','rb') as f:
            max_length = pickle.load(f)
    except FileNotFoundError:
        print('Please run preprocess.py first')
    
    fe = feature.reshape((1,2048))

    caption = k_beam_search(model, fe, word_to_id, id_to_word, max_length, k_beam, log)
    x = plt.imread(path)
    plt.imshow(x)
    plt.title(f'k_beam = {k_beam} - Caption: {caption}')
    plt.axis('off')
    plt.show()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='image path')
    parser.add_argument('--weight', nargs='+', type=str, default= ROOT / 'weights' / 'model_2.h5', help='weigths path(s)')
    parser.add_argument('--k-beam', type=int, default=9, help='beam size')
    parser.add_argument("--log", action=argparse.BooleanOptionalAction)
    opt = parser.parse_args()
    return opt


def main():
    opt = parse_opt()
    predict(opt.image, opt.weight, opt.k_beam, opt.log)


if __name__ == '__main__':
    main()
    