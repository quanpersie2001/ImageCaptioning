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

def predict(path, weight='weights/model.h5', k_beam=9, log=False):

    if not Path(weight).exists():
        print('Cannot find weight file, please run train.py or download pretrain model from https://drive.google.com/drive/u/2/folders/1q-COeg-nEMOnJIAfuJcKvcJl7sPQfLOn')
        return

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

    return caption


def run(path, weight='weights/model_28.h5', k_beam=5, log=False):
    caption = predict(path, weight, k_beam, log)
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
    opt = parser.parse_args()
    return opt


def main():
    opt = parse_opt()
    run(opt.image, opt.weight, opt.k_beam, opt.log)


if __name__ == '__main__':
    main()
