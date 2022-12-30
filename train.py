import os
import sys
import argparse
from pathlib import Path
import keras
import pickle
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau

from constants import WEIGHTS_FOLDER
from model import InceptionNet_LSTM
from utils import add_end_start_tokens, clean_bad_text_data, create_vocab, data_generator, extract_features, get_max_length, get_train_image_captions_mapping, train_test_split, word_index_mapping

import warnings
warnings.filterwarnings("ignore")

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]


def run(batch_size = 64, output = ROOT / 'weights', epochs = 100, save_history = True, mode='single'):
    if not os.path.exists(output):
        os.makedirs(output)

    if not isinstance(batch_size, int):
        raise ValueError('batch_size must be an integer')

    if not isinstance(epochs, int):
        raise ValueError('epochs must be an integer')
    
    if mode not in ['single', 'dual']:
        raise ValueError('mode must be either `single` or `dual`')

    history={'loss':[], 'BLEU_val':[]}
    try:
        with open('process_data/image_captoin_mapping_clean.pkl','rb') as f:
            image_captoin_mapping_clean= pickle.load(f)
    except Exception as err:
        print('>>> Cleaning and saving data...')
        image_captoin_mapping = get_train_image_captions_mapping()
        image_captoin_mapping_clean = clean_bad_text_data(image_captoin_mapping)
        pickle.dump(image_captoin_mapping_clean, open('process_data/image_captoin_mapping_clean.pkl', 'wb'))

    image_captoin_mapping_with_token = add_end_start_tokens(image_captoin_mapping_clean)

    print('>>> Creating vocab...')
    vocab = create_vocab(image_captoin_mapping_with_token)

    try:
        with open('process_data/word_tokenize.pkl','rb') as f:
            tokenizer = pickle.load(f)
    except Exception as err:
        id_to_word, word_to_id, tokenizer = word_index_mapping(vocab)

    train_data, test_data = train_test_split(image_captoin_mapping_with_token)

    print('>>> Get features...')
    if mode == 'single':
        print('>>> Single mode selected')
        ex = ''
    else:
        print('>>> Dual mode selected')
        ex = '_combine'
    try:
        with open(f'process_data/train{ex}_features.pkl','rb') as f:
            train_features= pickle.load(f)
    except FileNotFoundError as err:
        print('Please run `preprocess.py` first')

    try:
        with open(f'process_data/test{ex}_features.pkl','rb') as f:
            test_features= pickle.load(f)
    except FileNotFoundError as err:
        print('Please run `preprocess.py` first')
        

    mlength = get_max_length(train_data, 90)

    inception_lstm = InceptionNet_LSTM(vocab)
    model = inception_lstm.build_model(max_length=mlength, mode=mode)


    class CustomSaver(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if epoch % 5 == 0:  # or save after some epoch, each k-th epoch etc.
                self.model.save(output / f'model_{epoch}_{mode}.h5')

    saver = CustomSaver()
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    reduce_lr= ReduceLROnPlateau(monitor='loss', factor=0.9, patience=5, verbose=0, mode='auto', min_delta=0.0001, min_lr=0.000001)

    data_gen = data_generator(train_data, train_features, tokenizer, batch_size, mlength)
    steps = len(train_data) // batch_size

    print('>>> Training model...')
    h = model.fit(data_gen, epochs=epochs, steps_per_epoch=steps, verbose=1, callbacks=[early_stopping, reduce_lr, saver])

    if save_history:
        history['loss'].append(h.history['loss'])
        if not os.path.exists('weights/histories'):
            os.makedirs('weights/histories')
        pickle.dump(history, open(f'weights/histories/history_{mode}.pkl', 'wb'))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--output', type=str, default= ROOT / 'weights', help='weigths path(s)')
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument("--save-history", action=argparse.BooleanOptionalAction)
    parser.add_argument("--mode", type=str, default='single', help='single or dual')
    opt = parser.parse_args()
    return opt


def main():
    opt = parse_opt()
    run(opt.batch_size, opt.output, opt.epochs, opt.save_history, opt.mode)


if __name__ == '__main__':
    main()
    
    

