import os
import keras
import pickle
import argparse
import tensorflow as tf
from pathlib import Path
from keras.models import load_model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from models import ImageCaptionModel
from utils import add_end_start_tokens, clean_bad_text_data, create_vocab, data_generator, get_max_length, get_train_image_captions_mapping, train_test_split, word_index_mapping

import warnings
warnings.filterwarnings("ignore")

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]


def run(batch_size = 64, output = ROOT / 'weights', epochs = 100, save_history = True, mode='single', resume=False):
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

    inception_lstm = ImageCaptionModel(vocab)
    model = inception_lstm.build_model(max_length=mlength, mode=mode)


    class CustomSaver(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if epoch % 10 == 0:  # or save after some epoch, each k-th epoch etc.
                self.model.save(output / f'model_{epoch}_{mode}.h5')

    saver = CustomSaver()
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    reduce_lr= ReduceLROnPlateau(monitor='loss', factor=0.9, patience=5, verbose=0, mode='auto', min_delta=0.0001, min_lr=0.000001)

    data_train = data_generator(train_data, train_features, tokenizer, batch_size, mlength)
    data_val = data_generator(test_data, test_features, tokenizer, batch_size, mlength)
    steps = len(train_data) // batch_size

    checkpoint_filepath = f'weights/checkpoint/weights_{mode}.best.hdf5'
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='loss',
        mode='auto',
        save_best_only=True)

    print('>>> Training model...')
    if resume:
        model.load_weights(checkpoint_filepath)
    h = model.fit(data_train, epochs=epochs, steps_per_epoch=steps, verbose=1, callbacks=[early_stopping, saver, checkpoint])

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
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction)
    opt = parser.parse_args()
    return opt


def main():
    opt = parse_opt()
    run(opt.batch_size, opt.output, opt.epochs, opt.save_history, opt.mode, opt.resume)


if __name__ == '__main__':
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        print("physical_devices-------------", len(physical_devices))
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    main()
    
    

