import re
import cv2
import json
import keras
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import keras.backend as K

from keras.preprocessing import image
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import pad_sequences
from keras.applications.inception_v3 import preprocess_input

from constants import START_TOKEN, END_TOKEN


def get_train_image_captions_mapping():
    """
    Returns a dictionary mapping image path to the list of captions
    :return: dict {image_path: [caption1, caption2, ...]}
    """
    try:
        result = {}
        with open('data/annotations/captions_train2014.json', 'r') as f:
            annotations = json.load(f)
            for val in annotations['annotations']:
                caption = f"{val['caption']}"
                image_path = 'data/train2014/' + 'COCO_train2014_' + '%012d.jpg' % (val['image_id'])
                result.setdefault(image_path, []).append(caption)
        return result
    except FileNotFoundError:
        print("Don't have the annotations file. Please run data_download.py to download the data.")


def clean_bad_text_data(image_captions_mapping):
    """
    Removes punctuation and numbers, conver to lower from the captions
    """
    for path, captions in image_captions_mapping.items():
        for i in range(len(captions)):
            cap = captions[i]
            cap = cap.split()

            # convert to lower case
            cap = [w.lower() for w in cap if w]

            # remove punctuation from each token
            cap = [re.sub(r'[^\w\s]','',w) for w in cap]

            # remove tokens with numbers in them
            cap = [w for w in cap if w.isalpha()]

            captions[i] =  ' '.join(cap)
            
    return image_captions_mapping


def add_end_start_tokens(image_captions_mapping):
    """
    Adds start and end tokens to the captions
    <start>sentence<end>
    """
    result = {}
    for key in image_captions_mapping:
        for i in range(len(image_captions_mapping[key])):
            result.setdefault(key, []).append(f'{START_TOKEN} {image_captions_mapping[key][i]} {END_TOKEN}')
    return result


def create_vocab(mapping, word_count_threshold = 5):
    """
    Creates a vocabulary of words from the captions
    :param mapping: dict {image_path: [caption1, caption2, ...]}
    :param word_count_threshold: int
    :return: list of words
    """
    # Create list of captions
    all_captions = []
    for captions in mapping.values():
        all_captions.extend(captions)

    # Allow only words which appear at least special times
    word_counts = {}
    nsents = 0
    for sent in all_captions:
        nsents += 1
        for w in sent.split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1

    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]

    return vocab


def word_index_mapping(vocab):
    """
    Creates a dictionary mapping word to index
    :param vocab: list of words
    :return: dict {word: index}, dict {index: word}
    """
    oov_token = '<UNK>'
    filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n'
    tokenizer = keras.preprocessing.text.Tokenizer(filters = filters, oov_token=oov_token)
    tokenizer.fit_on_texts(vocab)

    id_to_word = {}
    word_to_id = {}
    tokenizer.word_index['<PAD0>'] = 0 
    word_to_id = tokenizer.word_index

    for w in tokenizer.word_index:
        id_to_word[tokenizer.word_index[w]] = w

    return id_to_word, word_to_id, tokenizer


def train_test_split(data, train_size = 0.9, seed = 42):
    """
    Splits the data into train and test
    :param data: dict {image_path: [caption1, caption2, ...]}
    :param train_size: float
    :return: train, test
    """
    random.seed(seed)
    
    dict(data)
    img_keys = list(data.keys())
    random.shuffle(img_keys)
    sl_idx = int(len(img_keys) * train_size)

    train = dict(list(data.items())[:sl_idx])
    test = dict(list(data.items())[sl_idx:])

    return train, test


def masked_loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = K.sparse_categorical_crossentropy(real, pred, from_logits= False)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


def extract_inception_features_images(data, model, input_size = (299,299)):
    features = {}
    for path in tqdm(data):
        _feature = _extract_inception_feature_one_image(path, model, input_size)
        features[path] = _feature
    return features


def _extract_inception_feature_one_image(path, model, input_size = (299,299)):
    image = load_img(path, target_size=input_size)
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature.reshape(2048)


def _extract_ssd_feature_one_image(path, ssd_model, confidence_threshold=0.5):
    img = load_img(path, target_size=(300, 300))
    img = img_to_array(img)
    input = [img]
    input = np.array(input)
    bboxes = ssd_model.predict(input, verbose=0)

    ssd_feature = [bboxes[k][bboxes[k,:,1] > confidence_threshold] for k in range(bboxes.shape[0])]

    ssd_feature = [list(y) for y in ssd_feature[0]]
    n = len(ssd_feature)
    
    # for each bounding box, append (area * confidence)
    for i in range(n):
        ssd_feature[i].append((ssd_feature[i][4] - ssd_feature[i][2]) * (ssd_feature[i][5] - ssd_feature[i][3]) * ssd_feature[i][1])
    ssd_feature = sorted(ssd_feature, key=lambda x: x[-1], reverse=True)
    ssd_feature = np.array(ssd_feature)

    # Convert to relative attributes
    frame = cv2.imread(path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_width, image_height = frame.shape[1], frame.shape[0]

    result = []
    for ft in ssd_feature:
        _cls, _conf, _xmin, _ymin, _xmax, _ymax, _ = ft

        x_coord = (_xmin + _xmax) / 2 / image_width
        y_coord = (_ymin + _ymax) / 2 / image_height
        shape_width = (_xmax - _xmin) / image_width
        shape_height = (_ymax - _ymin) / image_height
        importance_factor = shape_width * shape_height * _conf

        result.append([x_coord, y_coord, shape_width, shape_height, _cls, _conf, importance_factor])
    result = np.array(result)

    return result.flatten()


def extract_ssd_features_images(data, ssd_model, confidence_threshold=0.5):
    features = {}
    for path in tqdm(data):
        ssd_feature = _extract_ssd_feature_one_image(path, ssd_model, confidence_threshold)
        features[path] = ssd_feature
    return features


def combine_feature(feature, ssd_feature, features_shape = 2048):
    ssd_feature = np.pad(ssd_feature, (0, features_shape - ssd_feature.shape[0]), 'constant', constant_values=(0, 0)).astype(np.float32)
    combined_features = np.vstack((feature, ssd_feature)).astype(np.float32)
    return combined_features.flatten()


def get_max_length(captions, percentile):
    all_caps = []
    for i in captions:
        for j in captions[i]:
            all_caps.append(j)

    length_all_desc = list(len(d.split()) for d in all_caps)

    return int(np.percentile(length_all_desc, percentile))


def data_generator(captions, pictures ,tokenizer, batch_size, max_length):
    X1, X2, y = [], [], []
    n = 0
    while 1:
        for key, caps in captions.items():
            n += 1
            photo = pictures[key]
            for cap in caps:
                seq = tokenizer.texts_to_sequences(cap.split())
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]

                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)

            # Yield the batch data
            if n == batch_size:
                yield [[np.array(X1), np.array(X2).squeeze(axis=-1)], np.array(y).squeeze(axis=-1)]
                X1, X2, y = list(), list(), list()
                n=0


def k_beam_search(model, pic_fe, word_to_id, id_to_word, max_length, k_beams = 3, log = False, mode='single'):
    shape = 2048 if mode == 'single' else 4096
    start = [word_to_id[START_TOKEN]]
    
    start_word = [[start, 0.0]]
    
    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            sequence  = pad_sequences([s[0]], maxlen=max_length).reshape((1,max_length))
            preds = model.predict([pic_fe.reshape(1, shape), sequence], verbose=0)
            word_preds = np.argsort(preds[0])[-k_beams:]

            for w in word_preds:
                
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                if log:
                    prob += np.log(preds[0][w]) # assign a probability to each K words4
                else:
                    prob += preds[0][w]
                temp.append([next_cap, prob])
        start_word = temp

        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])

        # Getting the top words
        start_word = start_word[-k_beams:]
    
    start_word = start_word[-1][0]
    captions_ = [id_to_word[i] for i in start_word]

    final_caption = []
    
    for i in captions_:
        if i != END_TOKEN:
            final_caption.append(i)
        else:
            break
    
    final_caption = ' '.join(final_caption[1:])
    return final_caption