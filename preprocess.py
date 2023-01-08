import os
import pickle
from tqdm import tqdm
from models import inception_model, ssd_300_model

from utils import add_end_start_tokens, clean_bad_text_data, combine_feature, create_vocab, extract_inception_features_images, extract_ssd_features_images, get_max_length, get_train_image_captions_mapping, train_test_split, word_index_mapping

import warnings
warnings.filterwarnings("ignore")

_ssd300_model = ssd_300_model()
_inception_model = inception_model()

if not os.path.exists('process_data'):
    os.makedirs('process_data')

print('>>> Cleaning and saving data...')
image_captoin_mapping = get_train_image_captions_mapping()
image_captoin_mapping_clean = clean_bad_text_data(image_captoin_mapping)
if not os.path.exists('process_data/image_captoin_mapping_clean.pkl'):
    pickle.dump(image_captoin_mapping_clean, open('process_data/image_captoin_mapping_clean.pkl', 'wb'))

image_captoin_mapping_with_token = add_end_start_tokens(image_captoin_mapping_clean)

print('>>> Creating vocab...')
vocab = create_vocab(image_captoin_mapping_with_token)
id_to_word, word_to_id, tokenizer = word_index_mapping(vocab)

print('>>> Save tokenized data...')
if not os.path.exists('process_data/word_tokenize.pkl'):
    pickle.dump(tokenizer, open('process_data/word_tokenize.pkl', 'wb'))

if not os.path.exists('process_data/id_to_word.pkl'):
    pickle.dump(id_to_word, open('process_data/id_to_word.pkl', 'wb'))

if not os.path.exists('process_data/word_to_id.pkl'):
    pickle.dump(word_to_id, open('process_data/word_to_id.pkl', 'wb'))

train_data, test_data = train_test_split(image_captoin_mapping_with_token)

print('>>> Getting max length...')
if not os.path.exists('process_data/max_length.pkl'):
    mlength = get_max_length(train_data, 90)
    pickle.dump(mlength, open('process_data/max_length.pkl', 'wb'))

print('>>> Extracting and saving features...')
if not os.path.exists('process_data/train_features.pkl'):
    train_features = extract_inception_features_images(train_data, _inception_model)
    pickle.dump(train_features, open('process_data/train_features.pkl', 'wb'))

if not os.path.exists('process_data/test_features.pkl'):
    test_features = extract_inception_features_images(test_data, _inception_model)
    pickle.dump(test_features, open('process_data/test_features.pkl', 'wb'))

print('>>> Extracting and saving train ssd 300 features...')
if not os.path.exists('process_data/train_ssd_features.pkl'):
    train_ssd_features = extract_ssd_features_images(train_data, _ssd300_model)
    pickle.dump(train_ssd_features, open('process_data/train_ssd_features.pkl', 'wb'))

print('>>> Extracting and saving test ssd features...')
if not os.path.exists('process_data/test_ssd_features.pkl'):
    test_ssd_features = extract_ssd_features_images(test_data, _ssd300_model)
    pickle.dump(test_ssd_features, open('process_data/test_ssd_features.pkl', 'wb'))

print('>>> Extracting and saving train combine features...')
if not os.path.exists('process_data/train_combine_features.pkl'):
    combine_features = {}
    with open('process_data/train_features.pkl', 'rb') as f:
        train_features = pickle.load(f)
    with open('process_data/train_ssd_features.pkl', 'rb') as f:
        train_ssd_features = pickle.load(f)
    
    for name in tqdm(train_features):
        combine_features[name] = combine_feature(train_features[name], train_ssd_features[name])
    pickle.dump(combine_features, open('process_data/train_combine_features.pkl', 'wb'))

print('>>> Extracting and saving test combine features...')
if not os.path.exists('process_data/test_combine_features.pkl'):
    combine_features = {}
    with open('process_data/test_features.pkl', 'rb') as f:
        test_features = pickle.load(f)
    with open('process_data/test_ssd_features.pkl', 'rb') as f:
        test_ssd_features = pickle.load(f)
    
    for name in tqdm(test_features):
        combine_features[name] = combine_feature(test_features[name], test_ssd_features[name])
    pickle.dump(combine_features, open('process_data/test_combine_features.pkl', 'wb'))

print('>>> Done!')