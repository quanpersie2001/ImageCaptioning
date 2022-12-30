
import os
import pickle
from model import inception_model, yolo4_model

from utils import add_end_start_tokens, clean_bad_text_data, combine_feature, create_vocab, extract_features, extract_yolo_features, get_max_length, get_train_image_captions_mapping, train_test_split, word_index_mapping


_inception_model = inception_model()
_yolo_model = yolo4_model()

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
    train_features = extract_features(train_data, _inception_model)
    pickle.dump(train_features, open('process_data/train_features.pkl', 'wb'))

if not os.path.exists('process_data/test_features.pkl'):
    test_features = extract_features(test_data, _inception_model)
    pickle.dump(test_features, open('process_data/test_features.pkl', 'wb'))

print('>>> Extracting and saving train yolo features...')
if not os.path.exists('process_data/train_yolo_features.pkl'):
    train_yolo_features = extract_yolo_features(train_data, _yolo_model)
    pickle.dump(train_yolo_features, open('process_data/train_yolo_features.pkl', 'wb'))

print('>>> Extracting and saving test yolo features...')
if not os.path.exists('process_data/test_yolo_features.pkl'):
    test_yolo_features = extract_yolo_features(test_data, _yolo_model)
    pickle.dump(test_yolo_features, open('process_data/test_yolo_features.pkl', 'wb'))

print('>>> Extracting and saving train combine features...')
if not os.path.exists('process_data/train_combine_features.pkl'):
    combine_features = {}
    with open('process_data/train_features.pkl', 'rb') as f:
        train_features = pickle.load(f)
    with open('process_data/train_yolo_features.pkl', 'rb') as f:
        train_yolo_features = pickle.load(f)
    
    for name in train_features:
        combine_features[name] = combine_feature(train_features[name], train_yolo_features[name])
    pickle.dump(combine_features, open('process_data/train_combine_features.pkl', 'wb'))

print('>>> Extracting and saving test combine features...')
if not os.path.exists('process_data/test_combine_features.pkl'):
    combine_features = {}
    with open('process_data/test_features.pkl', 'rb') as f:
        test_features = pickle.load(f)
    with open('process_data/test_yolo_features.pkl', 'rb') as f:
        test_yolo_features = pickle.load(f)
    
    for name in test_features:
        combine_features[name] = combine_feature(test_features[name], test_yolo_features[name])
    pickle.dump(combine_features, open('process_data/test_combine_features.pkl', 'wb'))

print('>>> Done!')