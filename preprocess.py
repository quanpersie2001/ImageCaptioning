
import os
import pickle
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3

from utils import add_end_start_tokens, clean_bad_text_data, create_vocab, extract_features, get_max_length, get_train_image_captions_mapping, train_test_split, word_index_mapping

inception = InceptionV3()
inception_model = Model(inputs=inception.inputs, outputs=inception.layers[-2].output)

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
    train_features = extract_features(train_data, inception_model)
    pickle.dump(train_features, open('process_data/train_features.pkl', 'wb'))

if not os.path.exists('process_data/test_features.pkl'):
    test_features = extract_features(test_data, inception_model)
    pickle.dump(test_features, open('process_data/test_features.pkl', 'wb'))

print('>>> Done!')