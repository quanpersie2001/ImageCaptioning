import os
import numpy as np

from keras.layers import RepeatVector
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM ,GRU
from keras.layers import Embedding
from keras.layers import Dropout, Reshape, Lambda, Concatenate
from keras.layers import Embedding

from utils import masked_loss_function, word_index_mapping


class InceptionNet_LSTM():
    def __init__(self, vocab, embed_dim=50):
        self.vocab = vocab
        self.vocab_size = len(vocab) + 2
        self.embed_dim = embed_dim
        self.embed_layer = self.work_embedding()

    def work_embedding(self):
        glove_dir = 'data/glove/'
        _, word_to_id, _ = word_index_mapping(self.vocab)
        embeddings_index = {}

        with open(os.path.join(glove_dir, f'glove.6B.{self.embed_dim}d.txt'), encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        # Get x-dim dense vector for each of the vocab_rocc
        embedding_matrix = np.zeros((self.vocab_size, self.embed_dim))
        for word, i in word_to_id.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:

                # Words not found in the embedding index will be all zeros
                embedding_matrix[i] = embedding_vector
            
        embed_layer = Embedding(self.vocab_size, self.embed_dim, mask_zero=True, trainable=False)
        embed_layer.build((None,))
        embed_layer.set_weights([embedding_matrix])
        return embed_layer

    def build_model(self, max_length, feature_size = 2048, units= 512):
        
        features = Input(shape=(feature_size,))
        X_fe_one_dim = Dense(units, activation='relu')(features) 
        X_fe = RepeatVector(max_length)(X_fe_one_dim)
        X_fe = Dropout(0.2)(X_fe)
        
        seq = Input(shape=(max_length,))
        X_seq = self.embed_layer(seq)
        X_seq = Lambda(lambda x: x, output_shape=lambda s:s)(X_seq) # remove mask from the embedding cause concat doesn't support it
        X_seq = Dropout(0.2)(X_seq)
        X_seq = Concatenate(axis=-1)([X_fe,X_seq])
        X_seq = LSTM(units, return_sequences=True)(X_seq,initial_state=[X_fe_one_dim,X_fe_one_dim])
        X_seq = Dropout(0.5)(X_seq)
        X_seq = LSTM(units, return_sequences=False)(X_seq)

        outputs = Dense(self.vocab_size, activation='softmax')(X_seq)

        # merge the two input models
        model = Model(inputs=[features, seq], outputs = outputs, name='model_with_features_each_step')
        model.compile(loss=masked_loss_function, optimizer= 'RMSProp')
        return model

def inception_model():
    from keras.applications.inception_v3 import InceptionV3
    from keras.models import Model
    inception = InceptionV3()
    model = Model(inputs=inception.inputs, outputs=inception.layers[-2].output)
    return model