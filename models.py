import os
import numpy as np

from keras import backend as K
from keras.optimizers import Adam

from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import Dropout, Lambda, Concatenate

from ssd300.model import ssd_300
from ssd300.keras_loss_function.keras_ssd_loss import SSDLoss
from utils import masked_loss_function, word_index_mapping


class ImageCaptionModel():
    """
    Image Caption Model
    """
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


    def build_model(self, max_length, feature_size = 2048, units= 512, mode='single'):
        """
        Build the Image Captioning Model
        Args:
            - mode (str):
                    - single: use CNN backbone (InceptionNetV3)
                    - dual: use 2 model are CNN (InceptionNetV3) and Object Detection (SSD300)
        """
        feature_size = 2048 if mode == 'single' else 2048*2
        features = Input(shape=(feature_size,))
        X_fe_one_dim = Dense(units, activation='relu')(features) 
        X_fe = RepeatVector(max_length)(X_fe_one_dim)
        X_fe = Dropout(0.2)(X_fe)
        
        seq = Input(shape=(max_length,))
        X_seq = self.embed_layer(seq)
        X_seq = Lambda(lambda x: x, output_shape=lambda s:s)(X_seq) # Remove mask from the embedding cause concat doesn't support it
        X_seq = Dropout(0.2)(X_seq)
        X_seq = Concatenate(axis=-1)([X_fe,X_seq])
        X_seq = LSTM(units, return_sequences=True)(X_seq,initial_state=[X_fe_one_dim,X_fe_one_dim])
        X_seq = Dropout(0.5)(X_seq)
        X_seq = LSTM(units, return_sequences=False)(X_seq)

        outputs = Dense(self.vocab_size, activation='softmax')(X_seq)

        # Merge the two input models
        model = Model(inputs=[features, seq], outputs = outputs, name='model_with_features_each_step')
        model.compile(loss=masked_loss_function, optimizer= 'RMSProp')
        return model


def inception_model():
    from keras.applications.inception_v3 import InceptionV3
    from keras.models import Model
    inception = InceptionV3()
    model = Model(inputs=inception.inputs, outputs=inception.layers[-2].output)
    return model


def ssd_300_model():
    K.clear_session() # Clear previous models from memory.
    model = ssd_300(image_size=(300, 300, 3),
                    n_classes=80,
                    mode='inference',
                    l2_regularization=0.0005,
                    scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
                    aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                            [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                            [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                            [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                            [1.0, 2.0, 0.5],
                                            [1.0, 2.0, 0.5]],
                    two_boxes_for_ar1=True,
                    steps=[8, 16, 32, 64, 100, 300],
                    offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    clip_boxes=False,
                    variances=[0.1, 0.1, 0.2, 0.2],
                    normalize_coords=True,
                    subtract_mean=[123, 117, 104],
                    swap_channels=[2, 1, 0],
                    confidence_thresh=0.01,
                    iou_threshold=0.45,
                    top_k=200,
                    nms_max_output_size=400)

    # 2: Load the trained weights into the model.
    weights_path = 'ssd300/weights/weight.h5'
    if not os.path.exists(weights_path):
        print('Can not find the weights file')

    model.load_weights(weights_path, by_name=True)

    # 3: Compile the model so that Keras won't complain the next time you load it.
    adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

    return model