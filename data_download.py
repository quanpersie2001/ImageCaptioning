import os
import wget
import gdown
import zipfile
import tensorflow as tf
from pathlib import Path

from constants import GLOVE_URL, SSD300_WEIGHTS_URL_DOWNLOAD


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

if not os.path.exists(ROOT / 'data'):
    os.mkdir(ROOT / 'data')

print('>>> Downloading COCO dataset...')

# Download caption annotation files
print('>>> Downloading annotation files...')
if not (os.path.exists(ROOT / 'data' / 'annotations') and os.listdir(ROOT / 'data' / 'annotations')):
    annotation_zip = tf.keras.utils.get_file('captions.zip',
                                             cache_subdir=ROOT / 'data',
                                             origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                             extract=True)
    annotation_file = os.path.dirname(annotation_zip) + '/annotations/captions_train2014.json'
    os.remove(annotation_zip)
else:
    print('Annotation files already downloaded')

# Download image files
print('>>> Downloading annotation files...')
if not (os.path.exists(ROOT / 'data' / 'train2014') and os.listdir(ROOT / 'data' / 'train2014')):
    image_zip = tf.keras.utils.get_file('train2014.zip',
                                        cache_subdir=ROOT / 'data',
                                        origin='http://images.cocodataset.org/zips/train2014.zip',
                                        extract=True)
    os.remove(image_zip)
else:
    print('Image files already downloaded')


# Download glove embeddings
print('>>> Downloading glove embeddings <Wikipedia 2014 + Gigaword 5>...')
if not (os.path.exists(ROOT / 'data' / 'glove') and os.listdir(ROOT / 'data' / 'glove')):
    if not os.path.exists(ROOT / 'data' / 'glove'):
        os.mkdir(ROOT / 'data' / 'glove')
    wget.download(GLOVE_URL, out = 'data/glove')

    with zipfile.ZipFile(ROOT / 'data' / 'glove' / 'glove.6B.zip', 'r') as zip_ref:
        zip_ref.extractall(ROOT / 'data' / 'glove')

    os.remove(ROOT / 'data' / 'glove' / 'glove.6B.zip')
else:
    print('Glove embeddings already downloaded')


# Download cfg and weights of ssd300
print('>>> Downloading cfg and weights of ssd300...')
if not (os.path.exists(ROOT / 'ssd300' / 'weights') and os.listdir(ROOT / 'ssd300' / 'weights')):
    if not os.path.exists(ROOT / 'ssd300' / 'weights'):
        os.mkdir(ROOT / 'ssd300' / 'weights')
    gdown.download(SSD300_WEIGHTS_URL_DOWNLOAD, output='ssd300/weights/weights.h5', quiet=False)
