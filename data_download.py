import os
import sys
import tensorflow as tf
from pathlib import Path


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

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