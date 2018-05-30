import os
import sys
import pickle
import time
from PIL import Image
from random import shuffle

import gensim
import numpy as np
import tensorflow as tf


class Image_without_desc:
    pass


def load_pre_train_word2vec(model_location='/home/zyt/proj/data_and_process/GoogleNews-vectors-negative300.bin'):
    print('loading the pretrained model. It will takes a few minutes')
    model = gensim.models.Word2Vec.load_word2vec_format(model_location, binary=True)
    print('successfully load the model! \n start to generate training set.')

    return model


def get_files(dir, model):
    """ get all data from dir which consists all kind of images that will be used. The description of image
       wiil be read from the json file for the kind of image.


       Args:
           dir: directory that all kind of images places.

     Returns:
         all data of images in the directory.

    """

    dirs1 = os.listdir(dir)
    image_data = []
    # gen_desc_vectors(model, desc_data)
    count = 0
    for d1 in dirs1:
        now_dir = os.path.join(dir, d1)
        print('processing images for {} .....'.format(now_dir))
        dirs2 = os.listdir(now_dir)
        for d2 in dirs2:
            files = os.listdir(os.path.join(now_dir, d2))
            for f in files:
                image = Image_without_desc()
                image.data = Image.open(os.path.join(now_dir, d2 + '/' + f)).resize((64, 64))
                image.number = f[:-4]
                if np.array(image.data).shape == (64, 64, 3):
                    image_data.append(image)
                    count += 1
                else:
                    print("image number is {}. And shape is {}.".format(image.number, np.array(image.data).shape))

    shuffle(image_data)
    conver_to(image_data)
    print('total number of images is %d .' % count)


def conver_to(image_data, save_file_name='image_mix2_with_number'):

    if not os.path.isdir('train_set'):
        os.mkdir('train_set', 0755)
    file_name = os.path.join('train_set', save_file_name + '.tfrecords')
    writer = tf.python_io.TFRecordWriter(file_name)

    for image in image_data:
        # print("label shape is {}. image shape is {}.".format(np.array((image.desc_vector)).shape,
        #                                                      np.array((image.data)).shape))
        image_raw = (image.data).tobytes()
        image_number = image.number
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(image_raw),
            'image_number': _bytes_feature(image_number)}))

        writer.write(example.SerializeToString())

    writer.close()


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def main(argv):
    model = load_pre_train_word2vec()

    get_files(argv[1], model)


if __name__ == '__main__':
    main(sys.argv)
