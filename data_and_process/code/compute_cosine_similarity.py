#!/usr/bin/env python
# encoding: utf-8
import pickle

import numpy as np
from scipy import spatial


def cos_similarity(a, b):
    with open('image_vectors.pkl', 'rb') as f:
        images = pickle.load(f)
        s1 = images[a]
        s2 = images[b]
        return 1 - spatial.distance.cosine(s1, s2)


def print_all_similariry():
    with open('image_vectors.pkl', 'rb') as f:
        images = pickle.load(f)
        keys = images.keys()
        print("length is {} .".format(len(keys)))
        for i in range(len(keys)):
            for j in range(i + 20000, len(keys)):
                sim = 1 - spatial.distance.cosine(images[keys[i]], images[keys[j]])
                print("{} and {} similarity is {} .".format(keys[i], keys[j], sim))
                # time.sleep(0.5)


def print_all_dis():
    with open('image_vectors.pkl', 'rb') as f:
        images = pickle.load(f)
        keys = images.keys()
        print("length is {} .".format(len(keys)))
        for i in range(len(keys)):
            for j in range(i + 20000, len(keys)):
                sim = np.linalg.norm(images[keys[i]] - images[keys[j]])
                print("{} and {} similarity is {} .".format(keys[i], keys[j], sim))
                # time.sleep(0.5)


def get_top_sim(image_number, top_num):
    print('execute get top sim')
    top_image_numbers = []
    top_images = []
    with open('image_vectors.pkl', 'rb') as f:
        images = pickle.load(f)
        print('load image_vectors.pkl')
        print('image vector is {}'.format(images[image_number]))
        for i in images.keys():
            #sim = np.linalg.norm(images[image_number] - images[i])
            sim = 1 - spatial.distance.cosine(images[image_number], images[i])
            print(sim)
            if len(top_images) > 0:
                print('{} vector is {}'.format(i, images[i]))
                m = max(top_image_numbers)
            if len(top_image_numbers) >= top_num:
                if sim > m:
                    index = top_image_numbers.index(m)
                    top_image_numbers.remove(m)
                    top_image_numbers.append(sim)
                    del top_images[index]
                    top_images.append(i)

            else:
                top_image_numbers.append(sim)
                top_images.append(i)

    return zip(top_images, top_image_numbers)


if __name__ == '__main__':
    # print_all_similariry()
    print_all_dis()
