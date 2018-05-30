import os
import pickle
import sys

import numpy as np

import image_word2vec_process


def compute_similarity(v1, v2):
    pass


class Image:
    pass


def gen_desc_vectors():
    model = image_word2vec_process.load_pre_train_word2vec()
    descs = image_word2vec_process.load_desc_files('/home/ttf/Desktop/product_info.txt')

    desc_list = {}
    for image_num, desc in descs.items():
        image = Image()
        desc_ls = (desc[:-1]).strip().split(' ')
        desc_vector = np.zeros((300,), dtype=np.float32)
        for s in desc_ls:
            if model.__contains__(s):
                desc_vector += model[s]
        image.desc = desc
        image.desc_vector = desc_vector
        print("{} desc is {}".format(desc, desc_vector))
        desc_list[image_num] = image
    print('begin to write vectors to disk...')
    with open("desc_vectors.pkl", "wb") as f:
        pickle.dump(desc_list, f, pickle.HIGHEST_PROTOCOL)
    print('successfully')


def compute_desc_similarity_for_one(dir):
    class Desc_simi:
        pass

    print('executing .......')
    files = os.listdir(dir)
    simi_list = []
    with open("desc_vectors.pkl", "rb") as f:
        descs = pickle.load(f)
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                image1_num = (files[i])[:-4]
                image2_num = (files[j])[:-4]
                if descs.has_key(image1_num) and descs.has_key(image2_num):
                    # compute
                    desc_simi = Desc_simi()
                    image_1 = descs[image1_num]
                    image_1.image_num = image1_num
                    desc_simi.image_1 = image_1
                    image_2 = descs[image2_num]
                    image_2.image_num = image2_num
                    desc_simi.image_2 = image_2
                    desc_simi.simi = np.linalg.norm(image_1.desc_vector - image_2.desc_vector)
                    simi_list.append(desc_simi)

                    # print(desc_simi.image_1.desc)
    #sorted(simi_list, key=lambda x: x.simi)
    simi_list.sort(key=lambda x: x.simi)
    return simi_list


if __name__ == '__main__':
    # gen_desc_vectors()
    result = compute_desc_similarity_for_one(sys.argv[1])
    # print(result)
