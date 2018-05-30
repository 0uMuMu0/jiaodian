import image_word2vec_process
import sys
import os

def main(argv):
    model = image_word2vec_process.load_pre_train_word2vec()
    desc_data = image_word2vec_process.load_desc_files(argv[2])

    image_word2vec_process.get_files(argv[1], model, desc_data)

if __name__ == '__main__':
    main(sys.argv)
