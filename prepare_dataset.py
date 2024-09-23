import numpy as np
import os
from PIL import Image
import imageio
import random
from matplotlib import pyplot as plt


def generate_asl_data():
    dataset = []
    data_root = "./data/asl_dataset/"
    alphabets = os.listdir(data_root)
    i = 0
    for alphabet in alphabets:
        img_name_list = os.listdir(os.path.join(data_root, alphabet))
        batch = []
        img_name_list = random.choices(img_name_list, k=100)
        for img_name in img_name_list:
            # img_dir = data_root + alphabet + '/' + img_name
            img_dir = os.path.join(data_root, alphabet, img_name)
            img = imageio.imread(img_dir)
            # resized_img =  np.array(imageio.imread(img_dir))
            resized_img = np.array(Image.fromarray(img).resize(size=(80, 80)))
            if 'B' in img_name:
                plt.imshow(resized_img, interpolation='nearest')
                plt.show()
                plt.imshow(img, interpolation='nearest')
                plt.show()

            if (img.shape != (200, 200, 3)):
                print(i, img_name, img.shape)
                i += 1
            batch.append(resized_img)
        dataset.append(batch)

    print(len(dataset[0]))
    np.save('./data/' + "asl.npy", np.asarray(dataset, dtype='uint8'))


def generate_omniglot_data():
    dataset = []
    # images_background
    data_root = "./data/omniglot_dataset/"
    alphabets = os.listdir(data_root + "images_background")
    for alphabet in alphabets:
        characters = os.listdir(os.path.join(data_root, "images_background", alphabet))
        for character in characters:
            files = os.listdir(os.path.join(data_root, "images_background", alphabet, character))
            examples = []
            for img_file in files:
                img_name = os.path.join(data_root, "images_background", alphabet, character, img_file)
                img = imageio.imread(img_name)
                resized_img = np.array(Image.fromarray(img).resize(size=(28, 28)))
                # img = (np.float32(img) / 255.).flatten()
                examples.append(resized_img)
            dataset.append(examples)

    # images_evaluation
    alphabets = os.listdir(data_root + "images_evaluation")
    for alphabet in alphabets:
        characters = os.listdir(os.path.join(data_root, "images_evaluation", alphabet))
        for character in characters:
            files = os.listdir(os.path.join(data_root, "images_evaluation", alphabet, character))
            examples = []
            for img_file in files:
                img_name = os.path.join(data_root, "images_evaluation", alphabet, character, img_file)
                img = imageio.imread(img_name)
                resized_img = np.array(Image.fromarray(img).resize(size=(28, 28)))
                # img = (np.float32(img) / 255.).flatten()
                examples.append(resized_img)
            dataset.append(examples)

    # np.save(data_root + "omniglot.npy", np.asarray(dataset))

generate_asl_data()