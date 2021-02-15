import os
import random
from glob import glob

import numpy as np
from cv2 import cv2


def main():
    train_image_path = r'C:\inz\train_data\ocr_b1'
    validation_split = 0.2
    input_size = (16, 32)
    lr = 1e-4

    train_image_path = train_image_path.replace('\\', '/')
    dir_paths = glob(f'{train_image_path}/*')
    class_name_set = set()
    train_image_paths = []
    validation_image_paths = []
    for dir_path in dir_paths:
        if os.path.isdir(dir_path):
            class_name_set.add(dir_path.replace('\\', '/').split('/')[-1])
            class_image_paths = glob(rf'{dir_path}/*.jpg')
            if validation_split == 0.0:
                train_image_paths += class_image_paths
                continue
            random.shuffle(class_image_paths)
            num_train_images = int(len(class_image_paths) * (1.0 - validation_split))
            train_image_paths += class_image_paths[:num_train_images]
            validation_image_paths += class_image_paths[num_train_images:]

    class_names = list(class_name_set)
    train_image_paths = train_image_paths[:500]
    train_x = []
    train_y = []
    for path in train_image_paths:
        x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        x = cv2.resize(x, input_size)
        x = np.asarray(x).reshape((input_size[0] * input_size[1])).astype('float32') / 255.0
        train_x.append(x)

        dir_name = path.replace('\\', '/').split('/')[-2]
        y = [0.0 for _ in range(len(class_names))]
        if dir_name != 'unknown':
            y[class_names.index(dir_name)] = 1.0
        train_y.append(y)

    train_x = np.asarray(train_x).astype('float32')
    train_y = np.asarray(train_y).astype('float32')
    train_data = cv2.ml.TrainData_create(train_x, cv2.ml.ROW_SAMPLE, train_y)

    model = cv2.ml.ANN_MLP_create()
    layer_sizes = np.asarray([512, 256, 46])
    model.setLayerSizes(layer_sizes)
    model.setTermCriteria((cv2.TERM_CRITERIA_EPS, -1, 0.1))
    model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
    model.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
    model.setBackpropWeightScale(lr)

    model.train(train_data, cv2.ml.ANN_MLP_NO_INPUT_SCALE + cv2.ml.ANN_MLP_NO_OUTPUT_SCALE)


if __name__ == '__main__':
    main()
