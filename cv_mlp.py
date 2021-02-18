import os
import random
from concurrent.futures.thread import ThreadPoolExecutor
from glob import glob

import numpy as np
from cv2 import cv2


class DynamicDataLoader:
    def __init__(self, image_paths, class_names, input_size, channels, num_train_samples_per_epoch):
        self.image_paths = image_paths
        self.class_names = class_names
        self.input_size = input_size
        self.channels = channels
        self.num_train_samples_per_epoch = num_train_samples_per_epoch
        self.pool = ThreadPoolExecutor(8)

    def next(self):
        random.shuffle(self.image_paths)
        train_x = []
        train_y = []
        fs = []
        for i in range(self.num_train_samples_per_epoch):
            fs.append(self.pool.submit(self.__load_img, self.image_paths[i]))
        for f in fs:
            cur_img_path, x = f.result()
            x = cv2.resize(x, self.input_size)
            x = np.asarray(x).reshape(-1).astype('float32') / 255.0
            train_x.append(x)

            dir_name = cur_img_path.replace('\\', '/').split('/')[-2]
            y = [-1.0 for _ in range(len(self.class_names))]
            if dir_name != 'unknown':
                y[self.class_names.index(dir_name)] = 1.0
            train_y.append(y)
        train_x = np.asarray(train_x).astype('float32')
        train_y = np.asarray(train_y).astype('float32')
        return train_x, train_y

    def __load_img(self, path):
        return path, cv2.imread(path, cv2.IMREAD_GRAYSCALE if self.channels == 1 else cv2.IMREAD_COLOR)


def evaluate_using_static_data(model, train_data):
    train_x = train_data.getSamples()
    y_true = train_data.getResponses()
    y_pred = []
    for i in range(len(train_x)):
        y = model.precict(train_x[i])
        print(y)
    loss = 0
    recall = 0
    return loss, recall


def evaluate_using_image_paths(model, image_paths):
    loss = 0
    recall = 0
    return loss, recall


def evaluate(model, evaluate_data):
    if type(evaluate_data) is list:
        return evaluate_using_image_paths(model, evaluate_data)
    return evaluate_using_static_data(model, evaluate_data)


def main():
    train_image_path = r'C:\inz\train_data\test_ocr_b1'
    validation_split = 0.2
    input_size = (16, 32)
    channels = 1
    hidden_layer_units = [256]
    lr = 1e-4
    momentum = 0.1
    epochs = 300
    num_train_samples_per_epoch = 10000
    load_type = 'static'  # dynamic, static
    pretrained_model_path = ''

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

    class_names = sorted(list(class_name_set))
    data_loader = DynamicDataLoader(
        image_paths=train_image_paths,
        class_names=class_names,
        input_size=input_size,
        channels=channels,
        num_train_samples_per_epoch=num_train_samples_per_epoch)

    train_data = object()
    validation_data = object()
    if load_type == 'static':
        all_data_loader = DynamicDataLoader(
            image_paths=train_image_paths,
            class_names=class_names,
            input_size=input_size,
            channels=channels,
            num_train_samples_per_epoch=len(train_image_paths))
        train_x, train_y = all_data_loader.next()
        train_data = cv2.ml.TrainData_create(train_x, cv2.ml.ROW_SAMPLE, train_y)

        if len(validation_image_paths) > 0:
            all_data_loader = DynamicDataLoader(
                image_paths=validation_image_paths,
                class_names=class_names,
                input_size=input_size,
                channels=channels,
                num_train_samples_per_epoch=len(validation_image_paths))
            train_x, train_y = all_data_loader.next()
            validation_data = cv2.ml.TrainData_create(train_x, cv2.ml.ROW_SAMPLE, train_y)

    num_features = input_size[0] * input_size[1] * channels
    num_classes = len(class_names)
    layer_sizes = np.asarray([num_features] + hidden_layer_units + [num_classes])
    print(layer_sizes)

    model = cv2.ml.ANN_MLP_create()
    model.setLayerSizes(layer_sizes)
    model.setTermCriteria((cv2.TERM_CRITERIA_EPS, -1, 0.1))
    model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
    model.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 1.0, 1.0)
    model.setBackpropWeightScale(lr)
    model.setBackpropMomentumScale(momentum)

    max_recall = 0.0
    max_val_recall = 0.0
    for epoch in range(epochs):
        print(f'\nepoch {epoch + 1} / {epochs}')
        if load_type != 'static':
            train_x, train_y = data_loader.next()
            train_data = cv2.ml.TrainData_create(train_x, cv2.ml.ROW_SAMPLE, train_y)
        if epoch == 0:
            model.train(train_data, cv2.ml.ANN_MLP_NO_INPUT_SCALE + cv2.ml.ANN_MLP_NO_OUTPUT_SCALE)
        else:
            model.train(train_data, cv2.ml.ANN_MLP_NO_INPUT_SCALE + cv2.ml.ANN_MLP_NO_OUTPUT_SCALE + cv2.ml.ANN_MLP_UPDATE_WEIGHTS)
        loss, recall = evaluate(model, train_data)
        if len(validation_image_paths) > 0:
            if load_type == 'static':
                val_loss, val_recall = evaluate(model, validation_data)
            else:
                val_loss, val_recall = evaluate(model, validation_image_paths)
            if val_recall > max_val_recall:
                max_val_recall = val_recall
                model.save(f'checkpoints/epoch_{epoch + 1}_loss_{loss:.4f}_val_loss_{val_loss:.4f}_recall_{recall:.4f}_val_recall_{val_recall:.4f}.xml')
            print(f'loss : {loss:.4f}, val_loss : {val_loss:.4f}, recall : {recall:.4f}, val_recall : {val_recall:.4f}')
        else:
            if recall > max_recall:
                max_recall = recall
                model.save(f'checkpoints/epoch_{epoch + 1}_loss_{loss:.4f}_recall_{recall:.4f}.xml')
            print(f'loss : {loss:.4f}, recall : {recall:.4f}')


if __name__ == '__main__':
    main()
