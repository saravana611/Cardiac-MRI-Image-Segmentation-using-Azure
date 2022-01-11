import os
import argparse
import datetime
import uuid
import tensorflow as tf
import matplotlib.pyplot as plt

from azureml.core.run import Run
from azureml.core import Datastore
from azureml.core.model import Model, Dataset
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Flatten, Dense, Reshape, Conv2D, MaxPool2D, Conv2DTranspose)


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print ('\nSample Prediction after epoch {}\n'.format(epoch+1))


class Train():

    def __init__(self):

        self._parser = argparse.ArgumentParser("train")
        self._parser.add_argument("--model_name", type=str, help="Name of the tf model")

        self._args = self._parser.parse_args()
        self._run = Run.get_context()
        self._exp = self._run.experiment
        self._ws = self._run.experiment.workspace
        self._image_feature_description = {
            'height':    tf.io.FixedLenFeature([], tf.int64),
            'width':     tf.io.FixedLenFeature([], tf.int64),
            'depth':     tf.io.FixedLenFeature([], tf.int64),
            'name' :     tf.io.FixedLenFeature([], tf.string),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'label_raw': tf.io.FixedLenFeature([], tf.string),
        }
        self._model = self.__get_model()
        self._parsed_training_dataset, self._parsed_val_dataset = self.__load_dataset()
        self.__steps_per_epoch = len(list(self._parsed_training_dataset))
        self._buffer_size = 10
        self._batch_size = 1
        self.__epochs = 30


    def main(self):
        plt.rcParams['image.cmap'] = 'Greys_r'

        tf_autotune = tf.data.experimental.AUTOTUNE
        train = self._parsed_training_dataset.map(
            self.__read_and_decode, num_parallel_calls=tf_autotune)
        val = self._parsed_val_dataset.map(self.__read_and_decode)

        train_dataset = train.cache().shuffle(self._buffer_size).batch(self._batch_size).repeat()
        train_dataset = train_dataset.prefetch(buffer_size=tf_autotune)
        test_dataset  = val.batch(self._batch_size)

        for image, label in train.take(2):
            sample_image, sample_label = image, label
            self.__display("Training Images", [sample_image, sample_label])

        for image, label in val.take(2):
            sample_image, sample_label = image, label
            self.__display("Eval Images", [sample_image, sample_label])

        logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

        tf.keras.backend.clear_session()

        self._model = self.__get_model()

        model_history = self._model.fit(train_dataset, epochs=self.__epochs,
                          steps_per_epoch=self.__steps_per_epoch,
                          validation_data=test_dataset,
                          callbacks=[DisplayCallback()])

        metrics_results = self._model.evaluate(test_dataset)
        self._run.log("DICE", "{:.2f}%".format(metrics_results[0]))
        self._run.log("Accuracy", "{:.2f}%".format(metrics_results[1]))

        self.__plot_training_logs(model_history)
        self.__show_predictions(test_dataset, 5)
        self.__register_model(metrics_results)

    
    def __parse_image_function(self, example_proto):
        return tf.io.parse_single_example(example_proto, self._image_feature_description)


    def __load_dataset(self):
        raw_training_dataset = tf.data.TFRecordDataset('data/train_images.tfrecords')
        raw_val_dataset      = tf.data.TFRecordDataset('data/val_images.tfrecords')

        parsed_training_dataset = raw_training_dataset.map(self.__parse_image_function)
        parsed_val_dataset = raw_val_dataset.map(self.__parse_image_function)

        return parsed_training_dataset, parsed_val_dataset


    @tf.function
    def __read_and_decode(self, example):
        image_raw = tf.io.decode_raw(example['image_raw'], tf.int64)
        image_raw.set_shape([65536])
        image = tf.reshape(image_raw, [256, 256, 1])

        image = tf.cast(image, tf.float32) * (1. / 1024)

        label_raw = tf.io.decode_raw(example['label_raw'], tf.uint8)
        label_raw.set_shape([65536])
        label = tf.reshape(label_raw, [256, 256, 1])

        return image, label


    def __display(self, image_title, display_list):
        plt.figure(figsize=(10, 10))
        title = ['Input Image', 'Label', 'Predicted Label']

        for i in range(len(display_list)):
            display_resized = tf.reshape(display_list[i], [256, 256])
            plt.subplot(1, len(display_list), i+1)
            plt.title(title[i])
            plt.imshow(display_resized)
            plt.axis('off')
        title = uuid.uuid4()
        self._run.log_image(f'{title}', plot=plt)


    def __create_mask(self, pred_mask):
        pred_mask = tf.argmax(pred_mask, axis=-1)
        pred_mask = pred_mask[..., tf.newaxis]
        return pred_mask[0]


    def __show_predictions(self, dataset=None, num=1):
        if dataset:
            for image, label in dataset.take(num):
                pred_mask = self._model.predict(image)
                self.__display("Show predictions", [image[0], label[0], self.__create_mask(pred_mask)])
        else:
            prediction = self.__create_mask(self._.predict(sample_image[tf.newaxis, ...]))
            self.__display("Show predictions sample image", [sample_image, sample_label, prediction])

    
    def __get_dice_coef(self, y_true, y_pred, smooth=1):
        indices = K.argmax(y_pred, 3)
        indices = K.reshape(indices, [-1, 256, 256, 1])

        true_cast = y_true
        indices_cast = K.cast(indices, dtype='float32')

        axis = [1, 2, 3]
        intersection = K.sum(true_cast * indices_cast, axis=axis)
        union = K.sum(true_cast, axis=axis) + K.sum(indices_cast, axis=axis)
        dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)

        return dice

    
    def __get_model(self):
        layers = [
            Conv2D(input_shape=[256, 256, 1],
                filters=100,
                kernel_size=5,
                strides=2,
                padding="same",
                activation=tf.nn.relu,
                name="Conv1"),
            MaxPool2D(pool_size=2, strides=2, padding="same"),
            Conv2D(filters=200,
                kernel_size=5,
                strides=2,
                padding="same",
                activation=tf.nn.relu),
            MaxPool2D(pool_size=2, strides=2, padding="same"),
            Conv2D(filters=300,
                kernel_size=3,
                strides=1,
                padding="same",
                activation=tf.nn.relu),
            Conv2D(filters=300,
                kernel_size=3,
                strides=1,
                padding="same",
                activation=tf.nn.relu),
            Conv2D(filters=2,
                kernel_size=1,
                strides=1,
                padding="same",
                activation=tf.nn.relu),
            Conv2DTranspose(filters=2, kernel_size=31, strides=16, padding="same")
        ]

        tf.keras.backend.clear_session()
        model = tf.keras.models.Sequential(layers)

        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[self.__get_dice_coef, 'accuracy', self.__f1_score,
                    self.__precision, self.__recall])
        
        return model


    def __plot_training_logs(self, model_history):
        loss = model_history.history['loss']
        val_loss = model_history.history['val_loss']
        accuracy = model_history.history['accuracy']
        val_accuracy = model_history.history['val_accuracy']
        dice = model_history.history['__get_dice_coef']

        epochs = range(self.__epochs)

        plt.figure()
        plt.plot(epochs, loss, 'r', label='Training loss')
        plt.plot(epochs, val_loss, 'bo', label='Validation loss')
        plt.plot(epochs, dice, 'go', label='Dice Coefficient')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.ylim([0, 1])
        plt.legend()
        self._run.log_image("Training and Validation Loss", plot=plt)


    def __recall(self, y_true, y_pred):
        y_true = K.ones_like(y_true) 
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        
        recall = true_positives / (all_positives + K.epsilon())
        return recall


    def __precision(self, y_true, y_pred):
        y_true = K.ones_like(y_true) 
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision


    def __f1_score(self, y_true, y_pred):
        precision = self.__precision(y_true, y_pred)
        recall = self.__recall(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))


    def __register_model(self, metrics_results):
        tf.keras.models.save_model(
            self._model, "./model", overwrite=True, include_optimizer=True, save_format=tf,
            signatures=None, options=None)
        Model.register(workspace=self._ws,
                    model_path="./model",
                    model_name=self._args.model_name,
                    properties = {"run_id": self._run.id,
                                "experiment": self._run.experiment.name},
                    tags={
                        "DICE": float(metrics_results[0]),
                        "Accuracy": float(metrics_results[1])
                    })


if __name__ == '__main__':
    tr = Train()
    tr.main()