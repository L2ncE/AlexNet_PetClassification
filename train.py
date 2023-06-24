import mindspore.dataset as ds
import numpy as np
from mindspore import nn, Model
from mindspore.train.callback import LossMonitor
from mindspore.dataset import vision


class PetImageClassifier:
    def __init__(self, image_size=227, batch_size=8, learning_rate=0.001, momentum=0.9, epochs=10):
        self.image_size = image_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epochs = epochs

        # Define dataset transformations
        self.mean = [0.5 * 255] * 3
        self.std = [0.5 * 255] * 3
        self.transforms = [
            vision.Resize((self.image_size, self.image_size)),
            vision.Normalize(mean=self.mean, std=self.std),
            vision.HWC2CHW()
        ]

        # Load train dataset
        self.train_dataset = ds.ImageFolderDataset(
            dataset_dir='./dataset/PetImages/train',
            decode=True).map(
            operations=self.transforms, num_parallel_workers=1).batch(self.batch_size)
        self.train_dataset, self.val_dataset = self.train_dataset.split([0.9, 0.1])

        # Load test dataset
        self.test_dataset = ds.ImageFolderDataset(
            dataset_dir='./dataset/PetImages/eval',
            decode=True).map(
            operations=self.transforms, num_parallel_workers=1).batch(self.batch_size)

        self.net = AlexNet()
        self.loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        self.opt = nn.Momentum(self.net.trainable_params(), learning_rate=self.learning_rate, momentum=self.momentum)
        self.model = Model(self.net, loss_fn=self.loss, optimizer=self.opt, metrics={'accuracy'})

    def train(self):
        self.model.train(
            epoch=self.epochs,
            train_dataset=self.train_dataset,
            callbacks=[LossMonitor()],
            dataset_sink_mode=True)

    def test(self):
        accuracy = self.model.eval(self.val_dataset, dataset_sink_mode=False)
        print(f'Test accuracy: {accuracy}')

    def predict(self):
        predictions = []
        for data in self.test_dataset.create_dict_iterator():
            inputs = data['image']
            output = self.model.predict(inputs)
            pred = np.argmax(output.asnumpy(), axis=1)
            predictions.append(pred)
        print(predictions)


class AlexNet(nn.Cell):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.SequentialCell(
            nn.Conv2d(3, 96, 11, stride=4, pad_mode="valid"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, 5, pad_mode="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, 3, pad_mode="same"),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, pad_mode="same"),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, pad_mode="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten()
        )
        self.classifier = nn.SequentialCell(
            nn.Dense(6 * 6 * 256, 4096),
            nn.ReLU(),
            nn.Dense(4096, 4096),
            nn.ReLU(),
            nn.Dense(4096, 100)
        )

    def construct(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    # Create an instance of PetImageClassifier and train the model
    pet_image_classifier = PetImageClassifier()
    pet_image_classifier.train()

    # Test the trained model
    pet_image_classifier.test()

    # Make predictions on test dataset
    pet_image_classifier.predict()
