import os

import matplotlib.pyplot as plt
import numpy as np
import ximu3csv
from aeon.utils.discovery import all_estimators

all_estimators("classifier", tag_filter={"algorithm_type": "convolution"})

from aeon.classification.convolution_based import (
    Arsenal,
    HydraClassifier,
    MiniRocketClassifier,
    MultiRocketClassifier,
    MultiRocketHydraClassifier,
    RocketClassifier,
)
from sklearn.metrics import accuracy_score

labels = [
    "wave01",
    "wave02",
    "wave03",
    "chess01",
    "chess02",
    "chess03",
]


def load_data(directory):
    motions = np.empty([6, 6, 100])
    motions_labels = []

    for index, label in enumerate(labels):
        motions_labels.append(label[:-2])

        devices = ximu3csv.read(os.path.join(directory, label), ximu3csv.DataMessageType.INERTIAL)

        total_time = (devices[0].inertial.timestamp[-1] - devices[0].inertial.timestamp[0]) / 1e6

        devices = ximu3csv.resample(devices, 101 / total_time)

        imu = np.hstack([devices[0].inertial.gyroscope.xyz, devices[0].inertial.accelerometer.xyz]).T
        imu = imu[:, :100]

        motions[index, :, :] = imu

    return motions, np.array(motions_labels)


motions_train, motions_train_labels = load_data("train")
motions_test, motions_test_labels = load_data("test")

rocket = MiniRocketClassifier()
rocket.fit(motions_train, motions_train_labels)
y_pred = rocket.predict(motions_test)
accuracy = accuracy_score(motions_test_labels, y_pred)

print(accuracy)
print(y_pred)

plt.plot(motions_test[0][0])
plt.show()
