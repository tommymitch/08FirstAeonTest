import os

import matplotlib.pyplot as plt
import numpy as np
import ximu3csv

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

    return motions, motions_labels


motions_train, motions_train_labels = load_data("train")
motions_test, motions_test_labels = load_data("test")
print(motions_train.shape)
print(motions_train_labels)

print(motions_test.shape)
print(motions_test_labels)


plt.plot(motions_test[0][0])
plt.show()
