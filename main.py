import numpy as np
import datetime
from sklearn.datasets import load_iris
import random
import pandas as pd
from GMM import GMM
import matplotlib.pyplot as plt
from comparator import Comparator


Ng = 3

def cross_validation(dataset, target, test_size):
    c0 = dataset[0:50]
    c1 = dataset[50:100]
    c2 = dataset[100:150]

    t0 = target[0:50]
    t1 = target[50:100]
    t2 = target[100:150]

    n_interations = round(len(c0) * test_size)
    test_arr = []
    test_tar = []

    for i in range(0, n_interations):
        rand_i = random.randint(0, len(c0) - 1)
        test_arr.append(c0[rand_i])
        test_tar.append(t0[rand_i])
        c0 = np.delete(c0, rand_i, axis=0)
        t0 = np.delete(t0, rand_i, axis=0)

    for i in range(0, n_interations):
        rand_i = random.randint(0, len(c1) - 1)
        test_arr.append(c1[rand_i])
        test_tar.append(t1[rand_i])
        c1 = np.delete(c1, rand_i, axis=0)
        t1 = np.delete(t1, rand_i, axis=0)

    for i in range(0, n_interations):
        rand_i = random.randint(0, len(c2) - 1)
        test_arr.append(c2[rand_i])
        test_tar.append(t2[rand_i])
        c2 = np.delete(c2, rand_i, axis=0)
        t2 = np.delete(t2, rand_i, axis=0)

    return np.array([c0, c1, c2]), np.array([t0, t1, t2]), test_arr, test_tar

def plot():
    startTime = datetime.datetime.now()

    executions = 1
    xAxis = np.arange(executions)

    execs = [exec_with_time() for e in range(0, executions)]

    accepts, errors, impostors = zip(*execs)

    finishTime = datetime.datetime.now()

    duration = finishTime - startTime
    print('Duration Total')
    print(duration)


    fig, ax = plt.subplots()

    rects1 = ax.bar(xAxis, accepts, color='b')
    rects2 = ax.bar(xAxis, errors, color='r')

    ax.set_ylabel('Taxa de acerto')
    ax.set_title('GMM')

    ax.legend((rects1[0], rects2[0]), ('Acerto', 'Erro'))

    plt.show()


def exec_with_time():
    startTime = datetime.datetime.now()
    result = exec()
    finishTime = datetime.datetime.now()

    duration = finishTime - startTime
    print('Duration parcial: ')
    print(duration)
    print('\n')

    return result


def models_with_bms(x_train):
    c0, c1, c2 = x_train

    orq1 = np.array(c0)
    orq2 = np.array(c1)
    orq3 = np.array(c2)

    impostorOrq1 = np.concatenate([c1, c2])
    impostorOrq2 = np.concatenate([c0, c2])
    impostorOrq3 = np.concatenate([c0, c1])

    gmm = GMM(Ng)

    return [(gmm.model(orq1), gmm.model(impostorOrq1)),
            (gmm.model(orq2), gmm.model(impostorOrq2)),
            (gmm.model(orq3), gmm.model(impostorOrq3))]


def models_with_universal(x_train):
    c0, c1, c2 = x_train

    orq1 = np.array(c0)
    orq2 = np.array(c1)
    orq3 = np.array(c2)

    impostorUniversal = np.concatenate([c0, c1, c2])
    gmm = GMM(Ng)
    universal = gmm.model(impostorUniversal)

    return [(gmm.model(orq1), universal),
            (gmm.model(orq2), universal),
            (gmm.model(orq3), universal)]

def exec():

    comparator = Comparator(Ng)
    data, target = load_iris(True)
    iris = load_iris()

    frame = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
    groups = frame.groupby(['target'])
    d1 = frame.iloc[:, 0]

    # from cross_validation import CrossValidation
    # cross_validator = CrossValidation(0.2)
    # X_train, y_train, X_test, y_test = cross_validator.exec(data, target)

    X_train, y_train, X_test, y_test = cross_validation(data, target, 0.2)
    result = comparator.sumary(X_test, y_test, models_with_universal(X_train))

    return result

plot()
