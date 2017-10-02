import numpy as np
import math
import datetime
from sklearn.datasets import load_iris
import random
from gaussian import Util
from GMM import GMM
import matplotlib.pyplot as plt


Ng = 3


def compare(data_test, target_test, models):
    util = Util(Ng)
    accepts = []
    for ix in range(0, len(data_test)):
        data = data_test[ix]
        target = target_test[ix]
        best_p = 0
        best_class = -1
        for c in range(0, len(models)):
            p = util.sum_weighted_gauss(data, models[c])
            if p > best_p:
                best_p = p
                best_class = c

        accepts.append(best_class == target)

    taxa_accepts = sum(accepts) / len(data_test)
    taxa_errors = (len(data_test) - sum(accepts)) / len(data_test)

    print("Taxa de acertos: %f" % taxa_accepts)
    print("Taxa de errors: %f" % taxa_errors)

    return taxa_accepts, taxa_errors


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

def get_mean_model(X, models):
    util = Util(Ng)
    acc = []
    for m in models:
        aux = math.e ** util.get_prob(X, m)
        acc.append(aux)
    return math.log(sum(acc))


def get_impostor_model(lam_client):
    m_arr = lam_client[0]
    u_arr = lam_client[1]
    e_arr = lam_client[2]

    alpha = 0.1
    scale = 1

    m_impostor = []
    for m in m_arr:
        r = (alpha * m + (1 - alpha) * m) * scale
        m_impostor.append(r)

    u_impostor = []
    for u in u_arr:
        r = alpha * u + (1 - alpha) * u
        u_impostor.append(r)

def plot():
    startTime = datetime.datetime.now()

    executions = 2
    xAxis = np.arange(executions)

    execs = [exec() for e in range(0, executions)]

    accepts, errors = zip(*execs)

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


def exec():
    startTime = datetime.datetime.now()

    data, target = load_iris(True)

    # from cross_validation import CrossValidation
    # cross_validator = CrossValidation(0.2)
    # X_train, y_train, X_test, y_test = cross_validator.exec(data, target)

    X_train, y_train, X_test, y_test = cross_validation(data, target, 0.2)

    c0, c1, c2 = X_train

    orq1 = np.array(c0)
    orq2 = np.array(c1)
    orq3 = np.array(c2)

    startTime_models = datetime.datetime.now()

    gmm = GMM(Ng)

    modelOrq1 = gmm.model(orq1)
    modelOrq2 = gmm.model(orq2)
    modelOrq3 = gmm.model(orq3)

    models = [modelOrq1, modelOrq2, modelOrq3]

    finishTime_models = datetime.datetime.now()

    duration_models = finishTime_models - startTime_models
    print('Duration Creation models')
    print(duration_models)

    startTime_compare = datetime.datetime.now()
    result = compare(X_test, y_test, models)
    finishTime_models = datetime.datetime.now()

    duration_compare = finishTime_models - startTime_compare
    print('Duration Compare')
    print(duration_compare)

    finishTime = datetime.datetime.now()

    duration = finishTime - startTime
    print('Duration parcial: ')
    print(duration)
    print('\n \n')

    return result

plot()
