from copy import deepcopy
import csv
from math import sqrt, log2, floor
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from statistics import stdev

cur_path = os.path.dirname(__file__)
file_name = 'hw3_house_votes_84.csv'
# file_name = 'hw3_wine.csv'

def main(regularization, net_shape, k_folds):
    with open(f'{cur_path}/datasets/{file_name}', newline='') as csvfile:
        data = csvfile.read().splitlines()
        if file_name == 'hw3_house_votes_84.csv' or file_name == 'weather_data.csv':
            categories = np.array(data[0].split(','))
        else:
            categories = np.array(data[0].split('\t'))
        data = data[1:]
        # strip the classification from the categories
        if file_name == 'hw3_wine.csv':
            categories = categories[1:]
        else:
            categories = categories[:-1]
        # testing_categories = deepcopy(categories)
        # testing_accuracy = 0
        # testing_precision = 0
        # testing_recall = 0
        data_set = []
        for row in data:
            if file_name == 'hw3_house_votes_84.csv':
                arr = row.split(',')
                for i in range(len(arr)):
                    arr[i] = int(arr[i])
            else:
                if file_name == 'hw3_cancer.csv':
                    arr = row.split('\t')
                    for i in range(len(arr)):
                        arr[i] = int(float(arr[i]))
                else:
                    arr = row.split('\t')
                    for i in range(len(arr)):
                        arr[i] = float(arr[i])
            data_set.append(np.array(arr))
        data_set = np.array(data_set)
        # normalize/regularize data?
        # stratified cross validation
        weights = set_weights(net_shape)
        # for i in range(k_folds):
        training_set, testing_set = stratified_kfold(data_set, k_folds, 0)
        expected_outputs = []
        if file_name == 'hw3_house_votes_84.csv' or file_name == 'hw3_cancer.csv':
            classifications = training_set[:, -1:]
            for row in classifications:
                output = np.zeros(2)
                if row == 0:
                    output[0] = 1
                else:
                    output[1] = 1
                expected_outputs.append(output)
        elif file_name == 'hw3_wine.csv':
            classifications = training_set[:, 1:]
            for row in classifications:
                output = np.zeros(3)
                if row == 1:
                    output[0] = 1
                elif row == 2:
                    output[1] = 1
                else:
                    output[2] = 1
                expected_outputs.append(output)

        training_set = training_set[:, :-1]
        back_propogate(weights, training_set, expected_outputs, net_shape, regularization)

        

def stratified_kfold(data, k, test):
    classes = {}
    if file_name == 'hw3_house_votes_84.csv' or file_name == 'weather_data':
        file_index = -1
    else:
        file_index = 0
    for row in data:
        idx = int(row[file_index])
        if idx not in classes:
            classes[idx] = []
            classes[idx].append(row)
        else:
            classes[idx].append(row)

    for _, cls in classes.items():
        arr = np.array(cls)
        np.random.shuffle(arr)
        cls = list(arr)
    fold_size = int(len(data) / k)
    folds = []
    for i in range(k):
        fold = []
        for idx, cls in classes.items():
            ratio = len(classes[idx])/len(data)
            fold_range = int(fold_size * ratio)+1
            indices = cls[int(i*fold_range):int((i+1)*fold_range)]
            for row in indices:
                fold.append(row)
        folds.append(fold)

    testing_data = []
    for i in range(len(folds)):
        if i != test:
            testing_data = testing_data + folds[i]

    return np.array(testing_data), np.array(folds[i])


def set_weights(net_shape):
    weights = []
    for i in range(len(net_shape)-1):
        weights_i = np.random.randn(net_shape[i]+1, net_shape[i+1])
        weights.append(weights_i.T)
    return weights

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward_propogate(weights, inputs):
    outputs = []
    output = np.insert(inputs, 0, 1)
    outputs.append(output)
    for i in range(len(weights) - 1):
        output = np.matmul(weights[i], output)
        output = sigmoid(output)
        output = np.insert(output, 0, 1)
        outputs.append(output)

    output = np.matmul(weights[-1], output)
    output = sigmoid(output)
    outputs.append(output)
    return output, outputs

def cost_fn(expected_output, output):
    part1 = -1 * np.multiply(expected_output, np.log(output))
    part2 = np.multiply(np.subtract(1, expected_output), np.log(np.subtract(1,output)))
    cost = np.sum(part1 - part2) 
    return cost

def regularize(weights, inputs, regularization):
    square_weights = 0
    for matrix in weights:
        for row in matrix:
            for i in range(len(row)-1):
                square_weights += (row[i] * row[i])

    regularized = (regularization / (2 * len(inputs))) * square_weights
    return regularized

def back_propogate(weights, inputs, expected_outputs, net_shape, regularization):
    # initialize big ol D
    weights_copy = deepcopy(weights)
    gradients = []
    for i in range(len(weights)):
        gradients.append(np.zeros(np.array(weights_copy[i]).shape))
    for k in range(len(inputs)):
        deltas = []
        output, outputs = forward_propogate(weights, inputs[k])
        print('back')
        output_delta = output - expected_outputs[k]
        print(f'output_delta: {np.array([np.array(output_delta)])=}')
        deltas.append(np.array([np.array(output_delta)]))
        for i in range(len(net_shape) - 2):
            idx = len(net_shape) - i - 2
            # print(idx)
            # print(f'{np.array(weights[idx]).T=}')
            # print(f'{deltas[i]=}')
            layer_delta = np.matmul(np.array(weights[idx]).T, deltas[i].T)
            layer_delta = np.multiply(layer_delta, np.array([outputs[idx]]).T)
            layer_delta = np.multiply(layer_delta, np.array([np.subtract(1,outputs[idx])]).T)
            layer_delta = layer_delta[1:]
            deltas.append(layer_delta.T)
        print(f'{deltas=}')
        for i in range(len(net_shape)-1):
            idx = len(net_shape) - i - 2
            # print(idx)
            output = np.array([np.array(outputs[idx])])
            # print(f'gradient: {np.matmul(deltas[i].T, output)=}')
            gradients[idx] = np.add(gradients[idx], np.matmul(deltas[i].T, output))

    for i in range(len(net_shape)-1):
        idx = len(net_shape) - i - 2
        p = np.multiply(regularization, weights[idx])
        # set first column to zeros
        p[:, 0] =  0
        gradients[idx] = (gradients[idx] + p) / len(inputs)
    print(f'{gradients=}')
    for i in range(len(net_shape)-1):
        idx = len(net_shape) - i - 2
        weights[idx] = weights[idx] - (np.multiply(1, gradients[idx]))


def benchmark1(regularization, net_shape):    
    inputs = [[0.13000], [.42000]]
    expected_outputs = [[0.9000], [.23000]]
    input2 = [.42000]
    expected_output2 = [.23000]
    # weights = set_weights(inputs, net_shape)
    weights = [[[0.40000,  0.10000], [0.30000,  0.20000  ]], [[0.70000,  0.50000,  0.60000]]]
    output = back_propogate(weights, inputs, expected_outputs, net_shape, regularization)
    # cost = cost_fn(expected_output1, output1)
    # output2, _ = forward_propogate(weights, input2)
    # cost += cost_fn(expected_output2, output2)
    # cost = cost / 2
    # regularized = regularize(weights, input1, regularization)
    # regularized_cost = cost + regularized
    # print(f'{regularized_cost=}')

def benchmark2(regularization, net_shape):
    inputs = [[0.32000, 0.68000], [0.83000, 0.02000]]
    expected_outputs = [[0.75000, 0.98000], [0.75000, 0.28000]]
    weights = [[[0.42000, 0.15000, 0.40000], [0.72000, 0.10000, 0.54000], [0.01000, 0.19000, 0.42000],	[0.30000, 0.35000, 0.68000]], 
               [[0.21000, 0.67000, 0.14000, 0.96000, 0.87000], [0.87000, 0.42000, 0.20000, 0.32000, 0.89000], [0.03000, 0.56000, 0.80000, 0.69000, 0.09000]],
               [[0.04000, 0.87000, 0.42000, 0.53000], [0.17000, 0.10000, 0.95000, 0.69000]]]
    output = back_propogate(weights, inputs, expected_outputs, net_shape, regularization)
    # cost = cost_fn(expected_output1, output1)
    # output2, _ = forward_propogate(weights, input2)
    # cost += cost_fn(expected_output2, output2)
    # cost = cost / 2
    # regularized = regularize(weights, input1, regularization)
    # regularized_cost = cost + regularized
    # print(f'{regularized_cost=}')


if __name__ == '__main__':
    regularization = 0
    # net_shape = [1, 2, 1]
    # benchmark1(regularization, net_shape)
    # regularization = 0.25
    # net_shape = [2, 4, 3, 2]
    # benchmark2(regularization, net_shape)
    k_folds = 10
    net_shape = [16, 10, 5, 3, 2]
    main(regularization, net_shape, k_folds)