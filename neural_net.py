from copy import deepcopy
import csv
from math import sqrt, log2, floor
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from statistics import stdev

cur_path = os.path.dirname(__file__)
# file_name = 'hw3_house_votes_84.csv'
file_name = 'hw3_wine.csv'

def main(regularization, net_shape, k_folds):
    with open(f'{cur_path}/datasets/{file_name}', newline='') as csvfile:
        data = csvfile.read().splitlines()
        # exclude attributes
        data = data[1:]
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
        # stratified cross validation
        weights = set_weights(net_shape)
        # for i in range(k_folds):
        # training_set, testing_set = stratified_kfold(data_set, k_folds, 0)
        expected_outputs = []
        if file_name == 'hw3_house_votes_84.csv' or file_name == 'hw3_cancer.csv':
            classifications = data_set[:, -1:]
            for row in classifications:
                output = np.zeros(2)
                if row == 0:
                    output[0] = 1
                else:
                    output[1] = 1
                expected_outputs.append(output)
        elif file_name == 'hw3_wine.csv':
            classifications = data_set[:, :1]
            for row in classifications:
                output = np.zeros(3)
                if row == 1:
                    output[0] = 1
                elif row == 2:
                    output[1] = 1
                else:
                    output[2] = 1
                expected_outputs.append(output)
        if file_name == 'hw3_house_votes_84.csv' or file_name == 'hw3_cancer.csv':
            training_set = data_set[:, :-1]
        elif file_name == 'hw3_wine.csv':
            training_set = data_set[:, 1:]
            training_set = normalize(training_set)
        test(weights, training_set, expected_outputs)
        final_weights = back_propogate(weights, training_set, expected_outputs, net_shape, regularization)
        test(final_weights, training_set, expected_outputs)


def normalize(data_set):
    for column in range(len(data_set[0])):
        a_max = float('-inf')
        a_min = float('inf')
        for row in range(len(data_set)):
            a_max = max(a_max, data_set[row][column])
            a_min = min(a_min, data_set[row][column])
        for i in range(len(data_set)):
            data_set[i][column] = (data_set[i][column] - a_min) / (a_max - a_min)
    return data_set

def test(weights, data_set, expected_outputs):
    correct = 0
    for i in range(len(data_set)):
        output, _ = forward_propogate(weights, data_set[i])
        # print(output)
        argmax = np.argmax(output)
        # print(argmax)
        expected_argmax = np.argmax(expected_outputs[i])
        # print(expected_outputs[i])
        # print(expected_argmax)
        if expected_argmax == argmax:
            correct += 1 
    print(f'accuracy: {correct / len(data_set)}')

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

def forward_propogate(weights, inputs, benchmark=False):
    outputs = []
    output = np.insert(inputs, 0, 1)
    if benchmark:
        print(f'a1: {output}')
    outputs.append(output)
    for i in range(len(weights) - 1):
        output = np.matmul(weights[i], output)
        if benchmark:
            print(f'z{i+2}: {output}')
        output = sigmoid(output)
        output = np.insert(output, 0, 1)
        if benchmark:
            print(f'a{i+2}: {output}')
        outputs.append(output)

    output = np.matmul(weights[-1], output)
    if benchmark:
       print(f'z{i+3}: {output}')
    output = sigmoid(output)
    if benchmark:
        print(f'a{i+3}: {output}')
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
            for i in range(len(row)):
                if i == 0:
                    continue
                square_weights += (row[i] * row[i])

    regularized = (regularization / (2 * len(inputs))) * square_weights
    return regularized

def back_propogate(weights, inputs, expected_outputs, net_shape, regularization, benchmark=False):
    # initialize big ol D
    weights_copy = deepcopy(weights)
    gradients = []
    for i in range(len(weights)):
        gradients.append(np.zeros(np.array(weights_copy[i]).shape))

    if benchmark:
        loops = 1
    else:
        loops = 1000
    for _ in range(loops):
        for k in range(len(inputs)):
            if benchmark:
                print(f'Computing gradients based on training instance {k+1}')
            deltas = []
            output, outputs = forward_propogate(weights, inputs[k])
            output_delta = output - expected_outputs[k]
            if benchmark:
                print(f'delta{len(net_shape)}: {output_delta}')
            deltas.append(np.array([np.array(output_delta)]))
            for i in range(len(net_shape) - 2):
                idx = len(net_shape) - i - 2
                layer_delta = np.matmul(np.array(weights[idx]).T, deltas[i].T)
                layer_delta = np.multiply(layer_delta, np.array([outputs[idx]]).T)
                layer_delta = np.multiply(layer_delta, np.array([np.subtract(1,outputs[idx])]).T)
                layer_delta = layer_delta[1:]
                if benchmark:
                    print(f'delta{len(net_shape)-i-1}: {layer_delta.T}')
                deltas.append(layer_delta.T)
            for i in range(len(net_shape)-1):
                idx = len(net_shape) - i - 2
                # print(idx)
                output = np.array([np.array(outputs[idx])])
                if benchmark:
                    print(f'Gradients of Theta{idx+1} based on training instance {k+1}:')
                    print(f'{np.matmul(deltas[i].T, output)}')
                    print()
                gradients[idx] = np.add(gradients[idx], np.matmul(deltas[i].T, output))

        if benchmark:
            print(f'The entire training set has been processes. Computing the average (regularized) gradients:')
        for i in range(len(net_shape)-1):
            idx = len(net_shape) - i - 2
            p = np.multiply(regularization, weights[idx])
            # set first column to zeros
            p[:, 0] =  0
            gradients[idx] = (gradients[idx] + p) / len(inputs)
        
        if benchmark:
            for i in range(len(net_shape)-1):
                print(f'Final regularized gradients of Theta{i+1}:')
                print(f'{gradients[i]}')
                print()
        # print(f'{gradients=}')
        for i in range(len(net_shape)-1):
            idx = len(net_shape) - i - 2
            weights[idx] = weights[idx] - (np.multiply(1, gradients[idx]))

    return weights


def benchmark1(regularization, net_shape):    
    inputs = [[0.13000], [.42000]]
    expected_outputs = [[0.9000], [.23000]]
    # weights = set_weights(inputs, net_shape)
    weights = [[[0.40000,  0.10000], [0.30000,  0.20000  ]], [[0.70000,  0.50000,  0.60000]]]
    # output = back_propogate(weights, inputs, expected_outputs, net_shape, regularization, True)
    # cost = cost_fn(expected_output1, output1)
    costs = 0
    for i in range(len(inputs)):
        print(f'Forward propagate of instance {i+1}')
        output, _ = forward_propogate(weights, inputs[i], True)
        print(f'f(x) = {output}')
        print(f'Predicted output for instance {i+1}: {output}')
        print(f'Expected output for instance {i+1}: {expected_outputs[i]}')
        cost = cost_fn(expected_outputs[i], output)
        print(f'Cost, J, associated with instance {i+1}: {cost}')
        costs += cost
        print()
    costs = costs / len(inputs)
    regularized = regularize(weights, inputs, regularization)
    regularized_cost = costs + regularized
    print(f'Final (regularized) cost, J, based on the complete training set: {regularized_cost}')
    print()
    print(f'Running backpropagation')
    back_propogate(weights, inputs, expected_outputs, net_shape, regularization, True)

def benchmark2(regularization, net_shape):
    inputs = [[0.32000, 0.68000], [0.83000, 0.02000]]
    expected_outputs = [[0.75000, 0.98000], [0.75000, 0.28000]]
    weights = [[[0.42000, 0.15000, 0.40000], [0.72000, 0.10000, 0.54000], [0.01000, 0.19000, 0.42000],	[0.30000, 0.35000, 0.68000]], 
               [[0.21000, 0.67000, 0.14000, 0.96000, 0.87000], [0.87000, 0.42000, 0.20000, 0.32000, 0.89000], [0.03000, 0.56000, 0.80000, 0.69000, 0.09000]],
               [[0.04000, 0.87000, 0.42000, 0.53000], [0.17000, 0.10000, 0.95000, 0.69000]]]
    # output = back_propogate(weights, inputs, expected_outputs, net_shape, regularization, True)
    costs = 0
    for i in range(len(inputs)):
        print(f'Forward propagate of instance {i+1}')
        output, _ = forward_propogate(weights, inputs[i], True)
        print(f'f(x) = {output}')
        print(f'Predicted output for instance {i+1}: {output}')
        print(f'Expected output for instance {i+1}: {expected_outputs[i]}')
        cost = cost_fn(expected_outputs[i], output)
        print(f'Cost, J, associated with instance {i+1}: {cost}')
        costs += cost
        print()
    costs = costs / len(inputs)
    regularized = regularize(weights, inputs, regularization)
    regularized_cost = costs + regularized
    print(f'Final (regularized) cost, J, based on the complete training set: {regularized_cost}')
    print()
    print(f'Running backpropagation')
    back_propogate(weights, inputs, expected_outputs, net_shape, regularization, True)


if __name__ == '__main__':
    # regularization = 0
    # net_shape = [1, 2, 1]
    # benchmark1(regularization, net_shape)
    regularization = 0.25
    net_shape = [2, 4, 3, 2]
    benchmark2(regularization, net_shape)
    k_folds = 10
    # regularization = 0
    # net_shape = [16, 10, 10, 10, 2]
    # net_shape = [13, 10, 10, 10, 3]
    # main(regularization, net_shape, k_folds)