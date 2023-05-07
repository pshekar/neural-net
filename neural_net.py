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
# file_name = 'hw3_cancer.csv'

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
        accuracy = 0
        precision = 0
        recall = 0
        for i in range(k_folds):
            training_set, testing_set = stratified_kfold(data_set, k_folds, i)
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
                classifications = training_set[:, :1]
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
                training_set = training_set[:, :-1]
                testing_set = testing_set[:, :-1]
            elif file_name == 'hw3_wine.csv':
                training_set = training_set[:, 1:]
                testing_set = testing_set[:, 1:]
            if file_name == 'hw3_wine.csv' or file_name == 'hw3_cancer.csv':
                training_set = normalize(training_set)
            final_weights = back_propogate(weights, training_set, expected_outputs, net_shape, regularization)
            testing_accuracy, testing_precision, testing_recall = test(final_weights, testing_set, expected_outputs)
            accuracy += testing_accuracy
            precision += testing_precision
            recall += testing_recall

        accuracy = accuracy / k_folds
        precision = precision / k_folds
        recall = recall / k_folds
        if precision == 0 and recall == 0:
            f1 = 0
        else:
            f1 = (2 * precision * recall)/(precision + recall)
        print(f'testing set accuracy: {accuracy}')
        print(f'testing set precision: {precision}')
        print(f'testing set recall: {recall}')
        print(f'testing set f1 score: {f1}')


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
    true_pos = 0
    true_pos_1 = 0
    true_pos_2 = 0
    true_pos_3 = 0
    true_neg = 0
    true_neg_1 = 0
    true_neg_2 = 0
    true_neg_3 = 0
    false_pos = 0
    false_pos_1 = 0
    false_pos_2 = 0
    false_pos_3 = 0
    false_neg = 0
    false_neg_1 = 0
    false_neg_2 = 0
    false_neg_3 = 0
    class_1 = 0
    class_2 = 0
    class_3 = 0
    for i in range(len(data_set)):
        output, _ = forward_propogate(weights, data_set[i])
        argmax = np.argmax(output)
        expected_argmax = np.argmax(expected_outputs[i])
        if expected_argmax == argmax:
            correct += 1 
        if file_name == 'hw3_wine.csv':
            if expected_argmax == 0:
                class_1 += 1
                if argmax == 0:
                    true_pos_1 += 1
                    true_neg_2 += 1
                    true_neg_3 += 1
                elif argmax == 1:
                    false_neg_1 += 1
                    false_pos_2 += 1
                    true_neg_3 += 1
                else:
                    false_neg_1 += 1
                    true_neg_2 += 1
                    false_pos_3 += 1
            elif expected_argmax == 1:
                class_2 += 1
                if argmax == 0:
                    false_pos_1 += 1
                    false_neg_2 += 1
                    true_neg_3 += 1
                elif argmax == 1:
                    true_neg_1 += 1
                    true_pos_2 += 1
                    true_neg_3 += 1
                else:
                    true_neg_1 += 1
                    false_neg_2 += 1
                    false_pos_3 += 1
            else:
                class_3 += 1
                if argmax == 0:
                    false_pos_1 += 1
                    true_neg_2 += 1
                    false_neg_3 += 1
                elif argmax == 1:
                    true_neg_1 += 1
                    false_pos_2 += 1
                    false_neg_3 += 1
                else:
                    true_neg_1 += 1
                    true_neg_2 += 1
                    true_pos_3 += 1
        else:
            if argmax == expected_argmax:
                if argmax == 0:
                    true_pos += 1
                else:
                    true_neg += 1
            else:
                if expected_argmax == 0:
                    false_neg += 1
                else:
                    false_pos += 1  
    if file_name == 'hw3_wine.csv':
        if true_pos_1 == 0 and false_pos_1 == 0:
            precision_1 = 0
        else:
            precision_1 = true_pos_1/(true_pos_1 + false_pos_1)
        if true_pos_1 == 0 and false_neg_1 == 0:
            recall_1 = 0
        else:
            recall_1 = true_pos_1/(true_pos_1 + false_neg_1)
        if true_pos_2 == 0 and false_pos_2 == 0:
            precision_2 = 0
        else:
            precision_2 = true_pos_2/(true_pos_2 + false_pos_2)
        if true_pos_2 == 0 and false_neg_2 == 0:
            recall_2 = 0
        else:
            recall_2 = true_pos_2/(true_pos_2 + false_neg_2)
        if true_pos_3 == 0 and false_pos_3 == 0:
            precision_3 = 0
        else:
            precision_3 = true_pos_3/(true_pos_3 + false_pos_3)
        if true_pos_3 == 0 and false_neg_3 == 0:
            recall_3 = 0
        else:
            recall_3 = true_pos_3/(true_pos_3 + false_neg_3)
        testing_accuracy = (true_pos_1 + true_pos_2 + true_pos_3)/len(data_set)
        testing_precision = (precision_1 + precision_2 + precision_3)/3
        testing_recall = (recall_1 + recall_2 + recall_3)/3
    else:
        testing_accuracy = (true_pos + true_neg)/len(data_set)
        if true_pos == 0 and false_pos == 0:
            testing_precision = 0
        else:
            testing_precision = true_pos/(true_pos + false_pos)
        if true_pos == 0 and false_neg == 0:
            testing_recall = 0
        else:
            testing_recall = true_pos/(true_pos + false_neg)

    return testing_accuracy, testing_precision, testing_recall

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
        loops = 250
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
    # BACKPROP_EXAMPLE 1
    # regularization = 0
    # net_shape = [1, 2, 1]
    # benchmark1(regularization, net_shape)
    
    
    # BACKPROP_EXAMPLE 2
    # regularization = 0.25
    # net_shape = [2, 4, 3, 2]
    # benchmark2(regularization, net_shape)


    # MAIN FUNCTION
    k_folds = 10
    regularization = 0.25
    # HOUSE_VOTES
    net_shape = [16, 10, 10, 10, 2]
    # WINE
    # net_shape = [13, 10, 10, 10, 3]
    # CANCER
    # net_shape = [9, 5, 5, 2]
    main(regularization, net_shape, k_folds)