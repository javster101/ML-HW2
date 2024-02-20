#!/usr/bin/env python3

import numpy as np
from numpy.core.fromnumeric import ravel
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB, MultinomialNB
import matplotlib.pyplot as plt

# Define variables for processing

# True for mushroom dataset, False for sentences
mushrooms = False
if mushrooms:
    X = pd.read_csv('X_msrm.csv')
    Y = pd.read_csv('y_msrm.csv')
    random_seeds = [45]
    classifier_type = CategoricalNB
    log_terms = 200
    test_size = 0.20
else:
    X = pd.read_csv('X_snts.csv')
    Y = pd.read_csv('y_snts.csv')
    random_seeds = [1, 923478, 790, 172309, 5897392, 19283, 98230, 1391283, 4932840, 6795876, 16626]
    classifier_type = MultinomialNB
    log_terms = 50
    test_size = 0.2

print(f"Feature shape is {X.shape}, classes shape is {Y.shape}")

# Storage for graphing by alpha
accs = []
first_alphas = []
first_accs_by_alpha = []
first_aucs_by_alpha= []
first_f1s_by_alpha = []

first = True
for random_num in random_seeds:
    # Generate a new random distribution
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state = random_num)
    y_train = ravel(y_train)
    y_test = ravel(y_test)

    acc = 0
    f1 = 0
    auc = 0
    good_alpha = 0
    best_params = {}

    for alpha in np.logspace(-15, 5, base=2, num=log_terms):

        # Create classifier and classify dataset
        classifier = classifier_type(alpha=alpha, force_alpha=True)
        classifier.fit(x_train, y_train)
        
        y_pred = classifier.predict(x_test)
        
        this_acc = accuracy_score(y_test, y_pred)
        if mushrooms:
            this_f1 = f1_score(y_test ,y_pred)
            this_auc = roc_auc_score(y_test, classifier.predict_proba(x_test)[:,1])
            if this_auc > auc:
                acc = this_acc
                f1 = this_f1
                auc = this_auc
                good_alpha = alpha
                best_params = classifier.feature_log_prob_
        else:
            if this_acc > acc:
                acc = this_acc
                good_alpha = alpha
                best_params = classifier.feature_log_prob_

        if first:
            first_alphas.append(alpha)
            first_accs_by_alpha.append(this_acc)
            if mushrooms:
                first_aucs_by_alpha.append(this_auc)
                first_f1s_by_alpha.append(this_f1)

    print(f"Done with random {random_num}")
    print(f"Best alpha is {good_alpha}")
    if first:
        print("Accuracy: ", acc)
        if mushrooms:
            print("AUC: ", auc)
            print("F1: ", f1)
        else:
            print(best_params)

    first = False
    accs.append(acc)

stdev = np.std(accs)
print(f"Average accuracy of {np.average(accs)}")
print(f"Stdev of accuracy {np.std(accs)}")

# Convert arrays to np arrays
first_alphas = np.array(first_alphas)
first_accs_by_alpha = np.array(first_accs_by_alpha)
first_aucs_by_alpha = np.array(first_aucs_by_alpha)
first_f1s_by_alpha = np.array(first_f1s_by_alpha)

if mushrooms:

    # Graph Q3 graphs
    fig, axes = plt.subplots(nrows=3, ncols=1)
    auc_ax = axes[0]
    acc_ax = axes[1]
    f1_ax = axes[2]

    auc_ax.plot(first_alphas, first_aucs_by_alpha)
    auc_ax.set_xscale('log', base=2)
    auc_ax.set_xlabel('Alpha')
    auc_ax.set_ylabel('ROC AUC')

    acc_ax.plot(first_alphas, first_accs_by_alpha)
    acc_ax.set_xscale('log', base=2)
    acc_ax.set_xlabel('Alpha')
    acc_ax.set_ylabel('Accuracy')

    f1_ax.plot(first_alphas, first_f1s_by_alpha)
    f1_ax.set_xscale('log', base=2)
    f1_ax.set_xlabel('Alpha')
    f1_ax.set_ylabel('F1')
else:

    # Graph Q4 graphs
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(first_alphas, first_accs_by_alpha)
    ax.fill_between(first_alphas, (first_accs_by_alpha-stdev), (first_accs_by_alpha+stdev), color='b', alpha=.1)
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Accuracy')

plt.show()
