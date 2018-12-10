import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def angle_diff(a, b):
    # https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles
    return ((a-b) + np.pi) % (2*np.pi) - np.pi

def report_labels(labels, y):
    # print out the total number of examples
    total = len(y)
    print('Total examples: {}'.format(total))

    # print out the proportion of labels in each category
    for k in range(np.max(y)+1):
        this_cat = np.sum(y==k)
        print('{}: {:0.1f}%'.format(labels[k], 100 * this_cat / total))

def report_labels_regression(y, units, filename):
    # print out the total number of examples
    total = len(y)
    print('Total examples: {}'.format(total))

    # histogram plotting
    # ref: matplotlib documentation
    plt.hist(y, 10, facecolor='g', alpha=0.75)
    plt.xlabel(units)
    plt.ylabel('Count')
    plt.grid(True)

    # save histogram to a file
    plt.savefig(filename, bbox_inches='tight')

def report_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: {:0.3f}'.format(accuracy))

def report_model_regression(y_test, y_pred, units):
    std = np.std(y_test-y_pred)
    print('Error ({}): +/- {:0.3f}'.format(units, std))

def train_experiment_regression(train, units, trials=30):
    std = []

    for _ in range(trials):
        _, y_test, y_pred = train()
        std.append(np.std(y_test-y_pred))

    print('Ran {} trials.'.format(trials))
    print('Error ({}): +/- {:0.3f}'.format(units, np.mean(std)))

def train_experiment(train, trials=30):
    results = []

    for _ in range(trials):
        _, y_test, y_pred = train()
        accuracy = accuracy_score(y_test, y_pred)
        results.append(accuracy)

    ave = np.mean(results)
    std = np.std(results)

    print('Ran {} trials.'.format(trials))
    print('Accuracy: {:0.3f} +/- {:0.3f}'.format(ave, 2*std))