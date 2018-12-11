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

def report_model(clf, X_train, X_test, y_train, y_test):
    train_accuracy = accuracy_score(y_train, clf.predict(X_train))
    print('Train error: {:0.1f}%'.format(1e2 * (1 - train_accuracy)))

    test_accuracy = accuracy_score(y_test, clf.predict(X_test))
    print('Test error: {:0.1f}%'.format(1e2 * (1 - test_accuracy)))

    n_train = X_train.shape[0]
    print('N_train: {}'.format(n_train))

    n_test = X_test.shape[0]
    print('N_test: {}'.format(n_test))

def report_model_regression(units, clf, X_train, X_test, y_train, y_test):
    train_error = np.std(y_train-clf.predict(X_train))
    print('Train error: sigma={:0.3f} {}'.format(train_error, units))

    test_error = np.std(y_test-clf.predict(X_test))
    print('Test error: sigma={:0.3f} {}'.format(test_error, units))

    n_train = X_train.shape[0]
    print('N_train: {}'.format(n_train))

    n_test = X_test.shape[0]
    print('N_test: {}'.format(n_test))

# def train_experiment_regression(train, units, trials=30):
#     std = []
#
#     for _ in range(trials):
#         _, y_test, y_pred = train()
#         std.append(np.std(y_test-y_pred))
#
#     print('Ran {} trials.'.format(trials))
#     print('Error ({}): +/- {:0.3f}'.format(units, np.mean(std)))

# def train_experiment(train, trials=30):
#     results = []
#
#     for _ in range(trials):
#         _, y_test, y_pred = train()
#         accuracy = accuracy_score(y_test, y_pred)
#         results.append(accuracy)
#
#     ave = np.mean(results)
#     std = np.std(results)
#
#     print('Ran {} trials.'.format(trials))
#     print('Accuracy: {:0.3f} +/- {:0.3f}'.format(ave, 2*std))