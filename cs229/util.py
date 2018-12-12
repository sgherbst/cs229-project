import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def angle_diff(a, b):
    # https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles
    return ((a-b) + np.pi) % (2*np.pi) - np.pi

def report_labels_classification(y, label_names):
    # print out the total number of examples
    total = len(y)
    print('Total examples: {}'.format(total))

    # print out the proportion of labels in each category
    for k in range(np.max(y)+1):
        this_cat = np.sum(y==k)
        print('{}: {:0.1f}%'.format(label_names[k], 100 * this_cat / total))

def report_labels_regression(y, filename, units=None):
    # print out the total number of examples
    total = len(y)
    print('Total examples: {}'.format(total))

    # histogram plotting
    # ref: matplotlib documentation
    plt.hist(y, 10, facecolor='g', alpha=0.75)

    if units is not None:
        plt.xlabel(units)

    plt.ylabel('Count')
    plt.grid(True)

    # save histogram to a file
    plt.savefig(filename, bbox_inches='tight')

def report_model_classification(clf, X_train, X_test, y_train, y_test):
    train_accuracy = accuracy_score(y_train, clf.predict(X_train))
    print('Train error: {:0.1f}%'.format(1e2 * (1 - train_accuracy)))

    test_accuracy = accuracy_score(y_test, clf.predict(X_test))
    print('Test error: {:0.1f}%'.format(1e2 * (1 - test_accuracy)))

    n_train = X_train.shape[0]
    print('N_train: {}'.format(n_train))

    n_test = X_test.shape[0]
    print('N_test: {}'.format(n_test))

def report_model_regression(clf, X_train, X_test, y_train, y_test, units=None):
    units_str = ' ' + str(units)

    train_error = np.std(y_train-clf.predict(X_train))

    print('Train error: sigma={:0.3f}'.format(train_error) + units_str)

    test_error = np.std(y_test-clf.predict(X_test))
    print('Test error: sigma={:0.3f}'.format(test_error) + units_str)

    n_train = X_train.shape[0]
    print('N_train: {}'.format(n_train))

    n_test = X_test.shape[0]
    print('N_test: {}'.format(n_test))