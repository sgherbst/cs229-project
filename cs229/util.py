import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from prettytable import PrettyTable

def report_model(y_test, y_pred, categories):
    print(classification_report(y_test, y_pred, digits=3, target_names=categories))

def train_experiment(train, categories, trials=100):
    results = {}
    metrics = ['precision', 'recall', 'f1', 'support']
    for category in categories:
        results[category] = {metric: [] for metric in metrics}

    for _ in range(trials):
        _, y_test, y_pred = train()
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
        for k, category in enumerate(categories):
            results[category]['precision'].append(precision[k])
            results[category]['recall'].append(recall[k])
            results[category]['f1'].append(f1[k])
            results[category]['support'].append(support[k])

    table = PrettyTable()
    table.field_names = ['category'] + metrics
    for category in sorted(categories):
        row = [category]
        for metric in metrics:
            ave = np.mean(results[category][metric])
            std = np.std(results[category][metric])
            row.append('{:0.3f} +/- {:0.3f}'.format(ave, 2*std))
        table.add_row(row)

    print(table.get_string(title='Model Performance ({} trials)'.format(trials)))