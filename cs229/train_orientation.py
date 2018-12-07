import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

from cs229.load_data_orientation import CATEGORIES
import joblib

def plot_pca(clf, X, y):
    scaler = clf.named_steps['standardscaler']
    pca_std = clf.named_steps['pca']

    X_t = pca_std.transform(scaler.transform(X))

    for l, c, m in zip(range(2), ('blue', 'red'), ('o', 'x')):
        plt.scatter(X_t[y == l, 0], X_t[y == l, 1], color=c, label='class %s' % l, alpha=0.5, marker=m)

    plt.legend(loc='upper right')

    plt.grid()
    plt.show()

def train(X, y, plot=True, dump=True, report=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = make_pipeline(StandardScaler(), PCA(n_components=3), LogisticRegression())

    clf.fit(X_train, y_train)

    if plot:
        plot_pca(clf, X_train, y_train)

    if report:
        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=CATEGORIES))

    if dump:
        joblib.dump(clf, 'clf_orient.joblib')

def main():
    X = joblib.load('X_orient.joblib')
    y = joblib.load('y_orient.joblib')

    print('Training orientation detector for male fly...')
    train(X['male'], y['male'])

    print('Training orientation detector for female fly...')
    train(X['female'], y['female'])

if __name__ == '__main__':
    main()