import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm
import sklearn.datasets

# This is our tools
import svm_tools
import plot_tools

# We get the dataset...
iris = sklearn.datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

# We learn the svm

gamma = svm_tools.gamma_of_sigma(.75) # sigma=.75
C     = 100
classifier = sklearn.svm.SVC(C=C,
                             kernel='rbf', gamma=gamma,
                             tol=1e-5,
                             decision_function_shape='ovo')
classifier.fit(X, y)

# Let us measure the performances

scores = sklearn.model_selection.cross_val_score(classifier, X, y, cv=5)
print('scores = ')
for s in scores:
    print('         {:.2%}'.format(1-s))
print('        ------')
print('  risk = {:.2%}'.format(1-np.average(scores)))
print()
print('Empirical risk = {}'.format(svm_tools.empirical_risk(classifier, X, y)))
print()


# Let us plot the decision

def color_of_label(label) :
    return {0 : 'red', 1 : 'green', 2 : 'blue'}[label]

xlim = [3, 9]
ylim = [1, 5]

fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(1,1,1)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_aspect('equal')
plot_tools.plot_samples(ax, X, y, color_of_label)
plot_tools.plot_svc_partition(ax, classifier, xlim, ylim,
                              [(1.0, .75, .75),
                               (.75, 1.0, .75),
                               (.75, .75, 1.0)])
plt.show()
