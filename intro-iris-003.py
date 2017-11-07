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
sep, desc = svm_tools.svc_ovo_separators(classifier, svm_tools.gaussian_kernel(gamma), X)

# Let us plot the classifiers

def color_of_label(label) :
    return {0 : 'red', 1 : 'green', 2 : 'blue'}[label]

xlim = [3, 9]
ylim = [1, 5]

fig = plt.figure(figsize=(20,5))

nb_classifiers = len(sep)
plot_id = 0
for name, h in sep.items() :
    plot_id += 1
    ax = fig.add_subplot(1, nb_classifiers, plot_id)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_title(name)
    cpos, cneg = svm_tools.ovo_classes(name)
    plot_tools.plot_svc_samples(ax, desc[name], X, y,
                                cpos, cneg,
                                color_of_label(cpos), color_of_label(cneg),
                                None, None, 'black')
    plot_tools.plot_svc_separation(ax, h, xlim, ylim)
    



plt.show()
