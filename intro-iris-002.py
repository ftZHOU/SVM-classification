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

# Let inspect the internal classifiers

sep, desc = svm_tools.svc_ovo_separators(classifier, svm_tools.gaussian_kernel(gamma), X)

# The separators are actual functions (labalizers).
print()
print("##########")
print("Separators")
print("##########")
print()
for key, val in sep.items() :
    print('{} = {}'.format(key,val))

# The descriptors are the building blocks of the separators (coefficients, support vectors, etc...)
print()
print("######################")
print("Separator descriptions")
print("######################")
print()
for key, val in desc.items() :
    print('{} : h(x) = {}'.format(key, val[0]))
    for alpha, xi in val[1] :
        print('              + {} * K({}, x)'.format(alpha,xi))
    print()
