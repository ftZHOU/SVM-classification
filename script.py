import sklearn.datasets

import svm_tools
import plot_tools

# Let us build up a dataset

# X, y = sklearn.datasets.make_moons(n_samples=100, noise=0.1)import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm

X, y = sklearn.datasets.make_blobs(n_samples=100, centers=[[-.5, 1], [1, -.5]], cluster_std=.25)

# Let us learn a linear C-SVC

C     = 100

classifier = sklearn.svm.SVC(C=C,
                             kernel='linear',
                             tol=1e-5,
                             decision_function_shape='ovo')
classifier.fit(X, y)
sep, desc = svm_tools.svc_ovo_separators(classifier, svm_tools.linear_kernel, X)

# Print descriptors

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

# Let us display it

fig = plt.figure(figsize=(12,10))

xlim = [-1.5, 2.5]
ylim = [-1.5, 1.5]

ax = fig.add_subplot(1,1,1)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_aspect('equal')
plot_tools.plot_svc_samples(ax, desc['0 vs 1'], X, y,
                            0, 1, 'white', 'gray',
                            C, 'black', 'red')
plot_tools.plot_svc_separation(ax, sep['0 vs 1'], xlim, ylim)
plot_tools.plot_svc_partition(ax, classifier, xlim, ylim, [(.98,.98,1.), (.98, 1., .98)])

plt.show()




# C = None
# classifier = sklearn.svm.NuSVC(nu=0.1,
#                              kernel='linear',
#                              tol=1e-5,
#                              decision_function_shape='ovo')
# classifier.fit(X, y)
# sep, desc = svm_tools.svc_ovo_separators(classifier, svm_tools.linear_kernel, X)

# gamma = svm_tools.gamma_of_sigma(.75) # sigma=.75
# C     = 100
# classifier = sklearn.svm.SVC(C=C,
#                              kernel='rbf', gamma=gamma,
#                              tol=1e-5,
#                              decision_function_shape='ovo')
# classifier.fit(X, y)
# sep, desc = svm_tools.svc_ovo_separators(classifier, svm_tools.gaussian_kernel(gamma), X)

#performance cross-validation
#n'oublie pas de mettre C trop petit pour voir le risk qui n'est pas null
# scores = sklearn.model_selection.cross_val_score(classifier, X, y, cv=5)
# print('scores = ')
# for s in scores:
#     print('         {:.2%}'.format(1-s))
# print('        ------')
# print('  risk = {:.2%}'.format(1-np.average(scores)))
# print()
# print('Empirical risk = {}'.format(svm_tools.empirical_risk(classifier, X, y)))
# print()

