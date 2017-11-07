import numpy as np
import matplotlib.pyplot as plt # for plotting.
import itertools as it          # for smart lazy iterations.

# This is scikit modules
import sklearn.datasets 
import sklearn.utils
import sklearn.svm

import svm_tools
import plot_tools

# Let us load the dataset.
data_size = 1000
digits    = sklearn.datasets.fetch_mldata('MNIST original', data_home='.')
i, o      = sklearn.utils.shuffle(digits.data, digits.target)
data_size = min(data_size,len(i))
inputs    = i[:data_size]
outputs   = o[:data_size]
images    = (data.reshape((28,28))/255.0 for data in inputs)
labels    = np.array([int(i) for i in outputs])


# Let us apply a linear C-SVM (see the documentation for parameters)
print()
print('learning with a {}-sized dataset...'.format(data_size))
C = None
classifier = sklearn.svm.NuSVC(nu=0.1,
                              kernel='linear',
                              tol=1e-5,
                              decision_function_shape='ovo')

classifier.fit(inputs, labels) # This is learning
print('done')


print()
print('##############')
print('Empirical risk')
print('##############')
print()
print('Empirical risk = {}'.format(svm_tools.empirical_risk(classifier, inputs, labels)))

# Cross-validation
print()
print('################')
print('Cross validation')
print('################')
print()

scores = sklearn.model_selection.cross_val_score(classifier, inputs, labels, cv=5)
print('scores = ')
for s in scores:
    print('         {:.2%}'.format(1-s))
print('        ------')
print('  risk = {:.2%}'.format(1-np.average(scores)))
print()
