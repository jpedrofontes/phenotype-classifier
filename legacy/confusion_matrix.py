import sys

import matplotlib.pyplot as plt
import numpy as np
from mlxtend.plotting import plot_confusion_matrix

# [TN, FP],
# [FN, TP]
# binary1 = np.array([[88, 14],
#                     [21, 205]])

binary1 = np.array([[96, 7],
                    [9, 216]])

fig, ax = plot_confusion_matrix(
    conf_mat=binary1, show_absolute=True, show_normed=True)
plt.savefig(
    '/home/mguevaral/jpedro/phenotype-classifier/img/' + sys.argv[1] + '.png')
