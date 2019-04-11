import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.svm import SVC # "Support vector classifier"
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
# use seaborn plotting defaults
import seaborn as sns; sns.set()

from sklearn.datasets.samples_generator import make_blobs
#Dataset
X, labels = make_blobs(n_samples=250, centers=2,
                   random_state=1, cluster_std=3.25)
y = labels
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='cool');
#Cut Dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)    

#Validation Dataset
Xi, labels1 = make_blobs(n_samples=350, centers=2, random_state=1, cluster_std=3.25)
#print(X, labels)
#print(Xi, labels1)

# Different Kernel Functions
#model = SVC(kernel='poly', gamma='auto')
model = SVC(kernel='rbf',C=1,gamma=1)
#model = SVC(kernel='linear')
#model = SVC(kernel='sigmoid', gamma='auto') 
#model = SVR(kernel='rbf')

model.fit(X_train, Y_train)

print(model.score(X_test, Y_test))
#cross val over testing dataset
print(cross_val_score(model, X, y))
#cross val over validation dataset
print(cross_val_score(model, Xi, labels1, cv=5))
y_rbf = model.fit(X_train, Y_train).predict(X_test)
#plt.plot(X_test, y_rbf, label='RBF model')
#plt.show()
def plot_svc_decision_function(model, ax=None, plot_support=True):
     """Plot the decision function for a 2D SVC"""
     if ax is None:
         ax = plt.gca()
     xlim = ax.get_xlim()
     ylim = ax.get_ylim()
    
     # create grid to evaluate model
     x = np.linspace(xlim[0], xlim[1], 30)
     y = np.linspace(ylim[0], ylim[1], 30)
     Y, X = np.meshgrid(y, x)
     xy = np.vstack([X.ravel(), Y.ravel()]).T
     P = model.decision_function(xy).reshape(X.shape)
    
     # plot decision boundary and margins
     #ax.contour(X, Y, P, colors='k',
     ax.contour(X, Y, P, colors='k',
                levels=[-1, 0, 1], alpha=0.5,
                linestyles=['--', '-', '--'])
    
     # plot support vectors
     if plot_support:
         ax.scatter(model.support_vectors_[:, 0],
                    model.support_vectors_[:, 1],
                    s=300, linewidth=1, facecolors='none');
     ax.set_xlim(xlim)
     ax.set_ylim(ylim)


plot_svc_decision_function(model);
plt.show()
