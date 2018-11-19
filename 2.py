import FeatureSelection as fs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation

model = AffinityPropagation()
model.fit(fs.train_data)
labels = model.labels_

clusters = [[] for i in range(max(labels)+1)]

for i in range(len(labels)):
    clusters[labels[i]].append([fs.train_data[i],fs.train_labels[i]])


cluster_labels = np.zeros(len(clusters))

for i in range(len(clusters)):
    n = np.zeros(len(clusters))
    cluster = clusters[i]
    for sample in cluster:
        n[sample[1]] += 1
    cluster_labels[i] = np.argmax(n)


new_labels = [-1] * len(fs.train_data)

for i in range(len(fs.train_labels)):
    new_labels[i] = cluster_labels[labels[i]]


def ccr(labels, clustering_labels):
    return (fs.train_labels==new_labels).mean()   

print ccr(new_labels, fs.train_labels)

#########CCR  =   0.8988

def conf_matrix(labels, clustering_labels):
    confusion = np.zeros([10, 10])
    for i in range(len(labels)):
        l = int(labels[i])
        confusion[l][clustering_labels[i]] += 1
    return confusion


conf = conf_matrix(new_labels, fs.train_labels)

plt.imshow(conf)
plt.colorbar()
plt.show()


cluster_means = []
for cluster in clusters:
    cluster_means.append(np.array(cluster).mean())


distances = np.zeros([10, 10])
for i in range(10):
    for j in range(10):
        distances[i][j] = np.linalg.norm(cluster_means[i] - cluster_means[j])

print distances

np.save('affinity',clusters)