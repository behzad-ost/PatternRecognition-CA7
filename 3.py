import numpy as np
import matplotlib.pyplot as plt

def SW(types, nclass, nfeature):
    sw = np.zeros([nfeature,nfeature])
    for i in range(nclass):
        mu = np.mean(types[i],axis=0)[0]
        for s in types[i]:
            x = s[0]
            sw += np.dot((x-mu).reshape(nfeature,1),(x-mu).reshape(1,nfeature))
    return sw



def SB(mu, types, nclass, nfeature):
    sb=np.zeros([nfeature,nfeature])
    for i in range(nclass):
        m = np.mean(types[i],axis=0)[0]
        sb += np.dot((m-mu).reshape(nfeature,1),(m-mu).reshape(1,nfeature))*len(types[i])
    return sb



aff = np.load('affinity.npy')
agg = np.load('agglomerative.npy')


all=[]
for cluster in aff:
    for x in cluster:
        all.append(x[0])
mu = np.mean(all,axis=0)


def plot(matrix):
    plt.imshow(matrix)
    plt.colorbar()
    plt.show()


sw_agg = SW(agg,len(agg),62)
sb_agg = SB(mu,agg,len(agg),62)
sw_agg_inv = np.linalg.inv(sw_agg)
scatter_agg = np.dot(sw_agg_inv,sb_agg)

plot(sw_agg)
plot(sb_agg)
plot(scatter_agg)

sw_aff = SW(aff,len(aff),62)
sb_aff = SB(mu,aff,len(aff),62)
sw_aff_inv = np.linalg.inv(sw_aff)
scatter_aff = np.dot(sw_aff_inv,sb_aff)

plot(sw_aff)
plot(sb_aff)
plot(scatter_aff)

trace_scatt_agg = np.trace(scatter_agg)
trace_scatt_aff = np.trace(scatter_aff)
plt.bar([1,2],[trace_scatt_agg,trace_scatt_aff])
plt.show()