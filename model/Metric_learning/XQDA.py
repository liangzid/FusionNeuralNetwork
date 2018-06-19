# this file is learning to write a XQDA function
# transform it from MATLAB
# liangzi  2017 3.29


"""
modified the code..
some bug was found when I use this pyfile into person Re-ID
this code has no test before

liangzid, 2018,6,10
"""


import numpy
import math

def XQDA(galx, probx, gal_labels, prob_labels, learn_rate=0.001,qda_dims = -1):

    num_gals= galx.shape[0]
    d=galx.shape[1]
    num_probs=probx.shape[1]

    if d>num_gals+num_probs :
        W,X =numpy.linalg.qr(numpy.hstack((galx.T,probx.T)))
        galx=X[:,0:num_gals-1]
        probx=X[:,num_gals:-1]
        d=X.shape[0]

    labels=numpy.unique(numpy.vstack((gal_labels,prob_labels)))
    c=len(labels)

    galW=numpy.zeros((num_gals,1))
    gal_class_sum=numpy.zeros((c,d))
    probW=numpy.zeros((num_probs,1))
    prob_class_sum=numpy.zeros((c,d))
    ni=0
    for k in range(c):
        gal_index = numpy.where(gal_labels==labels(k))
        nk=len(gal_index)
        gal_class_sum[k-1,:]=numpy.sum(galx[gal_index-1,:],1)

        prob_index = numpy.where(prob_labels == labels(k))
        mk = len(prob_index)
        prob_class_sum[k - 1, :] = numpy.sum(probx[prob_index - 1, :], 1)

        ni += nk*mk
        galW[gal_index-1]=math.sqrt(mk)
        probW[prob_index-1]=math.sqrt(nk)

    gal_sum=numpy.sum(gal_class_sum,1)
    prob_sum=numpy.sum(prob_class_sum,1)
    gal_cov=galx.T.dot(galx)
    prob_cov=probx.T.dot(probx)

    galx=numpy.einsum('ij,jk',galW,galx)  # maybe wrong
    probx=numpy.einsum('i,i',probW,probx)

    in_cov=galx.T.dot(galx)+probx.T.dot(probx)-gal_class_sum.T.dot(prob_class_sum)-prob_class_sum.T.dot(gal_class_sum)
    ex_cov=num_probs.dot(gal_cov)+num_gals.dot(prob_cov)-gal_sum.T.dot(prob_sum)-prob_sum.T.dot(gal_sum)-in_cov

    ne=num_gals.dot(num_probs)-ni
    in_cov /=ni
    ex_cov /=ne

    in_cov += learn_rate*numpy.eye(d)
    V,S,u = numpy.linalg.svd(numpy.linalg.inv(in_cov).dot(ex_cov))
    latent=numpy.diag(S)

    # maybe??
    index=numpy.argsort(-latent)
    latent=numpy.sort(-latent)

    energy=numpy.sum(latent)
    minv=latent[-1]
    r=numpy.sum(latent>1)
    energy=numpy.sum(latent[0:r-1])/energy

    if qda_dims>r :
        qda_dims=max(1,r)

    V=V[:,index[0:qda_dims-1]]



    in_cov=V.T.dot(in_cov).V
    ex_cov=V.T.dot(ex_cov).V
    M=numpy.linalg.inv(in_cov)-numpy.linalg.inv(ex_cov)

    return M