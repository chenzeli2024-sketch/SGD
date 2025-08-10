# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 16:49:23 2021

@author: Tony Li
"""
import numpy as np

"""
This class is used to generate data for isotropic model which includes

the number of sample size---n;
the number of sample feature---d;
ratio d/n---r;
initial vector---x_0
true vector for the target function---x_1;
data matrix---A
batch size--beta

And we will use these variables as our inputs for numerical simulations.
"""

class isomodel:
    def generate(self):                                                         
        r=1.2
        n=1000
        d=int(r*n)
        np.random.seed(0)                                                     # set random seed to ensure the same result
        x_0=np.random.multivariate_normal(mean=[0]*int(d),cov=np.eye(d,d)/d)  # generate x_0
        x_1=np.random.multivariate_normal(mean=[0]*int(d),cov=np.eye(d,d)/d)  # generate x_1
        A=np.random.normal(loc=0, scale=1.0,size=(n,d))                       # generate data matrix
        beta=3                                                                # batch size is set as 3
        
        return A,x_0,x_1,beta,r,n,d