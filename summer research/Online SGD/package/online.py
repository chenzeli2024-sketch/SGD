# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 10:32:10 2021

@author: Tony Li
"""

import numpy as np
from tqdm import trange
from tqdm import tqdm

class onlinesgd:
    
    def __init__(self,n,r,x_0,x_1,beta):    #input the data information
        
        self.r=r                            # ratio r=d/n
        self.n=n                            # sample size
        self.d=int(n*r)                     # sample feature
        self.x_0=x_0                        # inital vector
        self.x_1=x_1                        # true vector
        self.beta=beta                      # batch size
    
    """
    This function conducts SGD iteration and is used in "batchsize" function 
    """
    
    def SGD(self,x,A,b,ss):                                     
        pos=np.random.choice(range(1000),self.beta,False)       # randomly select from 1 to 1000 to generate batch set
        Pk=np.zeros((self.n,self.n))
        Pk[pos,pos]=1                                           # matrix P_k
        x_new=x-ss*(1/self.n*np.dot(A.T.dot(Pk),(A.dot(x)-b)))  # SGD updates and ss means step size
        return x_new
    
    """
    This function is used in "batchone" function
    """
    
    def SGD_one(self,x,A,b,ss):                     
        x_new=x-ss/5*(np.dot(A.T,(A.dot(x)-b)))     # SGD updates, ss means step size
        return x_new
    
    """
     The function simulates the process of SGD model
     with regenerating data matrix A for each update
     """
    def batchsize(self,a):                                                 
                                                                            
        x1=self.x_0                                                         # initial vector      
        err=[]
        for i in trange(500):                                               # conduct 500 iterations
           A=np.random.normal(loc=0, scale=1.0,size=(self.n,self.d))        # regenerate data matrix A
           b=np.dot(A,self.x_1)                                             # calculate the vector b
           s=(2/(self.r*np.sum(np.diag(np.dot(A.T,A)/self.n))/self.d))/a    # calculate the step size
           x1=onlinesgd.SGD(self,x1,A,b,s)                                  # conduct the update
           err.append(np.sum((A.dot(x1)-b)**2)/(2*self.n))                  # record the value of f(x) for each update
        return err
    
    """
     input 5 samples for each updates and the number of 
     iteration is 500.
     """
    def batchone(self):                                             
                                                                    
       x1=self.x_0                                                  # initial vector   
       err=[]
       for i in trange(500):                                        # The amount of iteration is 500
          A=np.random.normal(loc=0, scale=1.0,size=(5,self.d))      # regenerate five sample with d features
          b=np.dot(A,self.x_1)                                      # calculate the vector b
          s=1e-3                                                    # step size as 1e-3
          x1=onlinesgd.SGD_one(self,x1,A,b,s)                       # conduct iterations
          err.append(np.sum((A.dot(x1)-b)**2)/(2*5))                # record the errors
       return err