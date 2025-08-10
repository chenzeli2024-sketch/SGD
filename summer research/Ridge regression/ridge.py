# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 22:48:03 2021

@author: Tony Li
"""
import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

"""
This class is related to SGD application on the redge regression.
"""


class ridge:
    
    """
    used to input all the data needed to conduct the simulation, including
    
    sample size---n;
    sample feature---d;
    d/n---r;
    initial vector---x_0
    true vector for the target function---x_1;
    data matrix---A
    batch size--beta
    step size---ss
    penalty---lambda_
    """
    def __init__(self,A,x_0,x_1,beta,r,n,d,ss):
        self.A=A
        self.x_0=x_0
        self.x_1=x_1
        self.beta=beta
        self.r=r
        self.n=n
        self.d=d
        self.ss=ss
        lambda_=list(np.around(np.linspace(0,0.1,100,endpoint=False)
                               ,decimals=6))
        temp=list(np.around(np.linspace(0.1,3,291),decimals=5))
        self.lambda_=lambda_+temp
        self.b=A.dot(x_1)
        
    
    """
    There is no difference between SGDs function and SGDc function except 
    the input variables are different.
    """
    
    def SGDs(self,x,lambda_):
        pos=np.random.choice(range(1000),self.beta,False)                                        # select the batch set from 1 to 1000
        Pk=np.zeros((self.n,self.n))       
        Pk[pos,pos]=1
        x_new=x-self.ss*1/self.n*(np.dot(self.A.T.dot(Pk),(self.A.dot(x)-self.b))+lambda_*x)   # conduct the SGD iteration
    
        return x_new 
    
    def SGDc(self,x,lambda_,ss):
        pos=np.random.choice(range(1000),self.beta,False)                                        # select the batch set from 1 to 1000
        Pk=np.zeros((self.n,self.n))
        Pk[pos,pos]=1
        x_new=x-ss*(1/self.n*np.dot(self.A.T.dot(Pk),(self.A.dot(x)-self.b))+lambda_*x)        # conduct the SGD iteration
    
        return x_new 
    
    """
    The simulation of iteration with update stops when k=100
    """
    def iterate_se(self): 
        err_whole=[]                                               # store error terms for all the lambdas
        collapse=[]                                                
        for la in tqdm(self.lambda_):                              # iterate on all the lambda
            x1=self.x_0                                            # initialize the start vector of SGD model
            err=[]                                                 # record f(x)
            ite=1                                                  # record the number of iterations
            slope=-1                                               # initialize the variable slope;slope is to detect if the iteration will diverge to infinity for some lambda
            while ite<=100 and slope<0:                            # iterate 100 times on SGD model for each lambda and ensure iterations converge
                x1=ridge.SGDs(self,x1,la)
                err.append(np.sum((self.A.dot(x1)-self.b)**2)/
                           (2*self.n)+la*np.sum(x1**2))
                ite=ite+1                                          
                if len(err)==10:                                   # Show the definition of slope. If slope>=0, then SGD model may diverge.
                    slope=err[9]-err[0]
            if slope>=0:                                           # if iteration diverges for some lambda
                collapse.append(la)                                # collapse records lambda which causes diverge
                err_whole.append([])                               # if this lambda diverges, error terms will be null.
            else:    
                err_whole.append(err)
        err_3=[]
        for k in err_whole:                                        # Record the f(x) where iteration stops and drop lambda that diverges
            if len(k):                                             # Exclude the lambda which causes diverge.
                err_3.append(k[-1])
            else:
                err_3.append(-1)
        return err_3,collapse
    
    """
    The simulation of iteration with update stops when k 
    satisfies the criteria mentioned in the report
    """
    def iterate_dls(self):
        err_whole=[]                                                # store error terms for all the lambdas
        collapse=[]
        for la in tqdm(self.lambda_):                               # iterate on all the lambda(penalty)
            x1=self.x_0 
            err=[]                                                  # store error terms for each lambda
            temp=1                                                  # initialize the variable temp; temp is used to decide when iteration stops
            slope=-1                                                # initialize the variable slope;slope is to detect if the iteration will diverge to infinity for some lambda
            k=0                                                     # initialize the variable k
            while k<=5 and slope<0:                                 # if slope<=0 then the iteration will not diverge.
                x1=ridge.SGDs(self,x1,la)
                err.append(np.sum((self.A.dot(x1)-self.b)**2)/
                               (2*self.n)+la*np.sum(x1**2))
                if len(err)==50:                                    # slope is defined as f(x_49)-f(x_0)
                    slope=err[49]-err[0]
                if len(err)>20:
                    temp=abs((err[-1]-err[-5])/4)                   # the definition of temp
                if temp<1e-3:                                       # if the number of the event that temp<1e-3 happens for 5 times, then iteration stops
                    k+=1                                            # k records the times
            if slope>=0:                                            # if iterations diverge
                collapse.append(la)                                 # collapse records lambda which causes diverge
                err_whole.append([])                                # if this lambda diverges, error terms will be null.
            else:    
                err_whole.append(err)
        err_3=[]
        for k in err_whole:                                         # Record the f(x) where iteration stops and drop lambda that diverges
            if len(k):
                err_3.append(k[-1])
            else:
                err_3.append(-1)
        return err_3,collapse
    
    """
    This function is used to draw the plot for convergence trace of 
    SGD model with lambda coresponding to smallest f(x). And the code is the same as 
    those in the iterate_dls and iterate_se function. "model" variable is used to 
    determine which stop criteria is used.
    """
    def iterate(self,lambda_,ss,model):
        la=lambda_
        if model=="different":                                            # the SGD iteration will stop with f(x) small enough
            slope=-1
            k=0
            temp=1
            x1=self.x_0
            err=[]
            while k<=5 and slope<0:                                       # k records the number of situation in which temp is smaller than 1e-3
                x1=ridge.SGDc(self,x1,la,ss)
                err.append(np.sum((self.A.dot(x1)-self.b)**2)
                           /(2*self.n)+la*np.sum(x1**2))
                if len(err)==50:
                    slope=err[49]-err[0]                                  # this is used to ensure SGD iteration converge: when slope>0 iteration diverges lambda will be dropped later
                if len(err)>20:
                    temp=abs((err[-1]-err[-5])/4)                         # if the number of the situation in which temp<1e-3
                if temp<1e-3:
                    k+=1

                
        if model=="same":                                                 # the SGD iteration will stop when the number of iterations is 100
            x1=self.x_0
            err=[]
            iteration=1
            slope=-1
            while iteration<=100 and slope<0:                             # the iteration stops when iteration<=100 and slope<0 (iteration converges)
                x1=ridge.SGDc(self,x1,la,ss)
                err.append(np.sum((self.A.dot(x1)-self.b)**2)/(2*self.n)+la*np.sum(x1**2))
                iteration=iteration+1
                if len(err)==10:
                    slope=err[9]-err[0]
                    
        return err
    """ 
    This function is used to plot the convergence trace of SGD model with lambda corresponding the smallest f(x)
    """
    def plot(self,lambda_,sg,model):
        k=["1.2","2","4","8"]
        fig, ax = plt.subplots(2, 4, sharex='col', sharey='row',figsize=(10,5))
        for i in range(2):
            for j in range(4):
                err=ridge.iterate(self,lambda_[i*4+j],sg[j],model[i])     # store the values of f(x) for each lambda
                ax[i,j].set_title(r'$\gamma=\gamma_{max}/$'+k[j])         # set the title
                ax[i, j].plot(range(1,len(err)+1),err)                    # plot the 
                if i==1:
                    ax[i,j].set_xlabel("iteration")                       # set the x label
                if j==0:
                    ax[i,j].set_ylabel(r"$f(x)$")                         # set the y label
        fig.savefig(".\\ridge.pdf",dpi=600,format="pdf")                  # save the plot as pdf

        
            