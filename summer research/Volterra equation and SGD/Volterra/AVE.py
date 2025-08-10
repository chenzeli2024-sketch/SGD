# -*- coding: utf-8 -*-
import numpy as np
from scipy.integrate import quad

"""
This class uses analytic solution of the Volterra equation mentioned in 
Courtney Paquette, Lee,Pedregosa and Elliot Paquette (2021)
"""
class Analytic:
    """
    Import all the data used later
    
    R--- l2 norm of x_0-x_1
    q,w--- used for analytic expression 
    d--- dimension of features
    
    other variables are the same as those introduced in other files
    """
    
    def __init__(self,A,x_1,n,r,ss,x_0):
        self.A=A
        self.x_1=x_1
        self.n=n
        self.r=r
        self.d=int(n*r)
        self.x_0=x_0
        self.R=np.sum((x_0-x_1)**2)
        self.ss=ss
        self.q=(1+self.r)/2*(1-self.r*self.ss/2)
        self.w=1/4*(1-self.r*self.ss/2)**2*(8/self.ss-(1+self.r)**2)
    
    """
    The common part of analytic solution when step size takes different values.
    """
    def basefun(self,t):
        l1=(1-np.sqrt(self.r))**2
        l2=(1+np.sqrt(self.r))**2
        temp=lambda x:np.exp(-2*self.ss*x*t)/((x-self.q)**2+self.w)*np.sqrt((x-l1)*(l2-x))
        res,pre=quad(temp,l1,l2,epsrel=1e-16,limit=1000)
        return self.R/self.ss*(1-self.r*self.ss/2)*res/(2*np.pi*self.r)
    
    """
    From the paper (Courtney Paquette, Lee,Pedregosa and Elliot Paquette (2021))
    we know the analytic solution is a piecewise function, the solution has 
    different expressions when step size is larger or smaller than a threshold
    """
    def anaso(self,t):
        stop=2/(np.sqrt(self.r)*(self.r-np.sqrt(self.r)+1))
        if self.ss<=stop:                                     # when step size is small
            l1=(1-np.sqrt(self.r))**2
            l2=(1+np.sqrt(self.r))**2
            temp=lambda x:np.exp(-2*self.ss*x*t)/((x-self.q)**2+self.w)*np.sqrt((x-l1)*(l2-x))
            res,pre=quad(temp,l1,l2,epsrel=1e-16,limit=1000)
            return self.R/self.ss*(1-self.r*self.ss/2)*res/(2*np.pi*self.r)
        else:                                                 # when step size is large
            res=self.R/(4*np.sqrt(np.abs(self.w)))*(self.q+np.sqrt(np.abs(self.w))-
                                                    ((2/self.ss)**2)*((1-self.r*self.ss/2)**2)
                                                    *(self.q-np.sqrt(np.abs(self.w)))
                                                    /(self.r*(self.q**2-
                                                              np.abs(self.w))))*np.exp(-2*self.ss*(self.q+np.sqrt(np.abs(self.w)))*t)
            return Analytic.basefun(self,t)+res
        
        """
        This function stores the values of the solution when independent 
        variable takes 1,2,...,40, corresponding to the value of f(x) 
        under SGD model of epoch 1,2,...40.
        """
    def iterate(self):
        t=np.around(np.linspace(1,40,40),decimals=6)
        y=[]
        for k in t:
            y.append(Analytic.anaso(self,k))                  # y contains the approximation of the solution
        return y