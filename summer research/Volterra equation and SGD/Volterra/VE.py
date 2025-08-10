# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm
from scipy.integrate import quad
from scipy.linalg import toeplitz
from tqdm import trange

"""
This class uses numerical method (Costarelli and Spigler(Summer 2013)) 
mentioned in the report to calculate the Volterra equation.
"""
class Volterra:
    
    
    def __init__(self,A,x_0,x_1,n,r,ss,beta):
        self.n=n
        self.r=r
        self.d=int(n*r)
        self.beta=beta
        self.A=A
        self.x_0=x_0
        self.x_1=x_1
        self.b=A.dot(x_1)
        self.ss=ss                                     # step size 
        self.R=np.linalg.norm(x_0-x_1)**2              # R is the l2 norm difference between x_0 and x_1
        self.l1=(1-np.sqrt(r))**2                      # The parameters l1,l2 are used in Marchenko-Pastur law 
        self.l2=(1+np.sqrt(r))**2
        self.a=1                                       # a,b1,N are used in the numerical method
        self.b1=40
        self.N=1000
        self.h=(self.b1-self.a)/self.N                 # h is the length of interval
    
    """
    Function h1,h2,hh1,hh2,ink are prepared to calculate matrix M_N and F_N 
    """
    
    def h1(self,x,t):
        return np.exp(-2*self.ss*t*x)*np.sqrt((x-self.l1)*(self.l2-x))
    def h2(self,x,t):
        return x*np.exp(-2*self.ss*t*x)*np.sqrt((x-self.l1)*(self.l2-x))
    def hh1(self,t):
        temp=lambda x:Volterra.h1(self,x,t)
        val,pr=quad(temp,self.l1,self.l2)
        return val
    def hh2(self,t):
        temp=lambda x:Volterra.h2(self,x,t)
        val,pr=quad(temp,self.l1,self.l2)
        return val
    def ink(self,t):
        temp=lambda x:Volterra.hh2(self,t-x)
        val,pr=quad(temp,0,t)
        return val*self.ss**2/(2*np.pi)
    
    """
    matrix function is used to solve matrix M_N
    """
    
    def matrix(self):
        a=self.a
        b1=self.b1
        N=self.N
        h=(b1-a)/N
        M=[]
        inter=[]
        for i in trange(-1,N+1):
            t=a+h*i
            M.append(1-Volterra.ink(self,t))
            inter.append(t)
            
        M=M[2:]
        M_new=[1]+M
        MN=np.tril(toeplitz(M_new),k=0)                     # matrix M_N
        
        return MN,inter                                     # MN is the matrix M_N and inter stores the 1000 equal points
    
    """
    Solve equation M_NY_N=F_N
    """
    
    def aprox(self,t,MN,inter):  
        a=self.a
        N=self.N
        h=self.h
        temp=[]
        [temp.append(self.R/(4*np.pi*self.r)*Volterra.hh1(self,a+h*i)) for i in range(N+1)]
        FN=np.mat(temp).T                                       # Transpose the matrix
    
        yn=np.linalg.inv(MN).dot(FN)                            # Solve M_NY_N=F_N
        top=inter[2:]
        num=[]
        [num.append(i+1) for i in range(len(top)) if t>=top[i]]
        res=np.sum(yn[num])+float(yn[0])                        # Calculate the function used to
                                                                # approx the Volterra equation
        
        return res
    """
    Calculate the values of the function when its independent variable takes 1,2,...,40
    """
    def point(self,MN,inter):
        x=np.linspace(1,40,40)
        y=[]
        for k in tqdm(x):
            y.append(Volterra.aprox(self,k,MN,inter))
        return y