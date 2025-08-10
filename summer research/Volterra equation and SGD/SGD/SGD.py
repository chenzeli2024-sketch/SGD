import numpy as np
from tqdm import trange
"""
This class iterates SGD model of isotropic model
"""

class SGDgenerate:
    """
    Import all the data:
        
        sample size---n;
        sample feature---d;
        d/n---r;
        initial vector---x_0
        true vector for the target function---x_1;
        data matrix---A
        batch size--beta
        step size---ss
        response vector---b
    """
    def __init__(self,A,n,r,x_0,x_1,ss,beta):
        self.A=A
        self.r=r
        self.n=n
        self.d=int(n*r)
        self.x_0=x_0
        self.x_1=x_1
        self.ss=ss
        self.beta=beta
        self.b=A.dot(x_1)

    
    """
    SGD function for each update and the function is very similar 
    with the function in other files.
    """
    
    def SGD(self,x):
        pos=np.random.choice(range(1000),self.beta,False)                         # randomly selecting from 1 to 1000
        Pk=np.zeros((self.n,self.n))
        Pk[pos,pos]=1
        x_new=x-self.ss/self.n*np.dot(self.A.T.dot(Pk),(self.A.dot(x)-self.b))    # ss is the step size
        
        return x_new
    """
    Iteration with SGD model with x_0 as initial vector
    """
    def iterate(self):
        x1=self.x_0                                                        # initial vector
        epoch=40                                                           # 40 epochs
        err=[]
        for i in trange(epoch):
            for j in range(int(self.n/self.beta)):                         # Each epoch contains n/beta iterations
                x1=SGDgenerate.SGD(self,x1)
            err.append(np.sum((self.A.dot(x1)-self.b)**2)/(2*self.n))      # record the values of f(x)
        return err    