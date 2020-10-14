# -*- coding: utf-8 -*-
import numpy as np
from scipy import stats
from scipy.stats import multivariate_normal
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import functools


class uncert_prop:
    """ 
    Compute uncertainty of scalar function func(x). 
    
    Attributes:
    --------
    func : callable
        - Should take a vector or real numbers x
        
    x : array
        - Array of variables

    cov_matrix : array or None, optional
        - The covariance matrix of the variables x. The default is `None` and equal to numpy.cov(x)
        
    method : str, optional
        - The desired method. There are 2 valid methods, `Delta` and `Monte_Carlo`. By default, method='Delta' 
        
    MC_sample_size : int, optional
        - The size of Monte Carlo sample. By default, MC_sample_size = 10000
        
    grad_dx : float, optional
        - The size of step to numerically compute gradient of func(x). By default, grad_dx = 1e-8
        
    Methods:
    --------
    x_MC_samples() : array
        - Array of sampled variables
        
    x_MC_dist_plot(contours = 15,cmap='jet',x_label=None, y_label=None, save_name=None) : array
        - 3D and 2D plots of x_MC_samples distribution  

    f_MC() : array
        - Array of func(x_MC_samples)
        
    f_MC_dist_plot(func_name='f',save_name=None) : str, optional
        - Plot of f_MC distribution 
        
    SEM() : float
        - the Standard Error of Mean
        
    confband(self,sample_size=None,conf=0.95) : tuple
        - Upper and lower confident bands of func(x). If sample_size = None then critical value is taken from normal distribution. Else, t-Student distribution is used.
   
    """ 
    
    def __init__(self,func,x,cov_matrix=None,method='Delta',MC_sample_size=10000,grad_dx=1e-8):
        self.func = func
        self.x = x
        try:
            if cov_matrix is list or (type(cov_matrix)==np.ndarray):
                self.cov_matrix =cov_matrix
            else: 
                self.cov_matrix = np.cov(self.x)
        except :
            print('Invalid list type')
            sys.exit()
        self.grad_dx = grad_dx
        self.MC_sample_size = int(np.floor(MC_sample_size))
        self.method = method
        if not ((method=='Delta') or method=='Monte_Carlo'):
            print('Incerted method is not valid.\nValid methods are: \n-- Delta-- \n--Monte_Carlo--.')
            sys.exit()
                            
    def __gradient(self):
        grad = np.zeros(len(self.x))
        for j in range(len(self.x)):  
            Dxj = (abs(self.x[j])*self.grad_dx if self.x[j] != 0 else self.grad_dx)
            x_plus = [(xi if k != j else xi + Dxj) for k, xi in enumerate(self.x)]
            grad[j] = (self.func(x_plus) - self.func(self.x))/Dxj
        return grad
            
    @functools.lru_cache(maxsize=128)
    def x_MC_samples(self):
        if self.method=='Monte_Carlo':
            return multivariate_normal.rvs(self.x,self.cov_matrix,self.MC_sample_size)
        else:
            return print('x_MC_samples defined only for Monte_Carlo method')
            
    # Plot distribution from which popt_MC is sampled
    def x_MC_dist_plot(self,contours = 15,cmap='jet',xlabel=None, ylabel=None, save_name=None):
        if self.method=='Monte_Carlo':
            if len(self.x)==2:
                # Create grid and multivariate normal
                x = np.linspace(min(self.x_MC_samples().T[0]),max(self.x_MC_samples().T[0]),500)
                y = np.linspace(min(self.x_MC_samples().T[1]),max(self.x_MC_samples().T[1]),500)
                X, Y = np.meshgrid(x,y)
                pos = np.empty(X.shape + (2,))
                pos[:, :, 0] = X; 
                pos[:, :, 1] = Y
                rv = multivariate_normal(self.x, self.cov_matrix)
                fig = plt.figure()
                ax1 = fig.add_subplot(1,2,1, projection='3d')
                ax1.plot_surface(X, Y, rv.pdf(pos),cmap=cmap,linewidth=0)
                ax2 = fig.add_subplot(1,2,2)
                ax2.contour(X,Y,rv.pdf(pos),contours,cmap=cmap)
                if xlabel != None:
                    ax2.set_xlabel(str(xlabel))
                else:
                    ax2.set_xlabel('$x_1$')
                if ylabel != None:
                    ax2.set_ylabel(str(ylabel))
                else:
                    ax2.set_ylabel('$x_2$')
                ax2.grid()
                if xlabel != None:
                    ax1.set_xlabel(str(xlabel))
                else:
                    ax1.set_xlabel('$x_1$')
                if ylabel != None:
                    ax1.set_ylabel(str(ylabel))
                else:
                    ax1.set_ylabel('$x_2$')
                ax1.set_zlabel('$Gaussian\ PDF$')
                ax1.view_init(elev=30, azim=-70)
                fig.set_size_inches((11.75,8.25), forward=False)
                if save_name != None:
                    fig.savefig(str(save_name)+'.png', dpi=300,bbox_inches='tight')
                del fig
            else:
                print('x_MC_dist_plot is only defined for x with 2 variables')
        else:
            print('x_MC_dist_plot is only defined for Monte_Carlo method')
            
    @functools.lru_cache(maxsize=128)
    def f_MC(self):
        if self.method=='Monte_Carlo':
            return [self.func(self.x_MC_samples()[i]) for i in range(self.MC_sample_size)]
        else:
            return print('f_MC defined only for Monte_Carlo method')
            
    # Plot distribution of f_MC
    def f_MC_dist_plot(self,func_name='f',save_name=None):
        if self.method=='Monte_Carlo':
            fig,ax=plt.subplots(1)
            sns.distplot(self.f_MC(),kde=True,kde_kws={"color": "b", "lw": 1.5, "label": "Kernel Density Estimation"})
            f_MC_lnsp = np.linspace(min(self.f_MC()),max(self.f_MC()),200)  
            plt.plot(f_MC_lnsp, stats.norm.pdf(f_MC_lnsp,loc=np.array(self.f_MC()).mean(),scale=np.std(self.f_MC())),'r--',label="Gaussian distribution \nwith same mean \nand standard deviation \nas Monte Carlo sample") 
            plt.title('Distribution of '+str(func_name)+ ' after '+str(self.MC_sample_size)+' Monte Carlo Simulations')
            plt.xlabel(str(func_name))
            plt.ylabel('Probability Density')
            plt.legend()
            plt.grid() 
            if save_name != None:
                fig.set_size_inches((8.25,5.8), forward=False)
                fig.savefig(str(save_name)+'.png', dpi=300,bbox_inches='tight')                       
        else:
            print('f_MC_dist_plot is only defined for Monte_Carlo method')  
        
    def SEM(self):
        if self.method =='Delta':
            return np.sqrt(self.__gradient().dot(self.cov_matrix).dot(self.__gradient().T))
        elif self.method == 'Monte_Carlo':
            return np.std(self.f_MC()) 
        else:
            print('Method is invalid')
            sys.exit()
        
    def confband(self,sample_size=None,conf=0.95):
        alpha = 1.0 - conf    # significance
        var_n = len(self.x)  # number of parameters
        if not type(sample_size)==int or type(sample_size)==float:
            # Quantile of Normal distribution for p=(1-alpha/2)
            q = stats.norm.ppf(1.0 - alpha / 2.0)
        else:    
            # Quantile of Student's t distribution for p=(1-alpha/2)
            q = stats.t.ppf(1.0 - alpha / 2.0, sample_size - var_n)
        # Predicted values 
        yp = self.func(self.x)
        # Prediction band
        dy = q * self.SEM()
        # Upper & lower prediction bands.
        lcb, ucb = yp - dy, yp + dy
        return (lcb, ucb)
    
