#from statistics import linear_regression
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

df = pd.read_csv("C:/Users/mannn/Downloads/housing.csv")

X = df.values[:,:-1]
y = df.values[:, -1]



class LinearRegression:
    
    def __init__(self, X,y,learning_rate,epsilon, max_iteration, gd = True, regularization = None, lambda_ = 0.01) -> None:
        self.X = X
        self.y = y
        self.max_iteration = max_iteration
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gd = gd
        self.regularization = regularization
        self.lambda_ = lambda_
        
    def split_data(self):
        X_train,X_test,y_train,y_test = train_test_split(self.X,self.y,test_size=0.3,shuffle = True)
        return X_train, X_test,y_train,y_test
     
    # basis expansion:
    def add_x0(self,X):
        return np.column_stack([np.ones([X.shape[0],1]),X])
    
    def normalize_train(self,X):
        mean = np.mean (X,axis=0)
        std = np.std (X,axis=0)
        X = (X-mean)/std
        X = self.add_x0(X)
        return X,mean,std
    
    def normalize_test(self,X,mean,std):
        X = (X-mean)/std
        X = self.add_x0(X)
        return X
    
    def rank(self,X):
        u ,s ,v = np.linalg.svd(X)
        return len ([x for x in s if x>0.0005])
    
    def check_fullRank(self,X):
        rank = self.rank(X)
        if rank == min(X.shape):
            self.full_rank = True
            print ("X is full rank")
        else:
            self.full_rank = False
            print ("X is not Full Rank")
            
    def check_lowRank(self,X):
        if X.shape[0] < X.shape[1]:
            self.low_rank = True
            print ("X is low rank")
        else:
            self.low_rank = False
            print ("X is not low rank")
            
    def closed_form_solution(self,X,y):
        if self.regularization == 'L2':
            self.theta = np.linalg.inv(X.T.dot(X) + self.lambda_ * np.eye(X.shape[1])).dot(X.T).dot(y)
        else:
            self.theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        
    def predict (self,X):
        return X.dot(self.theta)
    
    def sse (self,X,y):
        y_hat = self.predict(X)
        return ((y_hat-y)**2).sum()
    
    def cost_function(self,X,y):
        return self.sse(X,y)/2
    
    def cost_derivative(self,X,y):
        y_hat = self.predict(X)
        return (y_hat - y).dot(X)
    
    def gradient_descent(self,X,y):
        errors = []
        prev_error = float('inf')
        
        for i in tqdm(range(self.max_iteration), colour = 'red'):
            self.theta -= self.learning_rate * self.cost_derivative(X,y)
            
            if self.regularization == 'L2':
                self.theta -= self.learning_rate * self.lambda_ * self.theta
            
            error = self.cost_function(X,y)
            errors.append(error)
            
            if abs(error - prev_error) < self.epsilon:
                print ("Model Stopped Learning")
                break
        self.plot_rmse(errors)
    
    def stochastic_gradient_descent(self, X, y):
        errors = []
        prev_error = float('inf')
        
        for i in tqdm(range(self.max_iteration), colour = 'red'):
            random_index = np.random.randint(0, X.shape[0])
            X_i = X[random_index, :].reshape(1, -1)
            y_i = y[random_index]
            
            self.theta -= self.learning_rate * self.cost_derivative(X_i, y_i)
            
            if self.regularization == 'L2':
                self.theta -= self.learning_rate * self.lambda_ * self.theta
            
            error = self.cost_function(X, y)
            errors.append(error)
            
            if abs(error - prev_error) < self.epsilon:
                print ("Model Stopped Learning")
                break
        self.plot_rmse(errors)
    
    def fit(self):
        X_train,X_test,y_train,y_test = self.split_data()
        X_train,mean,std = self.normalize_train(X_train)
        X_test = self.normalize_test(X_test,mean,std)
        self.check_fullRank(X_train)
        self.check_lowRank(X_train)
        
        if self.full_rank and not self.low_rank and X_train.shape[1] < 1000 and not self.gd:
            self.closed_form_solution(X_train,y_train)
        else:
            self.theta = np.ones(X_train.shape[1])
            if self.gd:
                self.gradient_descent(X_train, y_train)
            else:
                self.stochastic_gradient_descent(X_train, y_train)
        print(self.theta)
    

    def plot_rmse(self, error_sequence):
        """
        @X: error_sequence, vector of rmse
        @does: Plots the error function
        @return: plot
        """
        # Data for plotting
        s = np.array(error_sequence)
        t = np.arange(s.size)

        fig, ax = plt.subplots()
        ax.plot(t, s)

        ax.set(xlabel='iterations', ylabel=list(range(len(error_sequence))))
        ax.grid()

        plt.legend(bbox_to_anchor=(1.05,1), loc=2, shadow=True)
        plt.show()    
    

        



#lr = LinearRegression(X,y,learning_rate=-.0001,max_iteration=1000,epsilon=0.1,gd=False)
    
#lr.fit().
lr = LinearRegression(X,y,learning_rate=-.0001,max_iteration=5000,epsilon=0.1,gd=True, regularization='L2', lambda_=0.01)
lr.fit()