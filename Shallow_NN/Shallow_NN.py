import sys
sys.path.insert(0, '../utils')
from planar_utils import load_planar_dataset,sigmoid
import numpy as np

class ShallowNN:

    def __repr__(self):
        return "ShallowNN()"
    
    def __str__(self):
        return "Shallow NN with 1Hidden layer  with tanh activation & Single Output with Sigmoid"

    def __init__(self, n_h =1, epochs=10000, learning_rate=4.2):
        self.iterations = epochs
        self.learning_rate = learning_rate
        self.n_h = n_h
#expecting Min 2 features
        self.in_features=2
#weights and bias init
        self.b1 = np.zeros((self.n_h, 1))
        self.W1 = np.random.randn(self.n_h, self.in_features)*0.05

        self.b2 = 0.0
        self.W2 = np.random.randn(1,self.n_h)*0.05
#Collector and Activations

        self.A1 = 0.0
        self.A2 = 0.0

    def layer_sizes(self, x, y):
        assert(x.shape[1] ==y.shape[1])
        self.in_features = x.shape[0]
        self.n_y = y.shape[0]
        self.samples = x.shape[1]

    def _init_params_(self):
        from copy import copy
        self.W1 = np.random.randn(self.n_h, self.in_features)*0.01
        self.b1 = np.zeros((self.n_h,1))

        self.W2 = np.random.randn(self.n_y,self.n_h)*0.01
        self.b2 = np.zeros((self.n_y,1))

        self.db1 = copy(self.b1) 
        self.dW1 = np.zeros((self.n_h, self.in_features))
        
        self.db2 = copy(self.b2)
        self.dW2 = np.zeros((self.n_y, self.n_h))

    def compute_cost(self, y, y_pred):
        assert (y.shape == y_pred.shape)
        m=y.shape[1]
        c = - np.multiply(y,np.log(y_pred))-np.multiply((1-y),np.log((1-y_pred)))
        cost = np.sum(c)/m
        return cost

    def predict(self,x):
        y = self.forward_pass(x)
        #applying threshold 
        y = y>=0.5
        return y

    def forward_pass(self, x):

        Z1 = np.dot(self.W1,x)+self.b1
        self.A1 = np.tanh(Z1)

        Z2 = np.dot(self.W2, self.A1)+self.b2
        self.A2 = sigmoid(Z2)

        return self.A2

    def update_params(self):
        W1 = self.W1 - (self.learning_rate*self.dW1)
        b1 = self.b1 - (self.learning_rate*self.db1)
        self.W1 = W1
        self.b1 = b1

        W2 = self.W2 - (self.learning_rate*self.dW2)
        b2 = self.b2 - (self.learning_rate*self.db2)
        self.W2 = W2
        self.b2 = b2

    def backward_pass(self,x,y):
        assert (y.shape == self.A2.shape)
        dZ2 = (self.A2-y)
        m = y.shape[1] 
        dW2 = np.dot(dZ2,self.A1.T)/m
        db2 = np.sum(dZ2, axis=1,keepdims=True)/m
    
    
        dZ1 = np.multiply(np.dot(self.W2.T,dZ2),1-np.power(self.A1,2))
    
    
        dW1 = np.dot(dZ1,x.T)/m
        db1 = np.sum(dZ1,axis=1,keepdims=True)/m
    
        self.dW1 = dW1
        self.db1 = db1
        self.dW2 = dW2
        self.db2 = db2


    def fit(self,x,y,print_cost=False):
        self.layer_sizes(x,y)
        self._init_params_()
        
        for i in range(self.iterations):
            self.forward_pass(x)
            self.backward_pass(x,y)
            self.update_params()
            if print_cost and (i%500==0):
                print ("cost at iteration:"+str(i)+"is: "+str(self.compute_cost(y,self.A2)))
    def accuracy(self,y,y_pred):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y,y_pred)

    def scores(self, y, y_pred):
        from sklearn.metrics import classification_report
        print (classification_report(y,y_pred))
    #Example for Cancer Classification

    def cancer_data(self):
        import pandas as pd
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.model_selection import train_test_split
        data = pd.read_csv('../utils/datasets/cancer.csv')
        Y = y = data['Diagnosis']
        X = x = data.drop('Diagnosis',axis=1)
        scaler = MinMaxScaler()
        X=scaler.fit_transform(x)
        x_train,x_test,y_train,y_test = train_test_split(X,y)
        x_train = x_train.T
        x_test = x_test.T
        y_train = np.array(y_train).reshape(1,y_train.shape[0])
        y_test = np.array(y_test).reshape(1,y_test.shape[0])
        from copy import deepcopy as deep_copy
        self.x_train = deep_copy(x_train)
        self.x_test  = deep_copy(x_test)

        self.y_train = deep_copy(y_train)
        self.y_test  = deep_copy(y_test)
        Y = np.array(Y).reshape(1,Y.shape[0])
        return X.T, Y

    def planar_data(self,s=0.75):
        X,Y = load_planar_dataset()
        m = X.shape[1]
        m = int(m*s)
        self.x_train = X[0:X.shape[0],0:m]
        self.x_test = X[0:X.shape[0],m:X.shape[1]]
        self.y_train = Y[0:Y.shape[0],0:m]
        self.y_test = Y[0:Y.shape[0],m:Y.shape[1]]
        return X,Y

model = ShallowNN(n_h=8)
X,Y = model.cancer_data()
model.fit(X, Y)
yp = model.predict(X)
print(model.accuracy(Y[0],yp[0]))
