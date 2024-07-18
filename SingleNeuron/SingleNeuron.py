import sys
sys.path.insert(0, '../utils')
from planar_utils import load_planar_dataset,sigmoid
import numpy as np
class SingleNeuron:

    def __repr__(self):
        return "SingleNeron()"
    
    def __str__(self):
        return "Single Neuron with sigmoid activation"

    def __init__(self, epochs=20000, learning_rate=0.9):
        self.iterations = epochs
        self.learning_rate = learning_rate
#expecting Min 2 features
        self.in_features=2

#weights and bias init
        self.b = 0.0
        self.W = np.random.randn(1,self.in_features)*0.05

#Collector and Activations
        self.Z = 0.0
        self.A = 0.0

    def _init_params_(self, n_w, n_b=1):

        self.W = np.random.randn(1,n_w)*0.01
        self.b = 0.0
        self.db = 0.0 
        self.dW = np.zeros((1,n_w))

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

        self.Z = np.dot(self.W,x)+self.b
        self.A = sigmoid(self.Z)
        return self.A

    def update_params(self):
        W = self.W - (self.learning_rate*self.dW)
        b = self.b - (self.learning_rate*self.db)
        self.W = W
        self.b = b

    def backward_pass(self,x,y,yp):
        assert (y.shape == yp.shape)
        dZ = (yp-y)
        m = y.shape[1]
        self.dW = np.dot(dZ, x.T)/m
        self.db = np.sum(dZ,axis=1,keepdims=True)/m
        self.update_params()

    def fit(self,x,y,print_cost=False):
        self._init_params_(x.shape[0])
        
        for i in range(self.iterations):
            yp = self.forward_pass(x)
            self.backward_pass(x,y,yp)
            if print_cost and (i%500==0):
                print ("cost at iteration:"+str(i)+"is: "+str(self.compute_cost(y,yp)))

    def scores(self, y, y_pred):
        from sklearn.metrics import classification_report
        print (classification_report(y,y_pred))

    #Example for Cancer Classification
    def data_preprocess(self):    
        import pandas as pd
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.model_selection import train_test_split
        data = pd.read_csv('cancer.csv')
        y = data['Diagnosis']
        x = data.drop('Diagnosis',axis=1)
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


model = SingleNeuron()
model.data_preprocess()
model.fit(model.x_train, model.y_train)
yp_train = model.predict(model.x_train)
yp_test  = model.predict(model.x_test)
model.scores(model.y_train[0],yp_train[0])
model.scores(model.y_test[0],yp_test[0])
