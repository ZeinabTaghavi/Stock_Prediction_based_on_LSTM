# libs
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


class load_dataset:
    
    
    def __init__(self):
        self.scale_dataset = MinMaxScaler(feature_range= (0,1))
        return None
    
    def scale_trainset(self,dataset_path):
        
        self.dataset_train = pd.read_csv(dataset_path)
        trainset = self.dataset_train.iloc[:,1:2].values
        trainset_scaled = self.scale_dataset.fit_transform(trainset)
        
        return trainset_scaled
    
    
    def RNN_X_train_y_train(self,time_step, trainset_scaled):
        X_train = []
        y_train = []
        
        for i in range(time_step , len(trainset_scaled)):
            X_train.append(trainset_scaled[i- time_step:i , 0])
            y_train.append(trainset_scaled[i, 0])
        
        X_train= np.array(X_train)
        y_train =  np.array(y_train)
        
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        
        
        return X_train, y_train
    
    def scale_testset(self,dataset_test_path ,time_step):
        
        self.dataset_test = pd.read_csv(dataset_test_path)
        self.real_stock_price = self.dataset_test.iloc[:, 1:2].values
        
        dataset_total = pd.concat((self.dataset_train['Open'], self.dataset_test['Open']) , axis=0)
        input_dataset = dataset_total[- (time_step + len(self.dataset_test)):].values
        
        input_dataset = input_dataset.reshape(-1 ,1)
        
        input_dataset_scale = self.scale_dataset.transform(input_dataset)
        
        X_test = []
        
        for i in range(time_step , len(input_dataset_scale)):
            X_test.append(input_dataset_scale[i - time_step:i , 0])

        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        return X_test
    
    def inverse_transform_result(self , scaled_result):
        
        self.result = self.scale_dataset.inverse_transform(scaled_result)    
        return self.result
    
    
    def plot_result(self):
    
        # Visualising the results
        plt.plot(self.real_stock_price, color = 'red', label = 'Real Google Stock Price')
        plt.plot(self.result, color = 'blue', label = 'Predicted Google Stock Price')
        plt.title('Google Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Google Stock Price')
        plt.legend()
        plt.show()