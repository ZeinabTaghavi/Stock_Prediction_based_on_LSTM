from keras.models import Sequential
from keras.layers import Dense , LSTM , Dropout


class RNN_model:
    
    def __init__(self):
        
        return None
    
    def fit(self ,X_train , y_train,nb_epochs,nb_batch_size):
    
        # creatring sequentoal model
        self.model = Sequential()
        
        
        # adding first lstm layer and dropout
        self.model.add(LSTM(units= 50, return_sequences= True , input_shape = (X_train.shape[1],1)))
        self.model.add(Dropout(rate = .2))
        
        # adding second lstm layer and dropout
        self.model.add(LSTM(units= 50, return_sequences= True))
        self.model.add(Dropout(rate = .2))
        
        # adding 3th lstm layer and dropout
        self.model.add(LSTM(units= 50, return_sequences= True))
        self.model.add(Dropout(rate = .2))
        
        
        # adding 4th lstm layer and dropout
        self.model.add(LSTM(units= 50, return_sequences= True))
        self.model.add(Dropout(rate = .2))
        
        
        # adding 5th lstm layer and dropout
        self.model.add(LSTM(units= 50, return_sequences= False))
        self.model.add(Dropout(rate = .2))
        
        # adding result layer
        self.model.add(Dense(units=1))
        print('RNN_model.fit : model created')
        
        # compile model
        # this model is regressor -> loss = 'mean_squared_error'
        self.model.compile(optimizer='adam' , loss = 'mean_squared_error')
        print('RNN_model.fit : model compiled')
        
        self.model.fit(X_train, y_train ,epochs = nb_epochs, batch_size = nb_batch_size)
        print('RNN_model.fit : model fitted')
        
        return self.model
    
    def predict(self, X_test):

        scaled_result = self.model.predict(X_test)
        return scaled_result

        