# libs

import load_dataset
from model import RNN_model



if __name__=='__main__':
    
    dataset_path ='Google_stockprice_dataset/Google_Stock_Price_Train.csv'
    dataset_test_path = 'Google_stockprice_dataset/Google_Stock_Price_Test.csv'
    
    load = load_dataset.load_dataset()
    trainset_scaled = load.scale_trainset(dataset_path)
    
    # 1 result based on 80(time_step) steps before
    time_step = 80
    X_train, y_train = load.RNN_X_train_y_train(time_step, trainset_scaled)
    
    regressor = RNN_model()
    regressor.fit(X_train, y_train , nb_epochs = 100, nb_batch_size = 32)
    

    X_test = load.scale_testset(dataset_test_path ,time_step)
    
    
    scaled_result = regressor.predict(X_test)

    result = load.inverse_transform_result(scaled_result)
    
    load.plot_result()
    
    
