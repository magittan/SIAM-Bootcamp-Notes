from sklearn.preprocessing import StandardScaler
import numpy as np

class Preprocessing():
    
    """We transform the data when we iterate through it."""
    
    def __init__(self, train_prices, train_factors):
        """
        Below we have x represent the time interval the train data is over 
        and z represents the number of factors that are included. We are going to need
        a factor scaler for every stock and just one to modify the prices.
        
        Args:
            train_prices: (x+1,680) 
            train_factors: (x,680,z)
        
        """
        #Storing the last price for return calculations
        self.last_price = train_prices[-1,:]
        
        #Initializing Scalers
        self.factor_scalers = [StandardScaler() for i in range(train_factors.shape[1])]
        self.return_scaler = StandardScaler()
        
        #Calculating Daily Returns
        train_returns=((train_prices[1:]-train_prices[:-1])/train_prices[:-1])
        
        #Updating Factor Scalers
        for i in range(len(self.factor_scalers)):
            self.factor_scalers[i].partial_fit(train_factors[:,i,:])
            
        #Updating Returns
        self.return_scaler.partial_fit(train_returns)
        
    def handle_update_factors(self,factors_update):
        """
        Expect the update to be in the form of (1,680,z) where z is the number of factors we are considering
        """
        transformed_output = []
        for i in range(len(self.factor_scalers)):
            self.factor_scalers[i].partial_fit(factors_update[i,:])
            transformed_output.append(self.factor_scalers[i].transform(factors_update[i,:]))
        
        #With this update we plug in to obtain the return with the loading matrix
        return np.array(transformed_output).swapaxes(0,1)
        
    def handle_update_price(self,price_update):
        """
        Expect the update to be in the form of (680)
        """
        #Using Normal Returns Since log Returns was not very effective
        return_value = (price_update-last_price)/last_price
        last_price = price_update
        transformed_output=self.return_scaler.partial_fit(price_update)
        
        #With this update we attempt to refactor the loading matrix
        return np.array(transformed_output)
    
    def inverse_transform_return(self,r_prediction):
        """
        r_prediction (680) from the factor model to be transformed into tangible returns
        """
        self.return_scaler.inverse_transform(r_prediction)
    
    def normalize_factors(self,factor_data):
        transformed_output = []
        for i in range(len(self.factor_scalers)):
            transformed_output.append(self.factor_scalers[i].transform(factor_data[:,i,:]))
        return np.array(transformed_output)
        
    def normalize_returns(self,return_data):
        return self.return_scaler.transform(return_data)
    
    def derive_returns(self, price_data):
        return ((price_data[1:]-price_data[:-1])/price_data[:-1])