import numpy as np
import pickle


def load_object(file_name):
    "load the pickled object"
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def view_data(data_path):
    data = load_object(data_path)
    prices = data['prices']
    names = data['features']['names']
    features = data['features']['values']
    print(prices.shape)
    print(names)
    print(features.shape)
    return prices, features


class Strategy():
    def __init__(self):
        pass

    def handle_update(self, inx, price, factors):
        """Put your logic here
        Args:
            inx: zero-based inx in days
            price: [num_assets, ]
            factors: [num_assets, num_factors]
        Return:
            allocation: [num_assets, ]
        """
        assert price.shape[0] == factors.shape[0]
        return np.array([1.0] * price.shape[0])
