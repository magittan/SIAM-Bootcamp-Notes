import numpy as np
from strategy import Strategy, view_data

"""
Any attempt at using inspection/meta programming to access the ground truth
data will be met with the most severe penalty.
We will obfuscate the testing environment to make it impossible.
This test class is functionally equivalent, but much less sophisticated to the
actual testing kit we use. It is kept simple to make your life easy, and to
prevent you from having too much knowledge that might be otherwise inviting.
Make sure the simulation finishes within 4 hours.
"""


class TestStrategy():
    def __init__(self, p_series, f_series):
        """
        Args:
            p_series: [num_steps+1, num_assets]
            f_series: [num_steps  , num_assets, num_factors]
        """
        self.strategy = Strategy()
        self.N, self.A, self.F = f_series.shape
        self.p_series = p_series
        self.f_series = f_series
        daily_rf_rate = np.log(1.025) / 252
        self.log_rets = np.diff(np.log(p_series), axis=0) - daily_rf_rate

    def simulate(self):
        participant_ret = []
        for i in range(self.N):
            alloc = self.strategy.handle_update(
                i, self.p_series[i], self.f_series[i]
            )
            assert alloc.shape == (self.A, ) and alloc.dtype == np.float
            alloc = alloc / np.abs(alloc).sum()
            ret = alloc @ self.log_rets[i]
            participant_ret.append(ret)
        return np.array(participant_ret)

    @staticmethod
    def evaluate_sharpe(participant_ret):
        mean = participant_ret.mean()
        std = participant_ret.std()
        sharpe = mean / std * np.sqrt(252)
        return sharpe


if __name__ == '__main__':
    # during real testing we will be using different data loading functions
    # we use view_data from dev.py to show you how to load the pickled arrays
    p_series, f_series = view_data('C3_train.pkl')
    tester = TestStrategy(p_series, f_series)
    ret_series = tester.simulate()
    sharpe = tester.evaluate_sharpe(ret_series)
    print("Your annualized sharpe ratio is {}".format(sharpe))
