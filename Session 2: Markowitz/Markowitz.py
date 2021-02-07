#Requires two imports

import numpy as np
import logging

class Markowitz():

    "Will remain a static class and take updates"

    def __init__(self,r, r_cov):
        """
        Purpose of thie class is to generate an optimal portfolio weighting based on a given mean
        and covariance matrix
        """
        self.r = r
        self.r_cov = r_cov

    def normal_update(self,desired_r,r,r_cov):
        """
        Update function will be called when a covariance and return matrix are given to be processed.

        Returns:
            (strat_alloc (np.array (n,),strat_var (float))
        """
        self.r = r
        self.r_cov = r_cov

        strat_alloc = self.design_portfolio_for_return(desired_r)
        strat_var = self.variance_of_strategy(strat_alloc)

        return (strat_alloc ,strat_var)

    def multi_update(self,desired_rs,r,r_cov):
        """
        Update function will be called when a covariance and return matrix are given to be processed.

        Returns:
            (strat_alloc (np.array (n,n) ,strat_var (np.array (n,))
        """
        self.r = r
        self.r_cov = r_cov

        strat_allocs = self.design_portfolios_for_return(desired_rs)
        strat_vars = np.array([self.variance_of_strategy(strat_alloc) for strat_alloc in strat_allocs])

        return (strat_allocs ,strat_vars)

    def design_portfolio_for_return(self,r,reinvert=True):
        """
        Given a specific return, we design a portfolio allocation in order to achieve that mean. We additionally return
        the variance associated with that return. The notation for this derivation is consistent with notes that will be
        included in this file.

        """
        logging.info("Calculating Optimal Portfolio")
        e = np.ones(self.r.shape)

        #Will automatically handle errors
        if reinvert:
            self.inv_cov = self.invert_covariance_matrix()

        #Calculating entries in C
        c_r=np.dot(self.inv_cov,self.r)
        c_e=np.dot(self.inv_cov,e)

        c_rr= np.dot(self.r,c_r)
        c_re= np.dot(self.r,c_e)
        c_er= np.dot(e,c_r)
        c_ee= np.dot(e,c_e)

        det_C = c_ee*c_rr-c_re*c_er

        C_inv = np.array([[c_ee,-c_re],[-c_er,c_rr]])/det_C

        #Deriving the Lagrangian Multipliers
        lamb, mu = np.dot(C_inv,np.array([r,1]))

        #Optimal Asset Allocation
        omega = lamb*c_r+mu*c_e

        return omega

    def design_portfolios_for_return(self,rs,reinvert=True):
        """
        Uses the two fund theorem and vectorized operations in order to quickly create a
        range of different portfolios to run.
        """
        logging.info("Calculating Optimal Portfolios")
        e = np.ones(self.r.shape)

        #Will automatically handle errors
        if reinvert:
            self.inv_cov = self.invert_covariance_matrix()

        #Calculating the two asset allocations for optimal
        v_1=np.dot(self.inv_cov,e) #min variance
        v_2=np.dot(self.inv_cov,self.r) #max return

        #Calculating entries in C
        c_r=v_2
        c_e=v_1

        c_rr= np.dot(self.r,c_r)
        c_re= np.dot(self.r,c_e)
        c_er= np.dot(e,c_r)
        c_ee= np.dot(e,c_e)

        det_C = c_ee*c_rr-c_re*c_er
        alphas = [c_ee*(c_rr-c_er*r)/det_C for r in rs]

        #Two identical funds
        w_1=v_1/c_ee
        w_2=v_2/c_er

        omegas = np.array([alpha*w_1+(1-alpha)*w_2 for alpha in alphas])

        return omegas

    def invert_covariance_matrix(self):
        """
        Will invert covariance matrices
        """
        logging.info("Attempting to Invert Matrix... ")
        try:
            return np.linalg.inv(self.r_cov)
        except:
            return self.handle_covariance_singularity()

    def variance_of_strategy(self,omega, reinvert=False):
        """
        Will always return the variance of the most current strategy. Will allow us to quantify risk.
        """
        logging.info("Calculating Variance of Strategy")
        if reinvert:
            self.inv_cov = self.invert_covariance_matrix()

        return np.dot(omega,np.dot(self.r_cov,omega))

    def handle_covariance_singularity(self,epsilon=1e-8):
        logging.error("Singular Matrix, perturbing it...")
        self.r_cov+=epsilon*np.eye(self.r_cov.shape[0])
        return self.invert_covariance_matrix()
