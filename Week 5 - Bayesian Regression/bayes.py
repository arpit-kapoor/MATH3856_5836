import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Functions
def read_data_from_csv(filename, test_ratio=0.4, shuffle=True):

    # Read as DataFrame
    df = pd.read_csv(filename, header=None)
    data = df.values

    # Extract Input and Target Arrays
    data_x = data[:, :8]
    data_y = data[:, 8]

    # TODO: Scale the data
    scaler = StandardScaler()
    scaler = scaler.fit(data_x)
    x_scaled = scaler.transform(data_x)

    # Split into train and test set
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, data_y, test_size=test_ratio, shuffle=shuffle)

    # return data_x, data_y
    return x_train, x_test, y_train, y_test



# A simple linear regression class
class LinearRegression:

    def __init__(self):
        self.coef_ = None  # Coefficients (slope(s))
        self.intercept_ = None  # Intercept (bias)
    
    def fit(self, X, y):
        """
        Fit linear model.
        Parameters:
        X: 2D array, shape (n_samples, n_features)
        y: 1D array, shape (n_samples,)
        """
        # Add bias (intercept) term to X
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Calculate weights using the Normal Equation: theta = (X.T * X)^(-1) * X.T * y
        theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        
        self.intercept_ = theta[0]  # Intercept is the first element of theta
        self.coef_ = theta[1:]  # Coefficients are the rest of the elements
    
    def predict(self, X):
        """
        Predict using the linear model.
        Parameters:
        X: 2D array, shape (n_samples, n_features)
        """
        # Add bias (intercept) term to X
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(np.r_[self.intercept_, self.coef_])
    
    def score(self, X, y):
        """
        Returns the coefficient of determination R^2 of the prediction.
        Parameters:
        X: 2D array, shape (n_samples, n_features)
        y: 1D array, shape (n_samples,)
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

    def get_params(self):
        """
        Returns the model parameters (theta).
        Returns:
        theta: array, shape (n_features + 1,)
        The first element is the intercept, followed by the coefficients.
        """
        return np.concatenate([[self.intercept_], self.coef_])

    def set_params(self, theta):
        """
        Sets the model parameters (theta).
        Parameters:
        theta: array, shape (n_features + 1,)
        The first element should be the intercept, followed by the coefficients.
        """
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]


# Class to run MCMC sampling of regression parameters
class MCMC:
    def __init__(self, model, n_samples=1000, step_size=0.01, prior_mean=0, prior_std=10):
        self.model = model
        self.n_samples = n_samples      # Number of samples to draw in the MCMC chain
        self.step_size = step_size      # Step size for the random walk
        self.prior_mean = prior_mean    # Mean of the Gaussian prior
        self.prior_std = prior_std      # Std deviation of the Gaussian prior
        self.samples = None             # To store MCMC samples of parameters
        self.burnin = 0.5               # Ratio of samples for burn-in

    def gaussian_likelihood(self, y, y_pred, sigma=1.0):
        """
        Gaussian likelihood function for the observed data.
        Assumes a normal distribution of residuals (errors).
        """
        return np.sum(-0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * ((y - y_pred)**2) / sigma**2)

    def gaussian_prior(self, theta):
        """
        Gaussian prior for the parameters (theta).
        Assuming zero mean and a fixed standard deviation (prior_std).
        """
        return np.sum(-0.5 * np.log(2 * np.pi * self.prior_std**2) - 0.5 * ((theta - self.prior_mean)**2) / self.prior_std**2)

    def posterior(self, X, y, theta):
        """
        Computes the log posterior = log(likelihood) + log(prior)
        """
        # Generate predictions
        self.model.set_params(theta)
        y_pred = self.model.predict(X)

        # Compute log-likelihood
        log_likelihood = self.gaussian_likelihood(y, y_pred)

        # Compute lop-prior
        log_prior = self.gaussian_prior(theta)

        # Compute log-posterior 
        log_posterior = log_likelihood + log_prior

        return log_posterior

    def random_walk(self, theta_current):
        """
        Proposes a new set of parameters by taking a random step.
        """
        return theta_current + np.random.normal(0, self.step_size, len(theta_current))

    def fit(self, X, y):
        """
        Runs the MCMC chain to sample from the posterior distribution.
        """
        # Initial guess for theta (starting from random values)
        theta_current = np.random.randn(X.shape[1] + 1)
        
        # Store samples
        self.samples = np.zeros((self.n_samples, len(theta_current)))

        # Compute the current posterior
        posterior_current = self.posterior(X, y, theta_current)

        for i in range(self.n_samples):
            # Propose new theta
            theta_proposed = self.random_walk(theta_current)

            # Compute the posterior of the proposed sample
            posterior_proposed = self.posterior(X, y, theta_proposed)

            # Accept/reject the new sample using the Metropolis-Hastings criterion
            acceptance_ratio = np.exp(posterior_proposed - posterior_current)

            if np.random.rand() < acceptance_ratio:
                # Accept the new sample
                theta_current = theta_proposed
                posterior_current = posterior_proposed

            # Store the sample
            self.samples[i] = theta_current

    def get_posterior_samples(self):
        """
        Returns the MCMC samples.
        """
        return self.samples[int(self.burnin*self.n_samples):]




# Function to generate trace plot and posterior histogram from sampling
def plot_trace_and_hist(samples, feature_names=None, filename='trace.png'):
    """
    Plot traceplot and histogram for MCMC samples.
    
    Parameters:
    samples: 2D numpy array of shape (n_samples, n_parameters)
             MCMC samples of the parameters.
    feature_names: List of strings, names of the parameters (optional).
    """
    n_params = samples.shape[1]
    
    if feature_names is None:
        feature_names = [f'Parameter {i}' for i in range(n_params)]
    
    fig, axes = plt.subplots(n_params, 2, figsize=(10, 2 * n_params))

    for i in range(n_params):
        # Trace plot
        axes[i, 0].plot(samples[:, i], color='b')
        axes[i, 0].set_title(f'Trace plot of {feature_names[i]}')
        axes[i, 0].set_xlabel('Iteration')
        axes[i, 0].set_ylabel('Value')

        # Histogram
        axes[i, 1].hist(samples[:, i], bins=30, color='c', edgecolor='k', alpha=0.7)
        axes[i, 1].set_title(f'Posterior of {feature_names[i]}')
        axes[i, 1].set_xlabel('Value')
        axes[i, 1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(filename)




if __name__ == "__main__":
    # Read data
    filepath = 'data/ENB2012_data.csv'
    x_train, x_test, y_train, y_test = read_data_from_csv(filename=filepath)

    # Create a LinearRegression object
    model = LinearRegression()

    # Fit the model to data
    model.fit(x_train, y_train)

    # R^2 score
    r2_score = model.score(x_test, y_test)
    print(f"R^2 score: {r2_score:.2f}")

    # Get the model parameters
    params = model.get_params()
    print("Model parameters (theta):", params)

    # Bayesian Linear Regression
    model = LinearRegression()
    sampler = MCMC(model=model, n_samples=50000, step_size=0.02)

    # Fit the model to the data using MCMC
    sampler.fit(x_train, y_train)

    # Get the MCMC samples of the parameters
    samples = sampler.get_posterior_samples()
    print(samples.shape)

    # Plot trace and hist
    plot_trace_and_hist(samples)