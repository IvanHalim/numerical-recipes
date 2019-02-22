# Import libraries
import pandas as pd
import numpy as np

# Number of random draws from the prior
n_draws = 10000

# Sample n_draws draws from the prior distribution into a pandas Series (to have convenient
# methods available for histograms and descriptive statistics, e.g. median)
prior_rate = pd.Series(np.random.uniform(0, 1, size = n_draws))

# It's always good to eyeball the prior to make sure it looks ok
prior_rate.hist()

# Defining the generative model
def gen_model(prob):
    return(np.random.binomial(16,prob))

# Simulating the data using the parameters from the prior and the generative model
sim_data = list()
for p in prior_rate:
    sim_data.append(gen_model(p))

# Observed data
observed_data = 6

# Here you filter off all draws that do not match the data
post_rate = prior_rate[list(map(lambda x: x == observed_data, sim_data))]

post_rate.hist() # Eyeball the posterior

# See that we got enough draws left after the filtering.
# There are no rules here, but you probably want to aim for >1000 draws.

# Now you can summarize the posterior, where a common summary is to take the mean or the median posterior,
# and perhaps a 95% quantile interval.

print('Number of draws left: %d\nPosterior mean: %.3f\nPosterior median: %.3f\nPosterior 95%% quantile interval: %.3f-%.3f' %
      (len(post_rate), post_rate.mean(), post_rate.median(), post_rate.quantile(.025), post_rate.quantile(.975)))
