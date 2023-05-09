import scipy.stats as ss
import numpy as np
import pandas as pd


def load_garcia2022():
    """Return a dataframe containing the behavioral data from the Magnitude task
    in Baretto Garcia et al., 2022
    """
    stream = '/Users/maximilianharl/Dropbox/001_university/008_ETH_UZH/002.Semester/IDB403/lab/code_repos/bauer/bauer/data/garcia2022.csv'
    df = pd.read_csv(stream, index_col=[0, 1, 2])
    df['log(n2/n1)'] = np.log(df['n2'] / df['n1'])
    return df


# standard stat functions
def get_posterior(mu1, sd1, mu2, sd2):
    var1, var2 = sd1 ** 2, sd2 ** 2
    return mu1 + (var1 / (var1 + var2)) * (mu2 - mu1), np.sqrt((var1 * var2) / (var1 + var2))


def get_diff_dist(mu1, sd1, mu2, sd2):
    return mu2 - mu1, np.sqrt(sd1 ** 2 + sd2 ** 2)


def get_choice_prob(x, sigma):
    p = 1 - ss.norm(x, sigma).cdf(0.0)
    return p


def get_likelihood(x, sigma, choices):
    p = get_choice_prob(x, sigma)
    return np.sum(np.log(ss.bernoulli(p).pmf(choices)))


# activation functions
def softplus_np(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def poly_sigma_mapping_3(x, pars):
    return softplus_np(pars[0] + pars[1] * x + pars[2] * x ** 2 + pars[3] * x ** 3)


def poly_sigma_mapping_2(x, pars):
    return softplus_np(pars[0] + pars[1] * x + pars[2] * x ** 2)


def make_cost_function_sensory(x1, x2, choices):
    def cost(pars):
        mu, sd = get_diff_dist(x1, poly_sigma_mapping_2(x1, pars), x2,
                               poly_sigma_mapping_2(x2, pars))  # we want our sd1 and sd2 to be dependent on x1 and x2
        return -get_likelihood(mu, sd, choices)

    return cost


def make_poly(pars):
    def poly(x):
        ret = 0
        for i in range(len(pars)):
            ret += pars[i] * x ** i
        return ret

    return poly


def make_softplus_poly(pars):
    def poly(x):
        ret = 0
        for i in range(len(pars)):
            ret += pars[i] * x ** i
        return softplus_np(ret)

    return poly


def calculate_bic(k, n, nll):
    return k * np.log(n) - 2 * nll


def calculate_aic(k, nll):
    return 2 * k - 2 * nll


poly_probit_inits = [[15.],
                     [15., .5],
                     [1e-3, 1., 2],
                     [1e-3, 1., 2, 1e-3],
                     [1e-3, 1., 2, 1e-3, 1e-5],
                     [1e-3, 1., 2, 1e-3, 1e-5, 1e-8],
                     [1e-3, 1., 2, 1e-3, 1e-5, 1e-8, 1e-8],
                     [1e-3, 1., 2, 1e-3, 1e-3, 1e-8, 1e-8, 1e-8]]

eff_coding_inits = [[3.0],
                    [1e-3, 1.],
                    [1e-3, 1., 2],
                    [1e-3, 1., 2, 1e-3],
                    [1e-3, 1., 2, 1e-3, 1e-5],
                    [1e-3, 1., 2, 1e-3, 1e-5, 1e-8],
                    [1e-3, 1., 2, 1e-3, 1e-5, 1e-8, 1e-8],
                    [1e-3, 1., 2, 1e-3, 1e-3, 1e-8, 1e-8, 1e-8]]

eff_coding_indiv_inits = [[3.0, 3.0],
                          [1e-3, 1., 1e-3, 1.],
                          [1e-3, 1., 2, 1e-3, 1., 2],
                          [1e-3, 1., 2, 1e-3, 1e-3, 1., 2, 1e-3],
                          [1e-3, 1., 2, 1e-3, 1e-5, 1e-3, 1., 2, 1e-3, 1e-5],
                          [1e-3, 1., 2, 1e-3, 1e-5, 1e-8, 1e-3, 1., 2, 1e-3, 1e-5, 1e-8],
                          [1e-3, 1., 2, 1e-3, 1e-5, 1e-8, 1e-8, 1e-3, 1., 2, 1e-3, 1e-5, 1e-8, 1e-8],
                          [1e-3, 1., 2, 1e-3, 1e-3, 1e-8, 1e-8, 1e-8, 1e-3, 1., 2, 1e-3, 1e-3, 1e-8, 1e-8, 1e-8]]

mean_inits = [[1e-3, 8.],
              [1e-3, 1., 8.],
              [3., 1., 1e-3, 8.],
              [3., 1., 1e-3, 1e-5, 8.],
              [3., 1., 1e-3, 1e-5, 1e-5, 8.],
              [3., 1., 1e-3, 1e-5, 1e-6, 1e-6, 8.],
              [3., 1., 1e-3, 1e-5, 1e-6, 1e-6, 1e-7, 8.]]

mean_variance_inits = [[3., 3.],
                       [3., 3., 3.],
                       [3., 3., 3., 3.],
                       [3., 3., 3., 3., 3.],
                       [3., 3., 3., 3., 3., 3.],
                       [3., 3., 3., 3., 3., 3., 3.],
                       [3., 3., 3., 3., 3., 3., 3., 3.]]



