from util import *
from scipy.optimize import minimize
from util import softplus_np
from plots import plot_softplus_poly
import matplotlib.pyplot as plt

# This is the model
# x -> data
# choices -> 0 or 1
# we optimize for sigma directly
""" old probit model
def make_cost_function(x, choices):
    def cost(pars):
        return -get_likelihood(x, pars[0], choices)
    return cost
"""
# to constrain the parameters -> softplus function -> N(x, s2)
# to keep the parameters as small as possible -> L2 regularization
# What we do not do:
# -- adjust optimization bounds -> to what???
# -- improve initialization -> with what information???
def make_cost_function(x1, x2, choices, alpha=.1):
    def cost(pars):
        llh = -get_likelihood(x2 - x1, softplus_np(pars), choices)
        reg = alpha * np.sum(np.square(pars))
        return llh + reg
    return cost

def fit_simple_probit(x1, x2, choices, init):
    alpha = .1
    cost = make_cost_function(x1, x2, choices, alpha)
    result = minimize(cost, init, bounds=[(1e-6, np.inf)]*len(init), method='L-BFGS-B') # cost ist the created cost function [5.0] is the list of parameters of the cost function
    max_likelihood = -cost(result.x) + alpha * np.sum(np.square(result.x))
    return result.x, max_likelihood

def make_simple_probid_pred(d):
    opt_paras, _ = fit_simple_probit(d['n1'], d['n2'], d['choice'], [8.])
    print(opt_paras)
    d['pred'] = get_choice_prob(d['n2-n1'], opt_paras[0]) # this does not modify the original dataframe
    plot_softplus_poly(opt_paras)
    return d


# simple probit with polynomial -> N(x, s2(x))
def make_cost_function_poly_probit(x1, x2, choices, alpha=.1):
    def cost(pars):
        llh = -get_likelihood(x2 - x1, make_softplus_poly(pars)(x2 - x1), choices)
        reg = alpha * np.sum(np.square(pars))
        return llh + reg
    return cost

def fit_poly_probit(x1, x2, choices, init):
    alpha = .1
    cost = make_cost_function_poly_probit(x1, x2, choices, alpha)
    # result = minimize(cost, init, bounds=[(1e-6, np.inf)]*len(init), method='L-BFGS-B') # cost ist the created cost function [5.0] is the list of parameters of the cost function
    # alternative - cost ist the created cost function [5.0] is the list of parameters of the cost function
    result = minimize(cost, init, method='Nelder-Mead')
    max_likelihood = -cost(result.x) + alpha * np.sum(np.square(result.x))
    return result.x, max_likelihood

def make_poly_probid_pred(d):
    opt_paras, _ = fit_poly_probit(d['n1'], d['n2'], d['choice'], [1e-3, 1., 2])
    poly = make_poly(opt_paras)
    d['pred'] = get_choice_prob(d['n2-n1'], poly(d['n2-n1'])) # this does not modify the original dataframe
    plot_softplus_poly(opt_paras)
    return d

"""This model is of no use anymore.
# This is another model
# n1, n2 -> replace x as the data
# choices -> 0 or 1
def make_cost_abs_diff(x1, x2, choices, alpha=.1):
    def cost(pars):
        mu, sd = get_diff_dist(x1, softplus_np(pars), x2, softplus_np(pars))
        llh = -get_likelihood(mu, sd, choices)
        reg = alpha * np.sum(np.square(pars))
        return llh + reg
    return cost

def fit_abs_diff(x1, x2, choices, init):
    cost = make_cost_abs_diff(x1, x2, choices)
    result = minimize(cost, init, bounds=[(1e-6, np.inf)]*len(init), method='L-BFGS-B')
    return result.x
"""






# N(m(x), s2)
def make_cost_function_mean_function(x1, x2, choices, alpha=.1):
    def cost(pars):
        mu1 = make_poly(pars[:-1])(x1)
        sd12 = softplus_np(pars[-1])
        mu2 = make_poly(pars[:-1])(x2)
        mu, sd = get_diff_dist(mu1, sd12, mu2, sd12)

        llh = -get_likelihood(mu, sd, choices)
        reg = alpha * np.sum(np.square(pars))

        return llh + reg
    return cost

def fit_mean(x1, x2, choices, init):
    alpha = .1
    cost = make_cost_function_mean_function(x1, x2, choices, alpha)
    result = minimize(cost, init, bounds=[(1e-10, np.inf)]*len(init), method='L-BFGS-B')
    max_likelihood = -cost(result.x) + alpha * np.sum(np.square(result.x))
    return result.x, max_likelihood

def make_mean_pred(d):
    opt_paras, _ = fit_mean(d['n1'], d['n2'], d['choice'], [1e-3, 1., 8.])
    print(opt_paras)
    poly = make_poly(opt_paras[:-1])
    d['pred'] = get_choice_prob(poly(d['n2-n1']), softplus_np(opt_paras[-1])) # this does not modify the original dataframe
    return d



# N(m(x), s2(x))
def make_cost_function_mean_variance_functions(x1, x2, choices, alpha=.1):
    def cost(pars):
        n_params = len(pars) // 2
        mu1 = make_poly(pars[:n_params])(x1)
        sd1 = make_softplus_poly(pars[n_params:])(x1)
        mu2 = make_poly(pars[:n_params])(x2)
        sd2 = make_softplus_poly(pars[n_params:])(x2)
        mu, sd = get_diff_dist(mu1, sd1, mu2, sd2)

        llh = -get_likelihood(mu, sd, choices)
        reg = alpha * np.sum(np.square(pars))

        return llh + reg
    return cost

def fit_mean_variance(x1, x2, choices, init):
    alpha = .1
    cost = make_cost_function_mean_variance_functions(x1, x2, choices, alpha)
    # result = minimize(cost, init, bounds=[(1e-6, np.inf)]*len(init), method='L-BFGS-B')
    # alt
    result = minimize(cost, init, method='Nelder-Mead')
    max_likelihood = -cost(result.x) + alpha * np.sum(np.square(result.x))
    return result.x, max_likelihood

def make_mean_variance_pred(d):
    opt_paras, _ = fit_mean_variance(d['n1'], d['n2'], d['choice'],
                                     [3., 3., 3.])
    n_params = len(opt_paras) // 2
    mu_poly = make_poly(opt_paras[:n_params])
    sd_poly = make_softplus_poly(opt_paras[n_params:])
    d['pred'] = get_choice_prob(mu_poly(d['n2-n1']), sd_poly(d['n2-n1'])) # this does not modify the original dataframe
    return d



def evaluate_model (eval_model, eval_inits, model_name, Eval_df, df):
    Eval_df[model_name + '_' + 'BIC'] = None
    Eval_df[model_name + '_' + 'AIC'] = None
    Eval_df[model_name + '_' + 'LL'] = None
    Eval_df[model_name + '_' + 'opt_params'] = None
    def make_eval_func(model, init):
        def make_eval_subj(d):
            # Fit and compute BIC for the efficient_coding model
            opt_params, ll = model(d['n1'], d['n2'], d['choice'], init)
            num_params = len(opt_params)
            Eval_df.loc[d.reset_index()['subject'][1], num_params][model_name + '_' + 'LL'] = ll
            Eval_df.loc[d.reset_index()['subject'][1], num_params][model_name + '_' + 'opt_params'] = opt_params
            Eval_df.loc[d.reset_index()['subject'][1], num_params][model_name + '_' + 'BIC'] = \
                calculate_bic(num_params, 216, ll)
            Eval_df.loc[d.reset_index()['subject'][1], num_params][model_name + '_' + 'AIC'] = \
                calculate_aic(num_params, ll)
        return make_eval_subj

    for init in eval_inits:
        create_eval_cols = make_eval_func(eval_model, init)
        df.groupby('subject').apply(create_eval_cols)

