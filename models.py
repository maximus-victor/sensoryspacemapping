from util import *
from scipy.optimize import minimize
from util import softplus_np
from plots import plot_softplus_poly, plot_ply
import matplotlib.pyplot as plt

def get_train_test_split(df):
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()

    for subj in range(1, 65):
        subject_data = df.loc[subj]
        subject_test_data = subject_data.iloc[i::n_splits]
        subject_train_data = subject_data.drop(subject_test_data.index)

        subject_test_data['subject'] = subj  # Add 'subject' level back to MultiIndex
        subject_train_data['subject'] = subj  # Add 'subject' level back to MultiIndex

        test_data = pd.concat([test_data, subject_test_data])
        train_data = pd.concat([train_data, subject_train_data])

    test_data.set_index('subject', append=True, inplace=True)
    train_data.set_index('subject', append=True, inplace=True)
    return train_data, test_data

# Efficient Coding Model
# n1, n2 -> the data
# we do not have a concrete representation of n1 and n2
# choices -> 0 or 1
def make_cost_dyn_noise(x1, x2, choices, alpha=.1):
    def cost(pars):
        sd1 = make_softplus_poly(pars)(x1)
        sd2 = make_softplus_poly(pars)(x2)
        mu, sd = get_diff_dist(x1, sd1, x2, sd2)
        nll = -get_likelihood(mu, sd, choices)
        reg = alpha * np.sum(np.square(pars))
        return nll + reg
    return cost

def fit_dyn_noise(x1, x2, choices, init):
    alpha = .1
    cost = make_cost_dyn_noise(x1, x2, choices, alpha)
    result = minimize(cost, init, method='Nelder-Mead')
    max_likelihood = -(cost(result.x) - alpha * np.sum(np.square(result.x)))
    return result.x, max_likelihood

def make_dyn_noise_pred(d):
    opt_paras, _ = fit_dyn_noise(d['n1'], d['n2'], d['choice'], [1e-3, 1., 2.])
    poly = make_softplus_poly(opt_paras)
    mu, sd = get_diff_dist(d['n1'], poly(d['n1']), d['n2'], poly(d['n2']))
    d['dyn_noise_pred'] = get_choice_prob(mu, sd)
    plot_softplus_poly(opt_paras, xmin=0, xmax=30, ylim=30, color='cyan')
    return d

def make_dyn_noise_pred_cv(d):
    subject_test_data = d.iloc[1::5]
    subject_train_data = d.drop(subject_test_data.index)
    opt_paras, _ = fit_dyn_noise(subject_train_data['n1'], subject_train_data['n2'], subject_train_data['choice'], [1e-3, 1.])
    poly = make_softplus_poly(opt_paras)
    mu, sd = get_diff_dist(subject_test_data['n1'], poly(subject_test_data['n1']), subject_test_data['n2'], poly(subject_test_data['n2']))

    # Set the fifth element with the get_choice_prob values using a list comprehension
    d.iloc[np.arange(1, len(d), 5), d.columns.get_loc('dyn_noise_pred')] = get_choice_prob(mu, sd)

    plot_softplus_poly(opt_paras, xmin=0, xmax=30, ylim=30, color='blue')
    return d

def make_dyn_noise_pred_wSP(d):
    subject_test_data = d.iloc[1::5]
    subject_train_data = d.drop(subject_test_data.index)
    opt_paras, _ = fit_dyn_noise(subject_train_data['n1'], subject_train_data['n2'], subject_train_data['choice'], [1e-3, 1.])
    poly = make_poly(opt_paras)
    mu, sd = get_diff_dist(subject_test_data['n1'], poly(subject_test_data['n1']), subject_test_data['n2'], poly(subject_test_data['n2']))

    # Set the fifth element with the get_choice_prob values using a list comprehension
    d.iloc[np.arange(1, len(d), 5), d.columns.get_loc('dyn_noise_pred')] = get_choice_prob(mu, sd)

    plot_ply(opt_paras, xmin=0, xmax=30, ylim=30, color='blue')
    return d




# Efficient Individual Coding Model
# n1, n2 -> the data
# we do not have a concrete representation of n1 and n2
# choices -> 0 or 1
def make_cost_dyn_noise_mem(x1, x2, choices, alpha=.1):
    def cost(pars):
        n_params = len(pars) // 2
        sd1 = make_softplus_poly(pars[:n_params])(x1)
        sd2 = make_softplus_poly(pars[n_params:])(x2)
        mu, sd = get_diff_dist(x1, sd1, x2, sd2)
        nll = -get_likelihood(mu, sd, choices)
        reg = alpha * np.sum(np.square(pars))
        return nll + reg
    return cost

def fit_dyn_noise_mem(x1, x2, choices, init):
    alpha = .1
    cost = make_cost_dyn_noise_mem(x1, x2, choices, alpha)
    result = minimize(cost, init, method='Nelder-Mead')
    max_likelihood = -(cost(result.x) - alpha * np.sum(np.square(result.x)))
    return result.x, max_likelihood

def make_dyn_noise_mem_pred_cv(d):
    subject_test_data = d.iloc[1::5]
    subject_train_data = d.drop(subject_test_data.index)
    opt_paras, _ = fit_dyn_noise_mem(subject_train_data['n1'], subject_train_data['n2'], subject_train_data['choice'], [1e-3, 1., 2, 1e-3, 1., 2])
    n_params = len(opt_paras) // 2
    poly1 = make_softplus_poly(opt_paras[:n_params])
    poly2 = make_softplus_poly(opt_paras[n_params:])
    mu, sd = get_diff_dist(subject_test_data['n1'], poly1(subject_test_data['n1']), subject_test_data['n2'], poly2(subject_test_data['n2']))
    d.iloc[np.arange(1, len(d), 5), d.columns.get_loc('dyn_noise_mem_pred')] = get_choice_prob(mu, sd)

    plot_softplus_poly(opt_paras[:n_params], xmin=0, xmax=30, ylim=30, color='lime')
    plot_softplus_poly(opt_paras[n_params:], xmin=0, xmax=30, ylim=30, color='darkgreen')
    return d

def make_dyn_noise_mem_pred_wSP(d):
    subject_test_data = d.iloc[1::5]
    subject_train_data = d.drop(subject_test_data.index)
    opt_paras, _ = fit_dyn_noise_mem(subject_train_data['n1'], subject_train_data['n2'], subject_train_data['choice'], [1e-3, 1., 2, 1e-3, 1., 2])
    n_params = len(opt_paras) // 2
    poly1 = make_poly(opt_paras[:n_params])
    poly2 = make_poly(opt_paras[n_params:])
    mu, sd = get_diff_dist(subject_test_data['n1'], poly1(subject_test_data['n1']), subject_test_data['n2'], poly2(subject_test_data['n2']))
    d.iloc[np.arange(1, len(d), 5), d.columns.get_loc('dyn_noise_mem_pred')] = get_choice_prob(mu, sd)

    plot_ply(opt_paras[:n_params], xmin=0, xmax=30, ylim=30, color='lime')
    plot_ply(opt_paras[n_params:], xmin=0, xmax=30, ylim=30, color='darkgreen')
    return d



def get_cross_validation_splits(df, n_splits=5):
    cv_splits = []

    for i in range(n_splits):
        train_data = pd.DataFrame()
        test_data = pd.DataFrame()

        for subj in range(1, 65):
            subject_data = df.loc[subj]
            subject_test_data = subject_data.iloc[i::n_splits]
            subject_train_data = subject_data.drop(subject_test_data.index)

            subject_test_data['subject'] = subj  # Add 'subject' level back to MultiIndex
            subject_train_data['subject'] = subj  # Add 'subject' level back to MultiIndex

            test_data = pd.concat([test_data, subject_test_data])
            train_data = pd.concat([train_data, subject_train_data])

        test_data.set_index('subject', append=True, inplace=True)
        train_data.set_index('subject', append=True, inplace=True)

        cv_splits.append((train_data, test_data))

    return cv_splits

def compute_test_log_likelihood_2(test_n1, test_n2, test_choice, opt_params):
    # Use the optimized parameters to compute the mean (mu) and standard deviation (sd) for the test data
    n_params = len(opt_params) // 2
    sd1 = make_softplus_poly(opt_params[:n_params])(test_n1)
    sd2 = make_softplus_poly(opt_params[n_params:])(test_n2)
    mu, sd = get_diff_dist(test_n1, sd1, test_n2, sd2)

    # Compute the log-likelihood for the test data using the get_likelihood function
    return get_likelihood(mu, sd, test_choice)

def compute_test_log_likelihood_1(test_n1, test_n2, test_choice, opt_params):
    # Use the optimized parameters to compute the mean (mu) and standard deviation (sd) for the test data
    sd1 = make_softplus_poly(opt_params)(test_n1)
    sd2 = make_softplus_poly(opt_params)(test_n2)
    mu, sd = get_diff_dist(test_n1, sd1, test_n2, sd2)

    # Compute the log-likelihood for the test data using the get_likelihood function
    return get_likelihood(mu, sd, test_choice)

def evaluate_model_cv(eval_model, eval_inits, model_name, Eval_df, df, n_splits=5, type=1):
    Eval_df[model_name + '_'+ 'BIC'] = None
    Eval_df[model_name + '_' + 'AIC'] = None
    Eval_df[model_name + '_' + 'TEST_BIC'] = None
    Eval_df[model_name + '_' + 'TEST_AIC'] = None
    Eval_df[model_name + '_' + 'TEST_LL'] = None
    Eval_df[model_name + '_'+ 'LL'] = None
    Eval_df[model_name + '_' + 'opt_params'] = np.empty((len(Eval_df), 0)).tolist()

    def make_eval_func(model, init):
        num_params = len(init)
        def make_eval_subj(train_data, test_data):
            # Fit the model using the train_data
            opt_params, ll = model(train_data['n1'], train_data['n2'], train_data['choice'], init)

            if type == 1:
                test_ll = compute_test_log_likelihood_1(test_data['n1'], test_data['n2'], test_data['choice'], opt_params)
            else:
                test_ll = compute_test_log_likelihood_2(test_data['n1'], test_data['n2'], test_data['choice'],
                                                        opt_params)

            Eval_df.loc[train_data.reset_index()['subject'][1], num_params][model_name + '_' + 'TEST_LL'] = \
                test_ll if Eval_df.loc[train_data.reset_index()['subject'][1], num_params][model_name + '_' + 'TEST_LL'] \
                           is None else Eval_df.loc[train_data.reset_index()['subject'][1], num_params][model_name + '_' + 'TEST_LL'] + test_ll
            Eval_df.loc[train_data.reset_index()['subject'][1], num_params][model_name + '_' + 'LL'] = ll if Eval_df.loc[train_data.reset_index()['subject'][1], num_params][model_name + '_' + 'LL'] is None else Eval_df.loc[train_data.reset_index()['subject'][1], num_params][model_name + '_' + 'LL'] + ll
            Eval_df.loc[train_data.reset_index()['subject'][1], num_params][model_name + '_' + 'opt_params'].append(opt_params)
            Eval_df.loc[train_data.reset_index()['subject'][1], num_params][model_name + '_' + 'BIC'] = \
                calculate_bic(num_params, 216//n_splits, -ll) if Eval_df.loc[train_data.reset_index()['subject'][1], num_params][model_name + '_' + 'BIC'] is None else Eval_df.loc[train_data.reset_index()['subject'][1], num_params][model_name + '_' + 'BIC'] + calculate_bic(num_params, 216//n_splits, -ll)
            Eval_df.loc[train_data.reset_index()['subject'][1], num_params][model_name + '_' + 'AIC'] = \
                calculate_aic(num_params, -ll) if Eval_df.loc[train_data.reset_index()['subject'][1], num_params][model_name + '_' + 'AIC'] is None else Eval_df.loc[train_data.reset_index()['subject'][1], num_params][model_name + '_' + 'AIC'] + calculate_aic(num_params, -ll)
            Eval_df.loc[train_data.reset_index()['subject'][1], num_params][model_name + '_' + 'TEST_BIC'] = \
                calculate_bic(num_params, 216 // n_splits, -test_ll) if Eval_df.loc[train_data.reset_index()['subject'][1], num_params][model_name + '_' + 'TEST_BIC'] is None else Eval_df.loc[train_data.reset_index()['subject'][1], num_params][model_name + '_' + 'TEST_BIC'] + calculate_bic(num_params, 216 // n_splits, -test_ll)
            Eval_df.loc[train_data.reset_index()['subject'][1], num_params][model_name + '_' + 'TEST_AIC'] = \
                calculate_aic(num_params, -test_ll) if Eval_df.loc[train_data.reset_index()['subject'][1], num_params][model_name + '_' + 'TEST_AIC'] is None else Eval_df.loc[train_data.reset_index()['subject'][1], num_params][model_name + '_' + 'TEST_AIC'] + calculate_aic(num_params, -test_ll)

            # TODO: Write it to the file at every step

        return make_eval_subj

    # Perform cross-validation
    cv_splits = get_cross_validation_splits(df, n_splits)

    for init in eval_inits:
        create_eval_cols = make_eval_func(eval_model, init)

        for train_data, test_data in cv_splits:
            grouped_train_data = train_data.groupby('subject')
            grouped_test_data = test_data.groupby('subject')
            for subj in grouped_train_data.groups:
                train_subject_data = grouped_train_data.get_group(subj)
                test_subject_data = grouped_test_data.get_group(subj)
                create_eval_cols(train_subject_data, test_subject_data)

    # TODO: file for every subject for every parameter number
    # TODO: maybe start at optimization of the previous optimized parameters.


