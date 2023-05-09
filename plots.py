import matplotlib.pyplot as plt
import numpy as np
import math
from util import make_poly, make_softplus_poly

def plot_poly(pars):
    x = np.linspace(-20, 30, 100)
    plt.plot(x, make_poly(pars)(x))
    plt.axvline(0.0, c='k', ls='--')

def plot_softplus_poly(pars, xmin=0, xmax=60, ylim=60, color='', axis=plt, alpha=0.5):
    x = np.linspace(xmin, xmax, 100)
    axis.plot(x, make_softplus_poly(pars)(x), color, alpha=alpha)
    axis.axvline(0.0, c='k', ls='--')
    plt.ylim(-2, ylim)

def plot_ply(pars, xmin=0, xmax=60, ylim=60, color=''):
    x = np.linspace(xmin, xmax, 100)
    plt.plot(x, make_poly(pars)(x), color)
    plt.axvline(0.0, c='k', ls='--')
    plt.ylim(0, ylim)


def plot_poly_3(pars, sd_):
    x = np.linspace(-20, 30, 100)
    if sd_ == True:
        plt.plot(x, make_softplus_poly(pars)(x), color='red')
    else:
        plt.plot(x, make_poly(pars)(x), color='cyan')
    plt.axvline(0.0, c='k', ls='--')
    plt.ylim(-20, 50)


def plot_all_models(models, model_names, model_inits, data):
    plots = sum([len(elem) for elem in model_inits])
    fig, axes = plt.subplots(math.ceil(plots/4), 4, figsize=(40, 10))

    idx = 0
    for _, (model, model_name, inits) in enumerate(zip(models, model_names, model_inits)):
        # Fit the model

        for init in inits:

            optimized_params = data.groupby('subject').apply(lambda d: model(d['n1'], d['n2'], d['choice'], init))
            # optimized_params = model(n1_data, n2_data, choice_data)

            for params in optimized_params:
                print(params)
                # Plot the fitted sigma as a function of x
                x = np.linspace(-7, 7, 100)
                fitted_sigma = make_poly(params)(x)

                axes[int(idx/4)][int(idx%4)].plot(x, fitted_sigma)
                axes[int(idx/4)][int(idx%4)].axvline(0.0, c='k', ls='--')
                axes[int(idx/4)][int(idx%4)].set_title(model_name)
                axes[int(idx/4)][int(idx%4)].set_xlabel('x')
                axes[int(idx/4)][int(idx%4)].set_ylabel('Fitted sigma')
                axes[int(idx/4)][int(idx%4)].set_ylim(-20, 50)
            idx += 1

    plt.show()