# Online learning of a 2d binary logistic regression model p(y=1|x,w) = sigmoid(w'x),
# using the Exponential-family Extended Kalman Filter (EEKF) algorithm
# described in "Online natural gradient as a Kalman filter", Y. Ollivier, 2018.
# https://projecteuclid.org/euclid.ejs/1537257630.

# The latent state corresponds to the current estimate of the regression weights w.
# The observation model has the form
# p(y(t) |  w(t), x(t)) propto Gauss(y(t) | h_t(w(t)), R(t))
# where h_t(w) = sigmoid(w' * x(t)) = p(t) and  R(t) = p(t) * (1-p(t))

# Dependencies:
#     * !pip install git+https://github.com/blackjax-devs/blackjax.git


# Author: Gerardo Durán-Martín (@gerdm)

from itertools import chain
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from blackjax import rmh
from jax import random
from functools import partial
from jax.scipy.optimize import minimize
from sklearn.datasets import make_biclusters
from ..nlds.extended_kalman_filter import ExtendedKalmanFilter
from jax.scipy.stats import norm


def sigmoid(x): return jnp.exp(x) / (1 + jnp.exp(x))
def log_sigmoid(z): return z - jnp.log1p(jnp.exp(z))
def fz(x): return x
def fx(w, x): return sigmoid(w[None, :] @ x)
def Rt(w, x): return (sigmoid(w @ x) * (1 - sigmoid(w @ x)))[None, None]


def plot_posterior_predictive(ax, X, Xspace, Zspace, title, colors, cmap="RdBu_r"):
    ax.contourf(*Xspace, Zspace, cmap=cmap, alpha=0.7, levels=20)
    ax.scatter(*X.T, c=colors, edgecolors="gray", s=80)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()


def inference_loop(rng_key, kernel, initial_state, num_samples):
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


def E_base(w, Phi, y, alpha):
    """
    Base function containing the Energy of a logistic
    regression with. Energy log-joint
    """
    an = Phi @ w
    log_an = log_sigmoid(an)
    log_likelihood_term = y * log_an + (1 - y) * jnp.log(1 - sigmoid(an))
    log_prior_term = alpha * w @ w / 2

    return -log_prior_term + log_likelihood_term.sum()


def mcmc_logistic_posterior_sample(key, Phi, y, alpha=1.0, init_noise=1.0,
                                   n_samples=5_000, burnin=300, sigma_mcmc=0.8):
    """
    Sample from the posterior distribution of the weights
    of a 2d binary logistic regression model p(y=1|x,w) = sigmoid(w'x),
    using the Metropolis-Hastings algorithm. 
    """
    _, ndims = Phi.shape
    key, key_init = random.split(key)
    w0 = random.multivariate_normal(key, jnp.zeros(ndims), jnp.eye(ndims) * init_noise)
    energy = partial(E_base, Phi=Phi, y=y, alpha=alpha)
    initial_state = rmh.new_state(w0, energy)

    mcmc_kernel = rmh.kernel(energy, sigma=jnp.ones(ndims) * sigma_mcmc)
    mcmc_kernel = jax.jit(mcmc_kernel)

    states = inference_loop(key_init, mcmc_kernel, initial_state, n_samples)
    chains = states.position[burnin:, :]
    return chains


def main():
    ## Data generating process
    n_datapoints = 50
    m = 2
    X, rows, _ = make_biclusters((n_datapoints, m), 2,
                                    noise=0.6, random_state=3141,
                                    minval=-4, maxval=4)
    # whether datapoints belong to class 1
    y = rows[0] * 1.0

    Phi = jnp.c_[jnp.ones(n_datapoints)[:, None], X]
    N, M = Phi.shape

    colors = ["black" if el else "white" for el in y]

    # Predictive domain
    xmin, ymin = X.min(axis=0) - 0.1
    xmax, ymax = X.max(axis=0) + 0.1
    step = 0.1
    Xspace = jnp.mgrid[xmin:xmax:step, ymin:ymax:step]
    _, nx, ny = Xspace.shape
    Phispace = jnp.concatenate([jnp.ones((1, nx, ny)), Xspace])

    ### EEKF Approximation
    mu_t = jnp.zeros(M)
    Pt = jnp.eye(M) * 0.0
    P0 = jnp.eye(M) * 2.0

    model = ExtendedKalmanFilter(fz, fx, Pt, Rt)
    (w_eekf, P_eekf), eekf_hist = model.filter(mu_t, y, Phi, P0, return_params=["mean", "cov"])
    w_eekf_hist = eekf_hist["mean"]
    P_eekf_hist = eekf_hist["cov"]

    ### Laplace approximation
    key = random.PRNGKey(314)
    alpha = 2.0
    init_noise = 1.0
    w0 = random.multivariate_normal(key, jnp.zeros(M), jnp.eye(M) * init_noise)

    E = lambda w: -E_base(w, Phi, y, alpha) / len(y)
    res = minimize(E, w0, method="BFGS")
    w_laplace = res.x
    SN = jax.hessian(E)(w_laplace)


    ### MCMC Approximation
    chains = mcmc_logistic_posterior_sample(key, Phi, y, alpha=alpha)
    Z_mcmc = sigmoid(jnp.einsum("mij,sm->sij", Phispace, chains))
    Z_mcmc = Z_mcmc.mean(axis=0)

    ### *** Ploting surface predictive distribution ***
    colors = ["black" if el else "white" for el in y]
    dict_figures = {}
    key = random.PRNGKey(31415)
    nsamples = 5000

    # EEKF surface predictive distribution
    eekf_samples = random.multivariate_normal(key, w_eekf, P_eekf, (nsamples,))
    Z_eekf = sigmoid(jnp.einsum("mij,sm->sij", Phispace, eekf_samples))
    Z_eekf = Z_eekf.mean(axis=0)

    fig_eekf, ax = plt.subplots()
    title = "EEKF  Predictive Distribution"
    plot_posterior_predictive(ax, X, Xspace, Z_eekf, title, colors)
    dict_figures["logistic_regression_surface_eekf"] = fig_eekf

    # Laplace surface predictive distribution
    laplace_samples = random.multivariate_normal(key, w_laplace, SN, (nsamples,))
    Z_laplace = sigmoid(jnp.einsum("mij,sm->sij", Phispace, laplace_samples))
    Z_laplace = Z_laplace.mean(axis=0)

    fig_laplace, ax = plt.subplots()
    title = "Laplace Predictive distribution"
    plot_posterior_predictive(ax, X, Xspace, Z_laplace, title, colors)
    dict_figures["logistic_regression_surface_laplace"] = fig_laplace

    # MCMC surface predictive distribution
    fig_mcmc, ax = plt.subplots()
    title = "MCMC Predictive distribution"
    plot_posterior_predictive(ax, X, Xspace, Z_mcmc, title, colors)
    dict_figures["logistic_regression_surface_mcmc"] = fig_mcmc

    ### Plot EEKF and Laplace training history
    P_eekf_hist_diag = jnp.diagonal(P_eekf_hist, axis1=1, axis2=2)
    P_laplace_diag = jnp.sqrt(jnp.diagonal(SN))
    lcolors = ["black", "tab:blue", "tab:red"]
    elements = w_eekf_hist.T, P_eekf_hist_diag.T, w_laplace, lcolors
    timesteps = jnp.arange(n_datapoints) + 1

    for k, (wk, Pk, wk_laplace, c) in enumerate(zip(*elements)):
        fig_weight_k, ax = plt.subplots()
        ax.errorbar(timesteps, wk, jnp.sqrt(Pk), c=c, label=f"$w_{k}$ online (EEKF)")
        ax.axhline(y=wk_laplace, c=c, linestyle="dotted", label=f"$w_{k}$ batch (Laplace)", linewidth=3)

        ax.set_xlim(1, n_datapoints)
        ax.legend(framealpha=0.7, loc="upper right")
        ax.set_xlabel("number samples")
        ax.set_ylabel("weights")
        plt.tight_layout()
        dict_figures[f"logistic_regression_hist_ekf_w{k}"] = fig_weight_k
    

    # *** Plotting posterior marginals of weights ***
    for i in range(M):
        fig_weights_marginals, ax = plt.subplots()
        mean_eekf, std_eekf = w_eekf[i], jnp.sqrt(P_eekf[i, i])
        mean_laplace, std_laplace = w_laplace[i], jnp.sqrt(SN[i, i])
        mean_mcmc, std_mcmc = chains[:, i].mean(), chains[:, i].std()

        x = jnp.linspace(mean_eekf - 4 * std_eekf, mean_eekf + 4 * std_eekf, 500)
        ax.plot(x, norm.pdf(x, mean_eekf, std_eekf), label="posterior (EEKF)")
        ax.plot(x, norm.pdf(x, mean_laplace, std_laplace), label="posterior (Laplace)", linestyle="dotted")
        ax.plot(x, norm.pdf(x, mean_mcmc, std_mcmc), label="posterior (MCMC)", linestyle="dashed")
        ax.legend()
        ax.set_title(f"Posterior marginals of weights ({i})")
        #dict_figures[f"weights_marginals_w{i}"] = fig_weights_marginals
        dict_figures[f"logistic_regression_weights_marginals_w{i}"] = fig_weights_marginals


    print("MCMC weights")
    w_mcmc = chains.mean(axis=0)
    print(w_mcmc, end="\n"*2)

    print("EEKF weights")
    print(w_eekf, end="\n"*2)

    print("Laplace weights")
    print(w_laplace, end="\n"*2)

    dict_figures["data"] = {
        "X": X,
        "y": y,
        "Xspace": Xspace,
        "Phi": Phi,
        "Phispace": Phispace,
        "w_laplace": w_laplace,
    }

    return dict_figures


if __name__ == "__main__":
    figs = main()
    plt.show()