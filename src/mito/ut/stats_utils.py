"""
Discretization module.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.special import logsumexp
from bbmix.models import MixtureBinomial


##


def fit_binom(ad, dp, logratio=False, BIC=True):
    """
    Fit a binomial distribution to ad, dp counts using MLE.
    """
    p_mle = np.sum(ad) / np.sum(dp)
    ad_th = dp * p_mle 

    d = {}
    if logratio:
        log_likelihood = np.sum(stats.binom.logpmf(ad, n=dp, p=p_mle))
        log_likelihood_saturated = np.sum(np.log([ stats.binom.pmf(k, n, k/n) for n,k in zip(dp,ad) ]))
        G2_stat = 2 * (log_likelihood_saturated - log_likelihood)
        G2_p_value = stats.chi2.sf(G2_stat, df=dp.size-1)
        d['G2_stat'] = G2_stat
        d['G2_p_value'] = G2_p_value
    elif BIC:
        log_likelihood = np.sum(stats.binom.logpmf(ad, n=dp, p=p_mle))
        d['BIC'] = 1 * np.log(dp.size) - 2 * log_likelihood
        d['L'] = log_likelihood
    
    return ad_th, d


##


def fit_nbinom(ad, dp, logratio=False, BIC=True):
    """
    Fit a negative-binomial distribution to ad, dp counts using MLE.
    """

    def _negbinom_log_likelihood(params, ad):
        """
        Negative log-likelihood function for the negative binomial distribution.
        params: list of [r, p], where r is the number of failures and p is the probability of success.
        ad: observed allele depths.
        dp: observed depths.
        """
        r, p = params
        if r <= 0 or not (0 < p < 1):
            return np.inf 
        log_likelihood = np.sum(stats.nbinom.logpmf(ad, n=r, p=p))
        return -log_likelihood

    mean_ad = ad.mean()
    var_ad = ad.var()
    p_initial = mean_ad / var_ad if var_ad > mean_ad else 0.5
    r_initial = mean_ad * (mean_ad / (var_ad - mean_ad)) if var_ad > mean_ad else 1

    bounds = [(1e-5, None), (1e-5, 1-1e-5)]  # r > 0 and 0 < p < 1
    result = minimize(_negbinom_log_likelihood, x0=[r_initial, p_initial], args=(ad), bounds=bounds)
    r_mle, p_mle = result.x
    np.random.seed(1234)
    ad_th = stats.nbinom.rvs(r_mle, p_mle, size=dp.size)

    d = {}
    if logratio:
        log_likelihood = -_negbinom_log_likelihood([r_mle, p_mle], ad)
        log_likelihood_saturated = np.sum(np.log([stats.nbinom.pmf(k, r_mle, (k+.0000001)/n) for n,k in zip(dp, ad)]))
        G2_stat = 2 * (log_likelihood_saturated - log_likelihood)
        G2_p_value = stats.chi2.sf(G2_stat, df=dp.size-1)
        d['G2_stat'] = G2_stat
        d['G2_p_value'] = G2_p_value
    elif BIC:
        log_likelihood = -_negbinom_log_likelihood([r_mle, p_mle], ad)
        d['BIC'] = 2 * np.log(dp.size) - 2 * log_likelihood
        d['L'] = log_likelihood
    
    return ad_th, d
    

##


def fit_betabinom(ad, dp, logratio=False, BIC=True):

    def _betabinom_log_likelihood(params, ad, dp):
        """
        Negative log-likelihood function for the Beta-Binomial distribution.
        params: list of [alpha, beta], where alpha and beta are the parameters of the Beta distribution.
        ad: observed allele depths (number of successes).
        dp: observed depths (total trials).
        """
        alpha, beta = params
        if alpha <= 0 or beta <= 0:
            return np.inf
        log_likelihood = np.sum(stats.betabinom.logpmf(ad, dp, alpha, beta))
        return -log_likelihood

    # Initial guess for alpha and beta
    mean_ad = ad.mean()
    var_ad = ad.var()
    p_initial = mean_ad / dp.mean()
    alpha_initial = p_initial * ((p_initial * (1 - p_initial)) / (var_ad / dp.mean() - 1) - 1)
    beta_initial = (1 - p_initial) * ((p_initial * (1 - p_initial)) / (var_ad / dp.mean() - 1) - 1)
    alpha_initial = max(alpha_initial, 1e-5)
    beta_initial = max(beta_initial, 1e-5)

    bounds = [(1e-5, None), (1e-5, None)]
    result = minimize(_betabinom_log_likelihood, x0=[alpha_initial, beta_initial], args=(ad, dp), bounds=bounds)
    alpha_mle, beta_mle = result.x
    np.random.seed(1234)
    ad_th = stats.betabinom.rvs(dp.astype(int), alpha_mle, beta_mle, size=dp.size)

    d = {}
    if logratio:
        log_likelihood = -_betabinom_log_likelihood([alpha_mle, beta_mle], ad, dp)
        log_likelihood_saturated = np.sum(np.log([stats.betabinom.pmf(k, n, alpha_mle, beta_mle) for n, k in zip(dp, ad)]))
        G2_stat = 2 * (log_likelihood_saturated - log_likelihood)
        G2_p_value = stats.chi2.sf(G2_stat, df=dp.size - 1)
        d['G2_stat'] = G2_stat
        d['G2_p_value'] = G2_p_value
    elif BIC:
        log_likelihood = -_betabinom_log_likelihood([alpha_mle, beta_mle], ad, dp)
        d['L'] = log_likelihood
        d['BIC'] = 2 * np.log(dp.size) - 2 * log_likelihood
    
    return ad_th, d


##


def fit_mixbinom(ad, dp, logratio=False, BIC=True):

    np.random.seed(1234)
    model = MixtureBinomial(n_components=2, tor=1e-20)
    params = model.fit((ad, dp), max_iters=500, early_stop=True)
    ad_th = model.sample(dp)

    d = {}
    if logratio:
        raise ValueError('Logratio test not implemented yet...')
    if BIC:
        d['L'] = model.log_likelihood_mixture_bin(ad, dp, params)
        d['BIC'] = model.model_scores['BIC']
    
    return ad_th, d


##


def get_posteriors(ad, dp):
    
    np.random.seed(1234)
    model = MixtureBinomial(n_components=2, tor=1e-20)
    model.fit((ad, dp), max_iters=500, early_stop=True)

    # Access the estimated parameters
    ps = model.params[:2]
    pis = model.params[2:]
    idx1 = np.argmax(ps)
    idx0 = 0 if idx1 == 1 else 1
    p1 = ps[idx1]
    p0 = ps[idx0]
    pi1 = pis[idx1]
    pi0 = pis[idx0]
    d = {'p':[p0,p1], 'pi':[pi0,pi1]}

    # Get posterior probabilities
    log_likelihoods = np.zeros((ad.size, 2))
    for k in range(2):
        log_likelihoods[:,k] = model.log_likelihood_binomial(ad, dp, d['p'][k], d['pi'][k])
    log_weighted_likelihoods = log_likelihoods + np.log(d['pi'])
    log_likelihood_sums = logsumexp(log_weighted_likelihoods, axis=1, keepdims=True)
    log_posterior_probs = log_weighted_likelihoods - log_likelihood_sums
    posterior_probs = np.exp(log_posterior_probs)

    return posterior_probs


##


def genotype_mix(
    ad: np.array, 
    dp: np.array, 
    t_prob: float = .7, 
    t_vanilla: float = 0, 
    debug: bool = False, 
    min_AD: int = 1
    ) -> np.array:
    """
    Derive a discrete genotype (1:'MUT', 0:'WT') for each cell, given the 
    AD and DP counts of one of its candidate mitochondrial variants.
    """

    positive_idx = np.where(dp>0)[0]
    posterior_probs = get_posteriors(ad[positive_idx], dp[positive_idx])
    tests = [ 
        (posterior_probs[:,1]>t_prob) & (posterior_probs[:,0]<(1-t_prob)), # & (ad[positive_idx]>=min_AD),  # REMOVE!!
        (posterior_probs[:,1]<(1-t_prob)) & (posterior_probs[:,0]>t_prob) 
    ]
    geno_prob = np.select(tests, [1,0], default=0)
    genotypes = np.zeros(ad.size, dtype=np.int16)
    genotypes[positive_idx] = geno_prob
    
    # Compare to vanilla genotyping (AF>t --> 1, 0 otherwise)
    if debug:
        test = (ad/dp>t_vanilla) & (ad>=min_AD)
        geno_vanilla = np.where(test,1,0)                        
        df = pd.DataFrame({
          'ad':ad[positive_idx], 'dp':dp[positive_idx], 
          'geno_vanilla':geno_vanilla[positive_idx], 
          'geno_prob':geno_prob, 
          'p0':posterior_probs[:,0], 'p1':posterior_probs[:,1]
        })
        print(pd.crosstab(df['geno_vanilla'], df['geno_prob'], dropna=False))
        return df
    else:
        return genotypes


##


def get_posteriors_and_params(ad, dp):
    
    np.random.seed(1234)
    model = MixtureBinomial(n_components=2, tor=1e-20)
    model.fit((ad, dp), max_iters=500, early_stop=True)

    ps = model.params[:2]
    pis = model.params[2:]
    idx1 = np.argmax(ps)
    idx0 = 0 if idx1 == 1 else 1
    p1 = ps[idx1]
    p0 = ps[idx0]
    pi1 = pis[idx1]
    pi0 = pis[idx0]

    return [p0,p1], [pi0,pi1]


##


def simulate_component_data(p, n_trials, n_samples):
    return np.random.binomial(n=n_trials, p=p, size=n_samples)


##


def get_components(ad, dp):

    ps, pis = get_posteriors_and_params(ad, dp)
    p0, p1 = ps
    pi0, pi1 = pis

    n_samples = 10000
    n_trials = int(np.mean(dp))
    n_samples_0 = int(n_samples * pi0)
    n_samples_1 = n_samples - n_samples_0

    x_component_0 = simulate_component_data(p0, n_trials, n_samples_0)
    x_component_1 = simulate_component_data(p1, n_trials, n_samples_1)

    return x_component_0, x_component_1


##