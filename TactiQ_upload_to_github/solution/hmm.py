"""Small, dependency-free HMM helpers (numpy only).

viterbi()           MAP state path for given per-frame log-emissions and a log-transition matrix.
fit_gaussian_hmm()  unsupervised Baum-Welch for a diagonal-Gaussian HMM whose features may be NaN
                    (a NaN feature simply contributes nothing to that frame's likelihood).
"""

import numpy as np


def viterbi(log_emit, log_trans, log_start=None):
    """log_emit (T, K), log_trans (K, K) [from, to]. Returns int path of length T."""
    T, K = log_emit.shape
    if log_start is None:
        log_start = np.full(K, -np.log(K))
    back = np.empty((T, K), dtype=np.int8)
    score = log_start + log_emit[0]
    for t in range(1, T):
        cand = score[:, None] + log_trans          # (from, to)
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], np.arange(K)] + log_emit[t]
    path = np.empty(T, dtype=np.int64)
    path[-1] = int(np.argmax(score))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def gaussian_log_emissions(X, means, variances):
    """X (T, F) with NaNs allowed; means/variances (K, F). Returns (T, K)."""
    T, F = X.shape
    K = means.shape[0]
    out = np.zeros((T, K))
    present = ~np.isnan(X)
    Xz = np.where(present, X, 0.0)
    for k in range(K):
        ll = -0.5 * (np.log(2 * np.pi * variances[k]) + (Xz - means[k]) ** 2 / variances[k])
        out[:, k] = np.sum(np.where(present, ll, 0.0), axis=1)
    return out


def _forward_backward(log_emit, log_trans, log_start):
    T, K = log_emit.shape
    trans = np.exp(log_trans)
    alpha = np.empty((T, K))
    scale = np.empty(T)
    em_max = log_emit.max(axis=1, keepdims=True)
    emit = np.exp(log_emit - em_max)
    a = np.exp(log_start) * emit[0]
    scale[0] = a.sum()
    alpha[0] = a / scale[0]
    for t in range(1, T):
        a = (alpha[t - 1] @ trans) * emit[t]
        scale[t] = a.sum()
        alpha[t] = a / scale[t]
    beta = np.empty((T, K))
    beta[-1] = 1.0
    for t in range(T - 2, -1, -1):
        b = trans @ (emit[t + 1] * beta[t + 1])
        beta[t] = b / scale[t + 1]
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    xi = np.zeros((K, K))
    for t in range(T - 1):
        m = alpha[t][:, None] * trans * (emit[t + 1] * beta[t + 1])[None, :]
        xi += m / m.sum()
    loglik = np.sum(np.log(scale)) + np.sum(em_max)
    return gamma, xi, loglik


def fit_gaussian_hmm(X, init_means, init_vars, p_stay=0.998, n_iter=15, fit_trans=False,
                     var_floor=1e-3):
    """Baum-Welch for K diagonal Gaussians. Returns (means, variances, log_trans)."""
    means = np.array(init_means, dtype=float)
    variances = np.array(init_vars, dtype=float)
    K = means.shape[0]
    trans = np.full((K, K), (1 - p_stay) / (K - 1))
    np.fill_diagonal(trans, p_stay)
    log_start = np.full(K, -np.log(K))
    present = ~np.isnan(X)
    Xz = np.where(present, X, 0.0)
    prev = None
    for _ in range(n_iter):
        log_emit = gaussian_log_emissions(X, means, variances)
        gamma, xi, ll = _forward_backward(log_emit, np.log(trans), log_start)
        for k in range(K):
            w = gamma[:, k:k + 1] * present
            denom = w.sum(axis=0) + 1e-9
            means[k] = (w * Xz).sum(axis=0) / denom
            variances[k] = np.maximum((w * (Xz - means[k]) ** 2).sum(axis=0) / denom, var_floor)
        if fit_trans:
            trans = xi / xi.sum(axis=1, keepdims=True)
        if prev is not None and abs(ll - prev) < 1e-4 * abs(prev):
            break
        prev = ll
    return means, variances, np.log(trans)
