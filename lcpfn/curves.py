import numpy as np
from collections import OrderedDict

prior = {
    "pow3": {
        "uniform": OrderedDict(
            a={"type": "uniform", "param1": -1, "param2": 1},
            c={"type": "uniform", "param1": 0, "param2": 1},
            alpha={"type": "uniform", "param1": 0, "param2": 1},
        ),
        "peaked": OrderedDict(
            a={"type": "uniform", "param1": -0.6, "param2": 0.6},
            c={"type": "uniform", "param1": 0, "param2": 1.25},
            alpha={"type": "log_normal", "param1": 0, "param2": 2},
        ),
    },
    "ilog2": {
        "uniform": OrderedDict(
            c={"type": "uniform", "param1": 0, "param2": 1},
            a={"type": "uniform", "param1": -1, "param2": 1},
        ),
        "peaked": OrderedDict(
            c={"type": "uniform", "param1": 0, "param2": 1},
            a={"type": "uniform", "param1": -0.5, "param2": 0.5},
        ),
    },
    "janoschek": {
        "uniform": OrderedDict(
            a={"type": "uniform", "param1": 0, "param2": 1},
            beta={"type": "uniform", "param1": 0, "param2": 2},
            k={"type": "uniform", "param1": 0, "param2": 1},
            delta={"type": "uniform", "param1": -5, "param2": 5},
        ),
        "peaked": OrderedDict(
            a={"type": "uniform", "param1": 0, "param2": 1},
            beta={"type": "uniform", "param1": 0, "param2": 2},
            k={"type": "log_normal", "param1": -2, "param2": 1},
            delta={"type": "log_normal", "param1": 0, "param2": 0.5},
        ),
    },
}


def prior_sampler(rng, type, param1, param2):
    if type == "uniform":
        return rng.uniform(param1, param2)
    elif type == "log_normal":
        return rng.lognormal(param1, param2)
    raise Exception("Unknown prior type: {}".format(type))


def pow3(x, c, a, alpha):
    return c - a * (x) ** (-alpha)


def prior_pow3(rng):
    return {
        p: prior_sampler(
            rng,
            prior["pow3"]["peaked"][p]["type"],
            param1=prior["pow3"]["peaked"][p]["param1"],
            param2=prior["pow3"]["peaked"][p]["param2"],
        )
        for p in ["a", "c", "alpha"]
    }


def uniform_prior_pow3(rng):
    return {
        p: prior_sampler(
            rng,
            prior["pow3"]["uniform"][p]["type"],
            param1=prior["pow3"]["uniform"][p]["param1"],
            param2=prior["pow3"]["uniform"][p]["param2"],
        )
        for p in ["a", "c", "alpha"]
    }


def ilog2(x, c, a):
    return c - a / (np.log(x + 1))


def prior_ilog2(rng):
    return {
        p: prior_sampler(
            rng,
            prior["ilog2"]["peaked"][p]["type"],
            param1=prior["ilog2"]["peaked"][p]["param1"],
            param2=prior["ilog2"]["peaked"][p]["param2"],
        )
        for p in ["a", "c"]
    }


def uniform_prior_ilog2(rng):
    return {
        p: prior_sampler(
            rng,
            prior["ilog2"]["uniform"][p]["type"],
            param1=prior["ilog2"]["uniform"][p]["param1"],
            param2=prior["ilog2"]["uniform"][p]["param2"],
        )
        for p in ["a", "c"]
    }


def janoschek(x, a, beta, k, delta):
    """
    http://www.pisces-conservation.com/growthhelp/janoschek.htm
    """
    return a - (a - beta) * np.exp(-k * x**delta)


def prior_janoschek(rng):
    return {
        p: prior_sampler(
            rng,
            prior["janoschek"]["peaked"][p]["type"],
            param1=prior["janoschek"]["peaked"][p]["param1"],
            param2=prior["janoschek"]["peaked"][p]["param2"],
        )
        for p in ["a", "beta", "k", "delta"]
    }


def uniform_prior_janoschek(rng):
    return {
        p: prior_sampler(
            rng,
            prior["janoschek"]["uniform"][p]["type"],
            param1=prior["janoschek"]["uniform"][p]["param1"],
            param2=prior["janoschek"]["uniform"][p]["param2"],
        )
        for p in ["a", "beta", "k", "delta"]
    }


def log_power(x, a, b, c):
    # a: upper bound
    # c: growth rate
    # initial = a/ (1 + (1/e^b)^c
    return a / (1.0 + (x / np.exp(b)) ** c)


def prior_log_power(rng):
    # a ~ N(0.8,0.1)
    # b ~ N(1,1)
    # c ~ U(-3,0)
    a = rng.normal(0.8, 0.1)
    b = rng.normal(1.0, 1.0)
    c = rng.uniform(-3.0, 0.0)
    return {"a": a, "b": b, "c": c}


def weibull(x, alpha, beta, kappa, delta):
    """
    Weibull modell
    http://www.pisces-conservation.com/growthhelp/index.html?morgan_mercer_floden.htm
    alpha: upper asymptote
    beta: lower asymptote
    k: growth rate
    delta: controls the x-ordinate for the point of inflection
    """
    return alpha - (alpha - beta) * np.exp(-((kappa * x) ** delta))


def prior_weibull(rng):
    alpha = rng.uniform(0.0, 1.5)
    beta = rng.uniform(0.0, 1)
    kappa = np.exp(rng.normal(-2.0, 1.0))
    delta = np.exp(rng.normal(0, 0.5))
    return {"alpha": alpha, "beta": beta, "kappa": kappa, "delta": delta}


def mmf(x, alpha, beta, kappa, delta):
    """
    Morgan-Mercer-Flodin
    description:
    Nonlinear Regression page 342
    http://bit.ly/1jodG17
    http://www.pisces-conservation.com/growthhelp/index.html?morgan_mercer_floden.htm
    alpha: upper asymptote
    kappa: growth rate
    beta: initial value
    delta: controls the point of inflection
    """
    return alpha - (alpha - beta) / (1.0 + (kappa * x) ** delta)


def prior_mmf(rng):
    # alpha ~ N(0.8,0.1)
    # beta ~ N(0.2,0.1)
    # ln(kappa) ~ N(0,2)
    # ln(delta) ~ N(0,1)
    alpha = rng.normal(0.8, 0.1)
    beta = rng.normal(0.2, 0.1)
    kappa = np.exp(rng.normal(0, 2))
    delta = np.exp(rng.normal(0, 1))
    return {"alpha": alpha, "beta": beta, "kappa": kappa, "delta": delta}


def vap(x, a, b, c):
    """Vapor pressure model"""
    # no upper bound if c > 0
    # a = ln(upper bound) for c=0
    # a+b = ln(initial)
    return np.exp(a + b / x + c * np.log(x))


def prior_vap(rng):
    a = rng.uniform(-2.0, 0.0)  # @heri: range check
    b = rng.uniform(-4.0, 0.0)  # @heri: range check
    c = np.exp(rng.uniform(-8.0, 0.0))  # @heri: same as weights
    return {"a": a, "b": b, "c": c}


def loglog_linear(x, a, b):
    x = np.log(x)
    return np.log(a * x + b)


def prior_loglog_linear(rng):
    # ln(a) ~ N(-2, 1)
    # ln(b) ~ U(0, 1)
    a = np.exp(rng.normal(-2.0, 1.0))
    b = np.exp(rng.uniform(0.0, 1.0))
    return {"a": a, "b": b}


def exp4(x, c, a, b, alpha):
    return c - np.exp(-a * (x**alpha) + b)


def prior_exp4(rng):
    # c ~ N(0.8,0.1)
    c = rng.normal(0.8, 0.1)
    # ln(a) ~ N(-2,1)
    a = np.exp(rng.normal(-2, 1))
    # ln(alpha) ~ N(0,1)
    alpha = np.exp(rng.normal(0, 1))
    # ln(b) ~ N(0,0.5)
    b = np.exp(rng.normal(0, 0.5))
    return {"a": a, "b": b, "c": c, "alpha": alpha}


def pow4(x, c, a, b, alpha):
    return c - (a * x + b) ** -alpha


def prior_pow4(rng):
    # ln(1 - c) ~ U(-5, 0)
    c = 1 - np.exp(rng.uniform(-5.0, 0))
    # ln(a) ~ N(-3, 2)
    a = np.exp(rng.normal(-3.0, 2))
    # ln(alpha) ~ N(0,1)
    alpha = np.exp(rng.normal(0, 1))
    # ln(b) ~ U(0, 1)
    b = np.exp(rng.uniform(0, 1))
    return {"a": a, "b": b, "c": c, "alpha": alpha}


def dr_hill_zero_background(x, theta, eta, kappa):
    # theta: upper bound
    # eta: growth rate
    # initial = theta/(kappa^eta + 1)
    return (theta * x**eta) / (kappa**eta + x**eta)


def prior_dr_hill_zero_background(rng):
    # theta ~ U(1,0) N(0.8,0.1)
    # ln(eta) ~ N(1,1)
    # ln(kappa) ~ N(1,2)
    theta = rng.normal(0.8, 0.1)
    eta = np.exp(rng.normal(1.0, 1.0))
    kappa = np.exp(rng.normal(1.0, 2.0))
    return {"theta": theta, "eta": eta, "kappa": kappa}
