"""NeuroDriver Models and Utilities
"""
from collections import OrderedDict
import numpy as np
import copy
import networkx as nx
import typing as tp
import os
from . import NDComponents as ndcomp

EXTRA_COMPS = [ndcomp.OTP, ndcomp.NoisyConnorStevens]

class OTP:
    """
    Odorant Transduction Process
    """
    states = OrderedDict(
        [("v", 0.0), ("uh", 0.0), ("duh", 0.0), ("x1", 0.0), ("x2", 0.0), ("x3", 0.0)]
    )

    params = dict(
        br=1.0,
        dr=10.0,
        gamma=0.138,
        a1=45.0,
        b1=0.8,
        a2=199.574,
        b2=51.887,
        a3=2.539,
        b3=0.9096,
        kappa=9593.9,
        p=1.0,
        c=0.06546,
        Imax=150.159,
    )
    _ndcomp = ndcomp.OTP


class NoisyConnorStevens:
    """
    Noisy Connor-Stevens Neuron Model

    F-I curve is controlled by `sigma` parameter

    Notes:
        `sigma` value should be scaled by `sqrt(dt)` as `sigma/sqrt(dt)`
        where `sigma` is the standard deviation of the Brownian Motion
    """

    states = dict(n=0.0, m=0.0, h=1.0, a=1.0, b=1.0, v1=-60.0, v2=-60.0, refactory=0.0)

    params = dict(
        ms=-5.3,
        ns=-4.3,
        hs=-12.0,
        gNa=120.0,
        gK=20.0,
        gL=0.3,
        ga=47.7,
        ENa=55.0,
        EK=-72.0,
        EL=-17.0,
        Ea=-75.0,
        sigma=2.05,
        refperiod=1.0,
    )
    _ndcomp = ndcomp.NoisyConnorStevens
