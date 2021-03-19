from . import NDComponents as ndComp
from ..antcircuits.model import EXTRA_COMPS, OTP, NoisyConnorStevens
from collections import OrderedDict


EXTRA_COMPS += [
    ndComp.OSNAxt,
    ndComp.OSNAxt2,
    ndComp.PN,
    ndComp.PreLN,
    ndComp.PostLN,
    ndComp.LeakyIAF,
    ndComp.Alpha,
]


class LeakyIAF:
    params = dict(
        vt=-0.025,
        c=1.5,
        vr=-0.07,
        r=0.2,
    )

    _ndcomp = ndComp.LeakyIAF


class OSNAxt:
    params = dict(
        gamma=100.0,
        bias=1.0,
        gain=1.0,
    )
    _ndcomp = ndComp.OSNAxt


class OSNAxt2:
    params = dict(
        bias=1.0,
        gain=1.0,
    )
    _ndcomp = ndComp.OSNAxt2


class PN:
    params = dict(
        threshold=0.0,
    )
    _ndcomp = ndComp.PN


class PreLN:
    params = dict(
        dummy=0.0,
    )
    _ndcomp = ndComp.PreLN


class PostLN:
    params = dict(
        a1=10.066,
        b1=0.110,
        ICR=80.74,
        s1=0.12,
        R=3.0,
        C=0.01,
        L=0.1,
        k=2.0,
        tau=96.61835748792272,
        RC=0.047,
        C1=4.7,
        polarity=1.0,
        a2=0.12,
        s2=0.05,
    )
    _ndcomp = ndComp.PostLN


class Alpha:
    params = dict(
        ar=12.5,
        ad=12.19,
        gmax=0.1,
    )
    _ndcomp = ndComp.Alpha
