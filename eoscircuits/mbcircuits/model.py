from . import NDComponents as ndComp
from neuroballad.models.element import Element
from collections import OrderedDict
from ..alcircuits.model import (
    EXTRA_COMPS,
    OTP,
    NoisyConnorStevens,
    OSNAxt,
    OSNAxt2,
    PN,
    PreLN,
    PostLN,
    LeakyIAF,
    Alpha,
)

EXTRA_COMPS += [
    ndComp.KC,
    ndComp.KCDend,
    ndComp.APL,
    ndComp.PN2KC,
]


class Model(Element):
    """NeuroBallad Element that also wraps the underlying NDComponent"""

    _ndcomp = None


class KC(Model):
    params = dict(
        threshold=1.0,
    )
    _ndcomp = ndComp.KC


class KCDend(Model):
    params = dict(
        bias=1.0,
        gain=1.0,
    )
    _ndcomp = ndComp.KCDend


class APL(Model):
    params = dict(
        N=1.0,
    )
    _ndcomp = ndComp.APL


class PN2KC(Model):
    params = dict(
        weight=1.0,
    )
    _ndcomp = ndComp.PN2KC
