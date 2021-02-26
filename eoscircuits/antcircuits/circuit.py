"""Antenna Circuit

This module supports:

1. Changing the affinity values of each of the odorant-receptor pairs
characterizing the input of the Odorant Transduction Process.
2. Changing parameter values of the Biological Spike Generators (BSGs)
associated with each OSN.
3. Changing the number of OSNs expressing the same Odorant Receptor (OR) type.
"""
import copy
import typing as tp
from dataclasses import dataclass, field
import numpy as np
import networkx as nx
from olftrans.olftrans import estimate_resting_spike_rate, estimate_sigma
from ..basecircuit import Circuit, EOSCircuitException
from . import model as NDModel
from . import NDComponents as ndcomp


class ANTException(EOSCircuitException):
    """Base Antenna Exception"""


@dataclass
class ANTConfig:
    """Configuration for Antenna Circuits"""

    NO: tp.Iterable[tp.Iterable[int]]
    """Number of OSNs per Receptor Type"""

    affs: tp.Iterable[float]
    """Affinity Values"""

    receptors: tp.Iterable[str] = None
    """Name of receptors of length NR"""

    resting: float = None
    """Resting OSN Spike Rates [Hz]"""

    node_params: dict = field(default_factory=lambda: dict(osn_bsgs=dict(sigma=0.0025)))
    """Parameters for each neuron type"""

    osns: tp.Iterable[tp.Iterable[str]] = field(repr=False, default=None)
    """Ids of OSNs for each channel"""

    def __post_init__(self):
        for n in self.node_types:
            if n not in self.node_params:
                self.node_params[n] = dict()

        self.affs = np.asarray(self.affs)

        # set receptor names
        self.receptors = self.set_or_assert(
            self.receptors, [f"{r}" for r in range(self.NR)], self.NR
        )

        # set osn names
        if np.isscalar(self.NO):
            self.NO = np.full((self.NR,), self.NO, dtype=int)
        else:
            if len(self.NO) != self.NR:
                raise ANTException(
                    f"If `NO` is iterable, it has to have length same as affs."
                )
        self.osns = self.set_or_assert(
            self.osns,
            [
                [f"OSN/{_or}/{o}" for o in range(self.NO[r])]
                for r, _or in enumerate(self.receptors)
            ],
            self.NO,
        )

        if self.drs is None:
            self.drs = np.full((self.NR,), 10.0)
        elif np.isscalar(self.drs):
            self.drs = np.full((self.NR,), self.drs)
        else:
            self.drs = np.asarray(self.drs)
            if len(self.drs) != self.NR:
                raise ANTException(
                    "If Dissociation rate (dr) is specified as iterable, "
                    "it needs to have length the same as affs."
                )
        self.node_params["osn_otps"]["br"] = self.drs * self.affs

        if all([v is None for v in [self.resting, self.sigma]]):
            raise ANTException("Resting and Sigma cannot both be None")
        if self.resting is not None:
            self.sigma = estimate_sigma(self.resting)

    def set_or_assert(
        self, var: "Config.Attribute", new_var: "Config.Attribute", N: np.ndarray
    ) -> "Config.Attribute":
        """Set Variable or Check Dimensionality

        If :code:`var` to new_names if None and perform dimensionality checks

        Arguments:
            var: old variable value
            new_var: new variable value
            N: dimensionality for the variable, could be multi-dimensional
        """
        if var is None:
            if hasattr(N, "__len__"):
                assert len(new_var) == len(N)
                assert all([len(v) == n for v, n in zip(new_var, N)])
            var = new_var
        else:
            if hasattr(N, "__len__"):
                assert len(new_var) == len(N)
                assert all([len(v) == n for v, n in zip(var, N)])
            else:
                assert len(var) == N
        return var

    def set_affs(self, new_affs):
        self.affs = new_affs
        self.brs = self.drs * self.affs

    @property
    def node_types(self) -> tp.List[str]:
        return ["osn_otps", "osn_bsgs"]

    @property
    def osn_otps(self):
        return [[f"{name}/OTP" for name in names] for names in self.osns]

    @property
    def osn_bsgs(self):
        return [[f"{name}/BSG" for name in names] for names in self.osns]

    @property
    def NR(self) -> int:
        """Number of Receptors"""
        return len(self.affs)

    @property
    def sigma(self) -> float:
        """Noisy Connor Stevens model Noise Level"""
        return self.node_params["osn_bsgs"]["sigma"]

    @sigma.setter
    def sigma(self, new_sigma) -> float:
        self.node_params["osn_bsgs"]["sigma"] = new_sigma

    @property
    def brs(self) -> float:
        """Binding Rates of the OTPs"""
        if "br" in self.node_params["osn_otps"]:
            return self.node_params["osn_otps"]["br"]
        return None

    @property
    def drs(self) -> float:
        """Binding Rates of the OTPs"""
        if "dr" in self.node_params["osn_otps"]:
            return self.node_params["osn_otps"]["dr"]
        return None

    @drs.setter
    def drs(self, new_drs) -> float:
        new_drs = np.atleast_1d(new_drs)
        if len(new_drs) != self.NR:
            raise ANTException(
                f"dr values length mismatch, expected {self.NR}, " f"got {len(new_drs)}"
            )
        self.node_params["osn_otps"]["dr"] = new_drs


@dataclass(repr=False)
class ANTCircuit(Circuit):
    """Antenna Circuit"""

    config: ANTConfig
    extra_comps: tp.List["NDComponent"] = field(
        init=False, default_factory=lambda: [ndcomp.NoisyConnorStevens, ndcomp.OTP]
    )

    @classmethod
    def create_graph(cls, cfg) -> nx.MultiDiGraph:
        G = nx.MultiDiGraph()
        for r, (_otp_ids, _bsg_ids) in enumerate(zip(cfg.osn_otps, cfg.osn_bsgs)):
            bsg_params = copy.deepcopy(NDModel.NoisyConnorStevens.params)
            bsg_params.update(
                {
                    key: val
                    for key, val in cfg.node_params["osn_bsgs"].items()
                    if not hasattr(val, "__len__")
                }
            )
            otp_params = copy.deepcopy(NDModel.OTP.params)
            otp_params.update({"br": cfg.brs[r], "dr": cfg.drs[r]})
            otp_params.update(
                {
                    key: val
                    for key, val in cfg.node_params["osn_otps"].items()
                    if key not in ["br", "dr"] and not hasattr(val, "__len__")
                }
            )

            for _o_id, _b_id in zip(_otp_ids, _bsg_ids):
                G.add_node(_o_id, **{"class": "OTP"}, **otp_params)
                G.add_node(_b_id, **{"class": "NoisyConnorStevens"}, **bsg_params)
                G.add_edge(_o_id, _b_id, variable="I")
        return G

    @classmethod
    def create_from_config(cls, cfg) -> "ANTCircuit":
        """Create Instance from Config

        Arguments:
            cfg: Config instance that specifies the configuration of the module

        Returns:
            A new ANTCircuit instance
        """
        return cls(graph=cls.create_graph(cfg), config=cfg)

    def set_affinities(self, value, receptors=None) -> None:
        """Set Affinity values.

        .. note::

            Because binding rates are computed from affinities :code:`config.affs`
            and dissociations rates :code:`config.drs`, change affinities
            will have effect of changing binding rates but not dissociation
            rates.
        """
        if receptors is None:
            receptors = list(self.config.receptors)
        else:
            receptors = list(np.atleast_1d(receptors))

        value = np.atleast_1d(value)
        if len(value) != len(receptors):
            raise ANTException(
                f"Attempting to set values of length {len(value)} into "
                f"{len(receptors)} receptors"
            )

        for r in receptors:
            r_idx = list(self.config.receptors).index(r)
            new_aff = value[r_idx]
            self.config.affs[r_idx] = new_aff
            otp_nodes = self.config.osn_otps[r_idx]
            update_dct = {
                n: {"br": self.graph.nodes[n]["dr"] * new_aff} for n in otp_nodes
            }
            nx.set_node_attributes(self.graph, update_dct)

    def set_bsg_params(self, key: str, value: float) -> None:
        """Set parameter value of BSG nodes"""
        if key == "sigma":
            self.config.sigma = value
        update_dict = {n: {key: value} for n in sum(self.config.osn_bsgs, [])}
        nx.set_node_attributes(self.graph, update_dict)

    def set_NO(
        self, NO: tp.Union[int, tp.Iterable[int]], receptor=None, aff_noise_std=0.0
    ) -> None:
        """Change number of OSNs expressing each receptor type"""
        if receptor is None:
            receptor = list(self.config.receptors)
        else:
            receptor = list(np.atleast_1d(receptor))

        if any([r not in self.config.receptors for r in receptor]):
            raise ANTException("Receptors not found in list of names")

        for r in receptor:
            r_idx = list(self.config.receptors).index(r)
            self.config.NO[r_idx] = NO
            self.config.osns[r_idx] = [f"OSN/{r}/{n}" for n in range(NO)]
        self.graph = self.create_graph(self.config)

    def get_node_ids(
        self,
        node_type: "ANTConfig.node_types",
        receptor: tp.Union[str, tp.Iterable[str]] = None,
    ) -> list:
        if receptor is None:
            receptor = self.config.receptors
        else:
            receptor = np.atleast_1d(receptor)

        for r in receptor:
            if r not in self.config.receptors:
                raise ANTException(f"Receptors {r} not found in list of receptor names")

        if node_type not in self.config.node_types:
            raise ANTException(
                f"node_type {node_type} not recognized, "
                f"must be one of {self.config.node_types}"
            )

        node_ids = getattr(self.config, node_type)
        return [node_ids[list(self.config.receptors).index(r)] for r in receptor]

    @property
    def inputs(self) -> dict:
        """Output OTP Nodes IDs and the Variables"""
        return {"conc": sum(self.config.osn_otps, [])}

    @property
    def outputs(self) -> dict:
        """Output BSG Nodes IDs and the Variables"""
        bsg_ids = sum(self.config.osn_bsgs, [])
        return {"V": bsg_ids, "spike_state": bsg_ids}
