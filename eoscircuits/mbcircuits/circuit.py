# pylint:disable=no-member
"""Mushroom Body Circuit

This module supports:

1. Generating and changing random connectivity patterns between PNs and KCs with
varying degree of fan-in ratio (number of PNs connected to a given KC)
2. Changing the strength of feedback inhibition of the APL neuron
"""
from itertools import product
import copy
import typing as tp
from dataclasses import dataclass, field, asdict
import numpy as np
import networkx as nx
import olftrans as olf
import olftrans.fbl
import olftrans.data
from ..alcircuits.circuit import ALConfig
from ..basecircuit import Circuit, EOSCircuitException

from . import model as NDModel
from . import NDComponents as ndcomp


class MBException(EOSCircuitException):
    """Base Mushroom Body Exception"""


@dataclass
class MBConfig(ALConfig):
    NK: int = None
    NAPL: int = None
    NFanIn: int = 6

    # names
    kcs: tp.Iterable[str] = field(repr=False, default=None)
    apls: tp.Iterable[str] = field(repr=False, default=None)

    # routings
    pn_to_kc: tp.Iterable[tp.Iterable[float]] = field(default=None, repr=False)
    kc_to_apl: tp.Iterable[float] = field(default=None, repr=False)
    apl_to_kc: tp.Iterable[float] = field(default=None, repr=False)

    def __post_init__(self):
        super().__post_init__()
        # set kc names
        self.kcs = self.set_or_assert(
            self.kcs, [f"KC/{k}" for k in range(self.NK)], self.NK
        )
        self.apls = self.set_or_assert(
            self.apls, [f"APL/{a}" for a in range(self.NAPL)], self.NAPL
        )

        self.pn_to_kc = self.set_or_assert_edges(
            self.pn_to_kc, self.default_pn_to_kc(), self.NR
        )
        self.kc_to_apl = self.set_or_assert_edges(
            self.kc_to_apl, self.default_kc_to_apl(), self.NK
        )
        self.apl_to_kc = self.set_or_assert_edges(
            self.apl_to_kc, self.default_apl_to_kc(), self.NAPL
        )

    @property
    def node_types(self):
        return [
            "osn_otps",
            "osn_bsgs",
            "osn_alphas",
            "osn_axts",
            "prelns",
            "postelns",
            "postilns",
            "pns",
            "kcs",
            "kcdends",
            "apls",
        ]

    @property
    def kcdends(self):
        return [f"{n}/DEND" for n in self.kcs]

    def default_pn_to_kc(self, fanin: int = None, seed: int = None) -> np.ndarray:
        """PN to KC Connectivity

        Arguments:
            fanin: number of PNs connected to a single KC
            seed: seed for random number generator
        """
        if fanin is not None:
            if not np.isscalar(fanin):
                raise MBException("Only supports integer value for PN-to-KC Fan-In")
            if fanin <= 0 or fanin > self.NK:
                raise MBException(
                    f"PN-to-KC Fan-In can only be in range [1, {self.NK}], "
                    f"got {fanin}"
                )
            self.NFanIn = fanin
        rng = np.random.RandomState(seed)
        idx = np.arange(np.sum(self.NP))  # index of all PNs from all glomeruli
        pn_kc_mat = np.zeros((self.NK, np.sum(self.NP)))
        for i in range(self.NK):
            rng.shuffle(idx)
            pn_kc_mat[i][idx[: self.NFanIn]] = 1

        pn_indices = [0] + list(np.cumsum(self.NP))  # start-stop of PN in each glom
        tbl = np.empty(self.NR, dtype=np.ndarray)
        for r, (start, stop) in enumerate(zip(pn_indices[:-1], pn_indices[1:])):
            kc_idx, pn_idx = np.nonzero(pn_kc_mat[:, start:stop])
            pn_to_kc_idx = np.array(list(zip(pn_idx, kc_idx)))
            tbl[r] = np.sort(pn_to_kc_idx, axis=0)
        return tbl

    def default_kc_to_apl(self):
        """Create Densely Connected KC to APL routing table"""
        tbl = np.empty(self.NK, dtype=np.ndarray)
        for k in range(self.NK):
            conn = product([0], np.arange(self.NAPL))
            tbl[k] = np.array(list(conn))
        return tbl

    def default_apl_to_kc(self):
        """Create Densely Connected APL to KC routing table"""
        tbl = np.empty(self.NAPL, dtype=np.ndarray)
        for a in range(self.NAPL):
            conn = product([0], np.arange(self.NK))
            tbl[a] = np.array(list(conn))
        return tbl


@dataclass
class MBCircuit(Circuit):
    """Mushroom Body Circuit"""

    config: MBConfig
    extra_comps: tp.List["NDComponent"] = field(
        init=False, default_factory=lambda: NDModel.EXTRA_COMPS
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

        cls.add_nodes_to_graph(G, cfg, "osn_alphas", "Alpha", NDModel)
        cls.add_nodes_to_graph(G, cfg, "osn_axts", "OSNAxt2", NDModel)
        cls.add_nodes_to_graph(G, cfg, "pns", "PN", NDModel)
        cls.add_nodes_to_graph(G, cfg, "prelns", "PreLN", NDModel)
        cls.add_nodes_to_graph(G, cfg, "postelns", "PostLN", NDModel)
        cls.add_nodes_to_graph(G, cfg, "postilns", "PostLN", NDModel)

        cls.add_nodes_to_graph(G, cfg, "kcs", "KC", NDModel)
        cls.add_nodes_to_graph(G, cfg, "kcdends", "KCDend", NDModel)
        cls.add_nodes_to_graph(G, cfg, "apls", "APL", NDModel)

        # connect nodes
        G.add_edges_from(
            zip(sum(cfg.osn_bsgs, []), sum(cfg.osn_alphas, [])), variable="spike_state"
        )
        G.add_edges_from(
            zip(sum(cfg.osn_alphas, []), sum(cfg.osn_axts, [])), variable="g"
        )

        G.add_edges_from(
            cfg.as_node_ids(cfg.osn_to_preln, cfg.osn_alphas, cfg.prelns), variable="g"
        )
        G.add_edges_from(
            cfg.as_node_ids(cfg.osn_to_postiln, cfg.osn_alphas, cfg.postilns),
            variable="g",
        )
        G.add_edges_from(
            cfg.as_node_ids(cfg.osn_to_posteln, cfg.osn_alphas, cfg.postelns),
            variable="g",
        )

        G.add_edges_from(
            cfg.as_node_ids(cfg.preln_to_axt, cfg.prelns, cfg.osn_axts), variable="r"
        )
        G.add_edges_from(
            cfg.as_node_ids(cfg.axt_to_pn, cfg.osn_axts, cfg.pns), variable="I"
        )
        G.add_edges_from(
            cfg.as_node_ids(cfg.postiln_to_pn, cfg.postilns, cfg.pns), variable="I"
        )
        G.add_edges_from(
            cfg.as_node_ids(cfg.posteln_to_pn, cfg.postelns, cfg.pns), variable="I"
        )

        G.add_edges_from(
            cfg.as_node_ids(cfg.pn_to_kc, cfg.pns, cfg.kcdends), variable="r"
        )
        G.add_edges_from(zip(cfg.kcdends, cfg.kcs), variable="I")
        G.add_edges_from(
            cfg.as_node_ids(cfg.kc_to_apl, cfg.kcs, cfg.apls), variable="r"
        )
        G.add_edges_from(
            cfg.as_node_ids(cfg.apl_to_kc, cfg.apls, cfg.kcdends), variable="g"
        )
        return G

    @classmethod
    def create_from_config(cls, cfg) -> "ALCircuit":
        """Create Instance from Config

        Arguments:
            cfg: Config instance that specifies the configuration of the module

        Returns:
            A new ANTCircuit instance
        """
        return cls(graph=cls.create_graph(cfg), config=cfg)

    def change_apl_strength(self, N: float) -> None:
        """Set APL Strength

        Arguments:
            N: The larger the N the weaker the inhibition. N should be in
                the range of [1, inf]. Typical values are around
                :code:`config.NK`
        """
        N = np.clip(N, 1, np.inf)
        nx.set_node_attributes(
            self.graph, {n: {"N": float(N)} for n in self.config.apls}
        )

    def change_pn_to_kc(
        self, routing: np.ndarray = None, fanin: int = None, seed: int = None
    ) -> None:
        if routing is not None:
            if routing.shape != self.config.pn_to_kc.shape:
                raise MBException("PN-to-KC Routing shape mismatch.")
            self.config.pn_to_kc = routing
        else:
            routing = self.config.default_pn_to_kc(fanin=fanin, seed=seed)
            self.config.pn_to_kc = routing
        self.graph.remove_edges_from(
            set(self.graph.in_edges(self.config.kcdends)).intersection(
                set(self.graph.out_edges(sum(self.config.pns, [])))
            )
        )
        self.graph.add_edges_from(
            self.config.as_node_ids(
                self.config.pn_to_kc, self.config.pns, self.config.kcdends
            ),
            variable="r",
        )

    @property
    def inputs(self) -> dict:
        """Output OTP Nodes IDs and the Variables"""
        return {"conc": sum(self.config.osn_otps, [])}

    @property
    def outputs(self) -> dict:
        """Output BSG Nodes IDs and the Variables"""
        return {"r": sum(self.config.kcs, [])}
