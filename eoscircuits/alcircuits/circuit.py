# pylint:disable=unsupported-membership-test
# pylint:disable=unsubscriptable-object
# pylint:disable=unsupported-assignment-operation
"""Antennal Lobe Circuit

This module supports:

1. Changing parameter values of Biological Spike Generators (BSGs) associated with each
of the Local and Projection Neurons,
2. Changing the number and connectivity of Projection Neurons innervating a given AL
Glomerulus,
3. Changing the number and connectivity of Local Neurons in the Predictive Coding and
ON-OFF circuits of the AL
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
from ..basecircuit import Circuit

from . import model as NDModel
from . import NDComponents as ndcomp
from ..antcircuits.circuit import ANTConfig


class ALException(Exception):
    """Base Antennal Lobe Exception"""


@dataclass
class ALConfig(ANTConfig):
    # numbers
    NP: tp.Union[int, tp.Iterable[int]] = None
    """Number of PNs, organized by Receptor Type"""

    NPreLN: int = None
    """Number of Pre-synaptic Local Neurons"""

    NPosteLN: tp.Union[int, tp.Iterable[int]] = None
    """Number of Post-synaptic Excitatory Local Neurons, organized by Receptor Type"""

    NPostiLN: tp.Union[int, tp.Iterable[int]] = None
    """Number of Post-synaptic Inhibitory Local Neurons, organized by Receptor Type"""

    # names
    prelns: tp.Iterable[str] = field(repr=False, default=None)
    postelns: tp.Iterable[tp.Iterable[str]] = field(repr=False, default=None)
    postilns: tp.Iterable[tp.Iterable[str]] = field(repr=False, default=None)
    pns: tp.Iterable[tp.Iterable[str]] = field(repr=False, default=None)

    # routings
    osn_to_preln: tp.Iterable[tp.Iterable[float]] = field(default=None, repr=False)
    osn_to_postiln: tp.Iterable[tp.Iterable[float]] = field(default=None, repr=False)
    osn_to_posteln: tp.Iterable[tp.Iterable[float]] = field(default=None, repr=False)
    preln_to_axt: tp.Iterable[tp.Iterable[float]] = field(default=None, repr=False)
    axt_to_pn: tp.Iterable[tp.Iterable[float]] = field(default=None, repr=False)
    postiln_to_pn: tp.Iterable[tp.Iterable[float]] = field(default=None, repr=False)
    posteln_to_pn: tp.Iterable[tp.Iterable[float]] = field(default=None, repr=False)

    def __post_init__(self):
        """Set Variable Names and Default Routing Tables"""
        super().__post_init__()
        self.node_params["postelns"]["polarity"] = 1.0
        self.node_params["postilns"]["polarity"] = -1.0
        # set pn names
        if np.isscalar(self.NP):
            self.NP = np.full(self.NR, self.NP)
        else:
            assert len(self.NP) == self.NR
        self.pns = self.set_or_assert(
            self.pns,
            [
                [f"PN/{_or}/{p}" for p in range(self.NP[r])]
                for r, _or in enumerate(self.receptors)
            ],
            self.NP,
        )

        # set prelns names
        self.prelns = self.set_or_assert(
            self.prelns, [f"PreLN/{r}" for r in range(self.NPreLN)], self.NPreLN
        )

        # set posteln names
        if np.isscalar(self.NPosteLN):
            self.NPosteLN = np.full(self.NR, self.NPosteLN)
        else:
            assert len(self.NPosteLN) == self.NR
        self.postelns = self.set_or_assert(
            self.postelns,
            [
                [f"PostLN/e/{_or}/{p}" for p in range(self.NPosteLN[r])]
                for r, _or in enumerate(self.receptors)
            ],
            self.NPosteLN,
        )

        # set postiln names
        if np.isscalar(self.NPostiLN):
            self.NPostiLN = np.full(self.NR, self.NPostiLN)
        else:
            assert len(self.NPostiLN) == self.NR
        self.postilns = self.set_or_assert(
            self.postilns,
            [
                [f"PostLN/i/{_or}/{p}" for p in range(self.NPostiLN[r])]
                for r, _or in enumerate(self.receptors)
            ],
            self.NPostiLN,
        )

        self.osn_to_preln = self.set_or_assert_edges(
            self.osn_to_preln, self.default_osn_to_preln(), self.NR
        )
        self.osn_to_postiln = self.set_or_assert_edges(
            self.osn_to_postiln, self.default_osn_to_postiln(), self.NR
        )
        self.osn_to_posteln = self.set_or_assert_edges(
            self.osn_to_posteln, self.default_osn_to_posteln(), self.NR
        )
        self.preln_to_axt = self.set_or_assert_edges(
            self.preln_to_axt, self.default_preln_to_axt(), self.NR
        )
        self.axt_to_pn = self.set_or_assert_edges(
            self.axt_to_pn, self.default_axt_to_pn(), self.NR
        )
        self.postiln_to_pn = self.set_or_assert_edges(
            self.postiln_to_pn, self.default_postiln_to_pn(), self.NR
        )
        self.posteln_to_pn = self.set_or_assert_edges(
            self.posteln_to_pn, self.default_posteln_to_pn(), self.NR
        )

    @property
    def node_types(self) -> tp.List[str]:
        """List of Recognized Node Types"""
        return [
            "osn_otps",
            "osn_bsgs",
            "osn_alphas",
            "osn_axts",
            "prelns",
            "postelns",
            "postilns",
            "pns",
        ]

    @property
    def routing_tables(self) -> tp.List[str]:
        """List of Recognized Routing Tables"""
        return [
            "osn_to_preln",
            "osn_to_postiln",
            "osn_to_posteln",
            "preln_to_axt",
            "axt_to_pn",
            "postiln_to_pn",
            "posteln_to_pn",
        ]

    @property
    def osn_otps(self):
        return [[f"{name}/OTP" for name in names] for names in self.osns]

    @property
    def osn_bsgs(self):
        return [[f"{name}/BSG" for name in names] for names in self.osns]

    @property
    def osn_alphas(self):
        return [[f"{name}/ALP" for name in names] for names in self.osns]

    @property
    def osn_axts(self):
        return [[f"{name}/AXT" for name in names] for names in self.osns]

    def as_node_ids(self, table, source, target) -> tp.List[tp.List[str]]:
        """Convert Routing Table's indices to node Ids

        Arguments:
            table: routing table
            source: source node ids
            target: target node ids

        Returns:
            A flattenend list of all [source, target] node ids
        """
        uids = []
        for r, tab in enumerate(table):
            if tab is None:
                continue
            try:
                source_id = np.asarray(source)[r][tab[:, 0]]
            except:
                source_id = np.asarray(source)[tab[:, 0]]
            try:
                target_id = np.asarray(target)[r][tab[:, 1]]
            except:
                target_id = np.asarray(target)[tab[:, 1]]
            uids += list(zip(source_id, target_id))
        return uids

    def set_or_assert_edges(self, array, new_array, size):
        if array is None:
            assert len(new_array) == size
            array = new_array
        else:
            assert len(array) == size
        return array

    def default_osn_to_preln(self):
        tbl = np.empty(self.NR, dtype=np.ndarray)
        for r in range(self.NR):
            conn = product(np.arange(self.NO[r]), np.arange(self.NPreLN))
            tbl[r] = np.array(list(conn))
        return tbl

    def default_osn_to_posteln(self):
        tbl = np.empty(self.NR, dtype=np.ndarray)
        for r, (no, npln) in enumerate(zip(self.NO, self.NPosteLN)):
            conn = product(np.arange(no), np.arange(npln))
            tbl[r] = np.array(list(conn))
        return tbl

    def default_osn_to_postiln(self):
        tbl = np.empty(self.NR, dtype=np.ndarray)
        for r, (no, npln) in enumerate(zip(self.NO, self.NPostiLN)):
            conn = product(np.arange(no), np.arange(npln))
            tbl[r] = np.array(list(conn))
        return tbl

    def default_preln_to_axt(self):
        tbl = np.empty(self.NR, dtype=np.ndarray)
        for r in range(self.NR):
            conn = product(np.arange(self.NPreLN), np.arange(self.NO[r]))
            tbl[r] = np.array(list(conn))
        return tbl

    def default_postiln_to_pn(self):
        tbl = np.empty(self.NR, dtype=np.ndarray)
        for r in range(self.NR):
            conn = product(np.arange(self.NPostiLN[r]), np.arange(self.NP[r]))
            tbl[r] = np.array(list(conn))
        return tbl

    def default_posteln_to_pn(self):
        tbl = np.empty(self.NR, dtype=np.ndarray)
        for r in range(self.NR):
            conn = product(np.arange(self.NPosteLN[r]), np.arange(self.NP[r]))
            tbl[r] = np.array(list(conn))
        return tbl

    def default_axt_to_pn(self):
        tbl = np.empty(self.NR, dtype=np.ndarray)
        for r in range(self.NR):
            conn = product(np.arange(self.NO[r]), np.arange(self.NP[r]))
            tbl[r] = np.array(list(conn))
        return tbl


@dataclass(repr=False)
class ALCircuit(Circuit):
    """Antennal Lobe Circuit"""

    config: ALConfig
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

    def set_node_params(
        self,
        node_type: "ALConfig.node_types",
        key: str,
        value: float,
        receptor: tp.Union[str, tp.Iterable[str]] = None,
    ):
        """Set Parameter Value of selected Nodes"""
        node_ids = self.get_node_ids(node_type=node_type, receptor=receptor)
        if key == "sigma":
            self.config.sigma = value
        update_dct = {n: {key: value} for n in sum(node_ids, [])}
        nx.set_node_attributes(self.graph, update_dct)

    def set_neuron_number(
        self,
        node_type: "ALConfig.node_types",
        number: int,
        receptor: tp.Union[str, tp.Iterable[str]] = None,
    ) -> "ALCircuit":
        """Set Number of Neurons and change Routing Table Appropriately"""
        if node_type not in self.config.node_types:
            raise ALException(
                f"Node Type {node_type} not found in graph"
                f"must be one of {self.config.node_types}"
            )
        cfg = asdict(self.config)

        def _set_PN(number):
            _ = cfg.pop("axt_to_pn")
            _ = cfg.pop("postiln_to_pn")
            _ = cfg.pop("posteln_to_pn")
            _ = cfg.pop("pns")
            cfg.update(dict(NP=number))
            return self.create_from_config(ALConfig(**cfg))

        def _set_PreLN(number):
            _ = cfg.pop("osn_to_preln")
            _ = cfg.pop("preln_to_axt")
            _ = cfg.pop("prelns")
            cfg.update(dict(NPreLN=number))
            return self.create_from_config(ALConfig(**cfg))

        def _set_PosteLN(number):
            _ = cfg.pop("osn_to_posteln")
            _ = cfg.pop("posteln_to_pn")
            _ = cfg.pop("postelns")
            cfg.update(dict(NPreLN=number))
            return self.create_from_config(ALConfig(**cfg))

        def _set_PostiLN(number):
            _ = cfg.pop("osn_to_postiln")
            _ = cfg.pop("postiln_to_pn")
            _ = cfg.pop("postilns")
            cfg.update(dict(NPreLN=number))
            return self.create_from_config(ALConfig(**cfg))

        def _set_OSN(number):
            _ = cfg.pop("osn_to_preln")
            _ = cfg.pop("osns")

            _ = cfg.pop("osn_to_postiln")
            _ = cfg.pop("osn_to_posteln")
            _ = cfg.pop("preln_to_axt")
            _ = cfg.pop("axt_to_pn")
            _ = cfg.update(dict(NPreLN=number))
            return self.create_from_config(ALConfig(**cfg))

        if "osn" in node_type:
            return _set_OSN(number)
        if "pn" in node_type:
            return _set_PN(number)
        if "posteln" in node_type:
            return _set_PosteLN(number)
        if "postiln" in node_type:
            return _set_PosteLN(number)
        if "preln" in node_type:
            return _set_PreLN(number)

    def set_routing(
        self,
        table: np.ndarray,
        name: str,
        receptor: tp.Union[str, tp.Iterable[str]] = None,
    ):
        """Seting Routing Table in Antennal Lobe"""
        if not name in self.config.routing_tables:
            raise ALException(
                f"Attempting to set table {name}, "
                f"Must be one of {self.config.routing_tables}"
            )

        if receptor is not None:
            receptor = np.atleast_1d(receptor)
        else:
            receptor = self.config.receptors

        if len(table) != len(receptor):
            raise ALException(
                "Table must be of shape "
                "(len(receptor),) with each entry being the routing "
                "in that particular channel"
            )

        cfg = asdict(self.config)
        update_table = cfg[name]
        for n, r in enumerate(receptor):
            r_idx = list(self.config.receptors).index(r)
            update_table[r_idx] = table[n]
        cfg.update({name: update_table})
        return self.create_from_config(ALConfig(**cfg))

    @property
    def inputs(self) -> dict:
        """Output OTP Nodes IDs and the Variables

        Returns:
            OTPs with input variable `conc`
        """
        return {"conc": sum(self.config.osn_otps, [])}

    @property
    def outputs(self) -> dict:
        """Output BSG Nodes IDs and the Variables

        Returns:
            PNs with output variable `r`
        """
        return {"r": sum(self.config.pns, [])}

    def get_node_ids(
        self,
        node_type: "ALConfig.node_types",
        receptor: tp.Union[str, tp.Iterable[str]] = None,
    ) -> list:
        if receptor is None:
            receptor = self.config.receptors
        else:
            receptor = np.atleast_1d(receptor)

        for r in receptor:
            if r not in self.config.receptors:
                raise ALException(f"Receptors {r} not found in list of receptor names")

        if node_type not in self.config.node_types:
            raise ALException(
                f"node_type {node_type} not recognized, "
                f"must be one of {self.config.node_types}"
            )

        node_ids = getattr(self.config, node_type)
        return [node_ids[list(self.config.receptors).index(r)] for r in receptor]
