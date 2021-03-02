"""Base Circuit Class for EOS Circuits"""
import copy
from warnings import warn
from abc import abstractclassmethod, abstractproperty
import typing as tp
import networkx as nx
import numpy as np
from dataclasses import dataclass, field


class EOSCircuitException(Exception):
    """Base EOS Circuit Exception"""

    pass

@dataclass
class Config:
    """A Base Dataclass of EOS Configuration

    :py:class:`Config` objects are full specifications of the executable circuit modules in
    the :py:mod:`EOSCircuits` module. It is consumed by the
    :py:meth:`Circuit.create_from_config` function to create instances of
    EOS Executable Circuits.

    Among others, :py:class:`Config` objects provide 3 types attributes:

    1. Number of Nodes: :code:`N{x}`

        * These are attributes that specify, for example, the number of receptor
          types in the circuit is specified as
          :py:obj:`~eoscircuits.antcircuits.circuit.ANTConfig.NR`

    2. Node Ids:

        * These are attributes that specify the unique node ids for every
          neuron type in the circuit. For example,
          :py:obj:`~eoscircuits.antcircuits.circuit.ANTConfig.osns` defines
          all unique node ids of the osns.

    3. Routing Table: :code:`{source_node_type}_to_{target_node_type}`

        * These are attributes that specify how node types are connected together.
          Routing tables are typically :py:class:`numpy.ndarray` of either
          :py:class:`numpy.ndarray` of :code:`(int, int)` or :code:`(int, int)`,
          where each entry is the integer indices of the source and target nodes in the
          corresponding *Node Ids* list.
        * For example, if the routing table :code:`osn_to_pn[0][1] = (1,2)` means that
          the in the *0-th* receptor type, the *1st* OSN  is connected to the *2nd* PN
          expressing this particular receptor type.
    """
    @abstractproperty
    def node_types(self) -> tp.List[str]:
        """List of Recognized Node Types"""


@dataclass(repr=False)
class Circuit:
    """A Dataclass of Base EOS Circuit"""

    graph: nx.MultiDiGraph
    """Specification of Executable Graph Compatible with NeuroDriver"""

    config: Config
    """Configuration of Circuit. Fully Specifies the Circuit"""

    extra_comps: tp.List["neurokernel.LPU.NDComponents.NDComponent"] = field(
        default_factory=list
    )
    """Extra Components to be aded to NeuroKernel at Run Time"""

    @abstractclassmethod
    def create_from_config(cls, config:Config) -> "eoscircuits.basecircuit.Circuit":
        """class method that creates an instance of circuit from configuration"""

    @classmethod
    def create_graph(cls, config:Config) -> nx.MultiDiGraph:
        """class method that creates an instance of networkx graph from configuration"""

    @abstractproperty
    def inputs(self) -> dict:
        """input variable and uids dictionary"""

    @abstractproperty
    def outputs(self) -> dict:
        """output variable and uids dictionary"""

    def simulate(
        self,
        t: np.ndarray,
        inputs: tp.Any,
        record_var_list: tp.Iterable[tp.Tuple[str, tp.Iterable]] = None,
        sample_interval: int = 1,
    ) -> tp.Tuple["InputProcessor", "OuputProcessor", "LPU"]:
        """Simulate The Circuit
        Update Affinities and Change Circuit Accordingly

        Arguments:
            t: input time array
            inputs: input data

                * if is `BaseInputProcessor` instance, passed to LPU directly
                * if is dictionary, passed to `ArrayInputProcessor` if is compatible
            record_var_list: :code:`[(var, uids)]` list of tuples of variables to be
                recorded during simulation
            sample_interval: interval at which output is recorded

        Returns:
            A tuple with the follow 3 components:

            * fi: :py:class:`~neurokernel.LPU.InputProcerssors.BaseInputProcessor.BaseInputProcessor`
            * fo: :py:class:`~neurokernel.LPU.OutputProcessors.OutputRecorder.OutputRecorder`
            * lpu: :py:class:`~neurokernel.LPU.LPU.LPU`
        """
        from neurokernel.LPU.LPU import LPU
        from neurokernel.LPU.InputProcessors.BaseInputProcessor import (
            BaseInputProcessor,
        )
        from neurokernel.LPU.InputProcessors.ArrayInputProcessor import (
            ArrayInputProcessor,
        )
        from neurokernel.LPU.OutputProcessors.OutputRecorder import OutputRecorder

        dt = t[1] - t[0]
        if isinstance(inputs, BaseInputProcessor):
            fi = [inputs]
        elif isinstance(inputs, (list, tuple, np.ndarray)) and isinstance(
            inputs[0], BaseInputProcessor
        ):
            fi = inputs
        elif isinstance(inputs, dict):
            for data in inputs.values():
                assert "uids" in data
                assert "data" in data
                assert isinstance(data["data"], np.ndarray)
            fi = [ArrayInputProcessor(inputs)]
        else:
            raise ValueError("Input not understood")
        fo = OutputRecorder(record_var_list, sample_interval=sample_interval)
        lpu = LPU(
            dt,
            "obj",
            self.graph,
            device=0,
            id="EOS",
            input_processors=fi,
            output_processors=[fo],
            debug=False,
            manager=False,
            extra_comps=self.extra_comps,
        )
        lpu.run(steps=len(t))
        return fi, fo, lpu

    def update_graph_attributes(
        self,
        data_dict: dict,
        node_predictive: tp.Callable[[nx.classes.reportviews.NodeView], bool] = None,
    ) -> None:
        """Update Attributes of the graph

        Arguments:
            data_dict: a dictionary of {attr: value} to be set for all filtered nodes
            node_predictive: a function that filtering of nodes from
              :py:func:`networkx.nodes(data=True)` call

        Example:
            >>> circuit.update_graph_attributes(
                    {'sigma':1.},
                    node_predictive=lambda key, val: val['class'] == 'NoisyConnorStevens'
                )
        """
        if node_predictive is None:
            node_predictive = lambda node_id, data: True

        node_uids = [
            key for key, val in self.graph.nodes(data=True) if node_predictive(key, val)
        ]
        update_dict = {_id: data_dict for _id in node_uids}
        nx.set_node_attributes(self.graph, update_dict)

    @classmethod
    def add_nodes_to_graph(
        cls,
        G: nx.MultiDiGraph,
        cfg: Config,
        node_type: "Config.node_types",
        ndcomp_clsname: str,
        ndcomp_module: "Module",
    ) -> None:
        """Add Node to Graph

        This method adds nodes to :py:attr:`Circuit.graph` with the appropriate
        ids and attributes based on the :py:attr:`Circuit.Config` specifications.

        Arguments:
            G: The networkx graph that the node should be added to
            cfg: The configuration instance
            node_type: The node types permissible according to the configuration
            ndcomp_clsname: The class name of the NeuroDriver component
            ndcomp_module: The name of the ndcomp_module in the associated *model.py*
              file for each circuit module. This name is used to pull the :code:`params`
              from the correct class from the *model.py* file to add to the graph.
        """
        if node_type not in cfg.node_types:
            raise EOSCircuitException(
                f"Attempting to add node of type {node_type}, "
                f"must be one of {cfg.node_types}"
            )
        node_ids = getattr(cfg, node_type)
        if isinstance(node_ids[0], (list, tuple, np.ndarray)):
            node_ids = sum(node_ids, [])
        try:
            _ndcomp = getattr(ndcomp_module, ndcomp_clsname)
        except AttributeError as e:
            raise EOSCircuitException(f"NDComponent {ndcomp_clsname} not found.") from e
        except Exception as e:
            raise EOSCircuitException(
                f"Unknown error encountered when adding NDComponent {ndcomp_clsname}"
            ) from e

        node_params = copy.deepcopy(_ndcomp.params)
        if not any(
            [hasattr(p, "__len__") for p in cfg.node_params[node_type].values()]
        ):  # only scalar parameter, set param to all nodes
            node_params.update(cfg.node_params[node_type])
            G.add_nodes_from(node_ids, **{"class": ndcomp_clsname}, **node_params)
        else:  # iterable parameter, not supported
            new_params = {
                key: np.full(len(node_ids), val) if np.isscalar(val) else val
                for key, val in cfg.node_params[node_type].items()
            }
            if not all(
                [len(p) == len(node_ids) for p in cfg.node_params[node_type].values()]
            ):
                raise EOSCircuitException(
                    "Some node parameters have length not equal to number of nodes. "
                    f"Please add node_type '{node_type}'' to graph manually."
                )

            warn(
                "Adding nodes with iterable parameter values could lead to "
                "parameters being assigned to the wrong node."
            )

            for p in new_params.keys():  # remove overlapping parameters
                _ = node_params.pop(p)

            for r, _id in enumerate(node_ids):
                G.add_node(
                    _id ** {"class": ndcomp_clsname},
                    **node_params,
                    **{key: new_params[key][r] for key in new_params.keys()},
                )
