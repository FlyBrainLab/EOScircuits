# pylint:disable=no-member
import os
from collections import OrderedDict
import numpy as np
import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from neurokernel.LPU.NDComponents.NDComponent import NDComponent

CURR_DIR = os.path.dirname(os.path.realpath(__file__))


class NoisyConnorStevens(NDComponent):
    accesses = ["I"]  # rate
    updates = ["spike_state", "V"]  # current
    params = [
        "ms",
        "ns",
        "hs",
        "gNa",
        "gK",
        "gL",
        "ga",
        "ENa",
        "EK",
        "EL",
        "Ea",
        "sigma",
        "refperiod",
    ]

    internals = OrderedDict(
        [
            ("n", 0.0),
            ("m", 0.0),
            ("h", 1.0),
            ("a", 1.0),
            ("b", 1.0),
            ("v1", -60.0),
            ("v2", -60.0),
            ("refractory", 0.0),
        ]
    )

    def maximum_dt_allowed(self):
        return 1e-5

    def __init__(
        self,
        params_dict,
        access_buffers,
        dt,
        LPU_id=None,
        debug=False,
        cuda_verbose=False,
    ):
        if cuda_verbose:
            self.compile_options = ["--ptxas-options=-v", "--expt-relaxed-constexpr"]
        else:
            self.compile_options = ["--expt-relaxed-constexpr"]

        self.debug = debug
        self.LPU_id = LPU_id
        self.num_comps = params_dict[self.params[0]].size
        self.dtype = params_dict[self.params[0]].dtype

        self.dt = dt * 1e3
        self.params_dict = params_dict
        self.access_buffers = access_buffers

        self.internal_states = {
            c: garray.zeros(self.num_comps, dtype=self.dtype) + self.internals[c]
            for c in self.internals
        }

        self.inputs = {
            k: garray.empty(self.num_comps, dtype=self.access_buffers[k].dtype)
            for k in self.accesses
        }

        # self.retrieve_buffer_funcs = {}
        # for k in self.accesses:
        #     self.retrieve_buffer_funcs[k] = \
        #         self.get_retrieve_buffer_func(
        #             k, dtype=self.access_buffers[k].dtype)
        dtypes = {"dt": self.dtype}
        dtypes.update(
            {"input_" + k.format(k): self.inputs[k].dtype for k in self.accesses}
        )
        dtypes.update({"param_" + k: self.params_dict[k].dtype for k in self.params})
        dtypes.update(
            {"state_" + k: self.internal_states[k].dtype for k in self.internals}
        )
        dtypes.update({"output_" + k: self.dtype for k in self.updates})
        #        dtypes.update({'output_' + k: self.dtype if k != 'spike_state' else np.int32 for k in self.updates})
        self.update_func = self.get_update_func(dtypes)
        import neurokernel.LPU.utils.curand as curand

        self.randState = curand.curand_setup(self.num_comps, np.random.randint(10000))

    def run_step(self, update_pointers, st=None):
        for k in self.inputs:
            self.sum_in_variable(k, self.inputs[k], st=st)

        self.update_func.prepared_async_call(
            self.update_func.grid,
            self.update_func.block,
            st,
            self.num_comps,
            self.dt,
            self.randState.gpudata,
            *[self.internal_states[k].gpudata for k in self.internals]
            + [self.params_dict[k].gpudata for k in self.params]
            + [self.inputs[k].gpudata for k in self.accesses]
            + [update_pointers[k] for k in self.updates]
        )

    def get_update_template(self, float_type):
        with open(
            os.path.join(os.path.dirname(CURR_DIR), "NK_kernels/NoisyConnorStevens.cu"),
            "r",
        ) as f:
            lines = f.read()
        if not float_type in (np.double, np.float64):
            from warnings import warn

            warn("float_type {} not implemented, default to double".format(float_type))
            float_type = np.double
        return lines

    def get_update_func(self, dtypes):
        from pycuda.compiler import SourceModule

        mod = SourceModule(
            self.get_update_template(self.dtype),
            options=self.compile_options,
            no_extern_c=True,
        )
        func = mod.get_function("NoisyConnorStevens")
        type_dict = {k: dtype_to_ctype(dtypes[k]) for k in dtypes}

        func.prepare("i" + np.dtype(self.dtype).char + "P" + "P" * (len(type_dict) - 1))
        func.block = (256, 1, 1)
        func.grid = (
            min(
                6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT,
                (self.num_comps - 1) // 256 + 1,
            ),
            1,
        )
        return func
