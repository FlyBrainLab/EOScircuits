# pylint:disable=no-member
import os
from collections import OrderedDict
import numpy as np
import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from neurokernel.LPU.NDComponents.NDComponent import NDComponent

CUDA_SRC = """



struct States {
    double I;
};

struct Derivatives {
    double I;
};



__device__ void forward(
    States &states,
    Derivatives &gstates,
    double dt
)
{
    states.I += dt * gstates.I;
}

__device__ int ode(
    States &states,
    Derivatives &gstates,
    double GAIN,
    double BIAS,
    double GAMMA,
    double &g,
    double &r
)
{

    gstates.I = ((GAMMA * g) - ((GAMMA * (BIAS + (GAIN * r))) * states.I));
    return 0;
}



__global__ void run_step (
    int num_thread,
    double dt,
    double *g_state_I,
    double *g_param_gain,
    double *g_param_bias,
    double *g_param_gamma,
    double *g_input_g,
    double *g_input_r,
    double *g_output_I
)
{
    /* TODO: option for 1-D or 2-D */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int nid = tid; nid < num_thread; nid += total_threads) {

        States states;
        Derivatives gstates;

        /* import data */
        states.I = g_state_I[nid];
        double param_GAIN = g_param_gain[nid];
        double param_BIAS = g_param_bias[nid];
        double param_GAMMA = g_param_gamma[nid];
        double input_g = g_input_g[nid];
        double input_r = g_input_r[nid];

        
        
        /* compute gradient */
        ode(states, gstates, param_GAIN, param_BIAS, param_GAMMA, input_g, input_r);

        /* solve ode */
        forward(states, gstates, dt);

        

        

        /* export state (internals) data */
        g_state_I[nid] = states.I;

        /* export output (updates) data */
        g_output_I[nid] = states.I;
    }

    return;
}


"""


class OSNAxt(NDComponent):
    """OSNAxt

    Attributes:
        accesses (list): list of input variables
        updates (list): list of output variables
        params (list): list of parameters
        params_default (dict): default values of the parameters
        internals (OrderedDict): internal variables of the model and initial value
        time_scale (float): scaling factor of the `dt`
    """

    accesses = [
        "g",
        "r",
    ]
    updates = [
        "I",
    ]
    params = [
        "gain",
        "bias",
        "gamma",
    ]
    params_default = dict(
        gamma=1.0,
        bias=0.0,
        gain=1.0,
    )
    internals = OrderedDict(
        [
            ("I", 0.0),
        ]
    )
    time_scale = 1.0  # scales dt
    _has_rand = False

    def maximum_dt_allowed(self):
        return np.inf

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

        self.dt = dt * self.time_scale
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

        # make all dtypes consistent
        dtypes = {"dt": self.dtype}
        dtypes.update(
            {"state_" + k: self.internal_states[k].dtype for k in self.internals}
        )
        dtypes.update({"param_" + k: self.params_dict[k].dtype for k in self.params})
        dtypes.update(
            {"input_" + k.format(k): self.inputs[k].dtype for k in self.accesses}
        )
        dtypes.update({"output_" + k: self.dtype for k in self.updates})
        self.update_func = self.get_update_func(dtypes)

        if self._has_rand:
            import neurokernel.LPU.utils.curand as curand

            self.randState = curand.curand_setup(
                self.num_comps, np.random.randint(10000)
            )
            dtypes.update({"rand": self.dtype})

    def run_step(self, update_pointers, st=None):
        for k in self.inputs:
            self.sum_in_variable(k, self.inputs[k], st=st)
        args = (
            [self.internal_states[k].gpudata for k in self.internals]
            + [self.params_dict[k].gpudata for k in self.params]
            + [self.inputs[k].gpudata for k in self.accesses]
            + [update_pointers[k] for k in self.updates]
        )
        if self._has_rand:
            args += [self.randState.gpudata]

        self.update_func.prepared_async_call(
            self.update_func.grid,
            self.update_func.block,
            st,
            self.num_comps,
            self.dt,
            *args
        )

    def get_update_func(self, dtypes):
        from pycuda.compiler import SourceModule

        mod = SourceModule(
            CUDA_SRC,
            options=self.compile_options,
            no_extern_c=self._has_rand,
        )
        func = mod.get_function("run_step")
        type_dict = {k: dtype_to_ctype(dtypes[k]) for k in dtypes}

        func.prepare("i" + np.dtype(self.dtype).char + "P" * (len(type_dict) - 1))
        func.block = (256, 1, 1)
        func.grid = (
            min(
                6 * drv.Context.get_device().MULTIPROCESSOR_COUNT,
                (self.num_comps - 1) // 256 + 1,
            ),
            1,
        )
        return func
