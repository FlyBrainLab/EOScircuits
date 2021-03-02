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


#define  G_MIN		0.0
#define  G_MAX		50000.0

struct States {
    double s;
    double u;
    double g;
};

struct Derivatives {
    double s;
    double u;
};


__device__ void clip(States &states)
{
    states.g = fmax(states.g, G_MIN);
    states.g = fmin(states.g, G_MAX);
}

__device__ void forward(
    States &states,
    Derivatives &gstates,
    double dt
)
{
    states.s += dt * gstates.s;
    states.u += dt * gstates.u;
}

__device__ int ode(
    States &states,
    Derivatives &gstates,
    double AD,
    double AR,
    double GMAX,
    double &spike
)
{

    gstates.s = states.u;
    gstates.u = (((-(AR + AD)) * states.u) - ((AR * AD) * states.s));
    if (spike) {
        states.u = (states.u + (AR * AD));
    }
    states.g = (states.s * GMAX);
    return 0;
}



__global__ void run_step (
    int num_thread,
    double dt,
    double *g_state_s,
    double *g_state_u,
    double *g_state_g,
    double *g_param_ad,
    double *g_param_ar,
    double *g_param_gmax,
    double *g_input_spike,
    double *g_output_g
)
{
    /* TODO: option for 1-D or 2-D */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int nid = tid; nid < num_thread; nid += total_threads) {

        States states;
        Derivatives gstates;

        /* import data */
        states.s = g_state_s[nid];
        states.u = g_state_u[nid];
        states.g = g_state_g[nid];
        double param_AD = g_param_ad[nid];
        double param_AR = g_param_ar[nid];
        double param_GMAX = g_param_gmax[nid];
        double input_spike = g_input_spike[nid];

        
        
        /* compute gradient */
        ode(states, gstates, param_AD, param_AR, param_GMAX, input_spike);

        /* solve ode */
        forward(states, gstates, dt);

        /* clip */
        clip(states);

        

        /* export state (internals) data */
        g_state_s[nid] = states.s;
        g_state_u[nid] = states.u;
        g_state_g[nid] = states.g;

        /* export output (updates) data */
        g_output_g[nid] = states.g;
    }

    return;
}


"""


class Alpha(NDComponent):
    """Alpha

    Attributes:
        accesses (list): list of input variables
        updates (list): list of output variables
        params (list): list of parameters
        params_default (dict): default values of the parameters
        internals (OrderedDict): internal variables of the model and initial value
        time_scale (float): scaling factor of the `dt`
    """

    accesses = [
        "spike_state",
    ]
    updates = [
        "g",
    ]
    params = [
        "ad",
        "ar",
        "gmax",
    ]
    params_default = dict(
        ar=12.5,
        ad=12.19,
        gmax=0.1,
    )
    internals = OrderedDict(
        [
            ("s", 0.0),
            ("u", 0.0),
            ("g", 0.0),
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
