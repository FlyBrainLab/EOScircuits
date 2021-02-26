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


#define  X1_MIN		0.0
#define  X1_MAX		1.0

struct States {
    double x1;
    double v;
    double i;
    double x2;
    double y1;
    double I;
};

struct Derivatives {
    double x1;
    double v;
    double i;
    double x2;
};


__device__ void clip(States &states)
{
    states.x1 = fmax(states.x1, X1_MIN);
    states.x1 = fmin(states.x1, X1_MAX);
}

__device__ void forward(
    States &states,
    Derivatives &gstates,
    double dt
)
{
    states.x1 += dt * gstates.x1;
    states.v += dt * gstates.v;
    states.i += dt * gstates.i;
    states.x2 += dt * gstates.x2;
}

__device__ int ode(
    States &states,
    Derivatives &gstates,
    double K,
    double TAU,
    double A2,
    double S2,
    double L,
    double RC,
    double ICR,
    double C1,
    double A1,
    double POLARITY,
    double R,
    double B1,
    double C,
    double S1,
    double &g
)
{
    double X1;
    double Z;

    gstates.x1 = (((B1 * g) * (1.0 - states.x1)) - (A1 * states.x1));
    X1 = (ICR * states.x1);
    gstates.v = ((states.i / C) + (((S1 * X1) - states.v) / (R * C)));
    gstates.i = (((S1 * X1) - states.v) / L);
    Z = (states.i * K);
    gstates.x2 = (((-states.x2) / RC) + (states.y1 / C1));
    states.y1 = (S2 * (exp((TAU * ((POLARITY * Z) - states.x2))) - 1));
    states.I = ((states.y1 * POLARITY) * A2);
    return 0;
}



__global__ void run_step (
    int num_thread,
    double dt,
    double *g_state_x1,
    double *g_state_v,
    double *g_state_i,
    double *g_state_x2,
    double *g_state_y1,
    double *g_state_I,
    double *g_param_k,
    double *g_param_tau,
    double *g_param_a2,
    double *g_param_s2,
    double *g_param_L,
    double *g_param_RC,
    double *g_param_ICR,
    double *g_param_C1,
    double *g_param_a1,
    double *g_param_polarity,
    double *g_param_R,
    double *g_param_b1,
    double *g_param_C,
    double *g_param_s1,
    double *g_input_g,
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
        states.x1 = g_state_x1[nid];
        states.v = g_state_v[nid];
        states.i = g_state_i[nid];
        states.x2 = g_state_x2[nid];
        states.y1 = g_state_y1[nid];
        states.I = g_state_I[nid];
        double param_K = g_param_k[nid];
        double param_TAU = g_param_tau[nid];
        double param_A2 = g_param_a2[nid];
        double param_S2 = g_param_s2[nid];
        double param_L = g_param_L[nid];
        double param_RC = g_param_RC[nid];
        double param_ICR = g_param_ICR[nid];
        double param_C1 = g_param_C1[nid];
        double param_A1 = g_param_a1[nid];
        double param_POLARITY = g_param_polarity[nid];
        double param_R = g_param_R[nid];
        double param_B1 = g_param_b1[nid];
        double param_C = g_param_C[nid];
        double param_S1 = g_param_s1[nid];
        double input_g = g_input_g[nid];

        
        
        /* compute gradient */
        ode(states, gstates, param_K, param_TAU, param_A2, param_S2, param_L, param_RC, param_ICR, param_C1, param_A1, param_POLARITY, param_R, param_B1, param_C, param_S1, input_g);

        /* solve ode */
        forward(states, gstates, dt);

        /* clip */
        clip(states);

        

        /* export state (internals) data */
        g_state_x1[nid] = states.x1;
        g_state_v[nid] = states.v;
        g_state_i[nid] = states.i;
        g_state_x2[nid] = states.x2;
        g_state_y1[nid] = states.y1;
        g_state_I[nid] = states.I;

        /* export output (updates) data */
        g_output_I[nid] = states.I;
    }

    return;
}


"""


class PostLN(NDComponent):
    """PostLN

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
    ]
    updates = [
        "I",
    ]
    params = [
        "k",
        "tau",
        "a2",
        "s2",
        "L",
        "RC",
        "ICR",
        "C1",
        "a1",
        "polarity",
        "R",
        "b1",
        "C",
        "s1",
    ]
    params_default = dict(
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
    internals = OrderedDict(
        [
            ("x1", 0.0),
            ("v", 0.0),
            ("i", 0.0),
            ("x2", 0.0),
            ("y1", 0.0),
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
