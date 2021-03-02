#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>


#define  V_MIN		-80
#define  V_MAX		80
#define  N_MIN		0.0
#define  N_MAX		1.0
#define  M_MIN		0.0
#define  M_MAX		1.0
#define  H_MIN		0.0
#define  H_MAX		1.0
#define  A_MIN		0.0
#define  A_MAX		1.0
#define  B_MIN		0.0
#define  B_MAX		1.0
extern "C"{

__global__ void  generate_seed(
    int num,
    curandState *seed
)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int nid = tid; nid < num; nid += total_threads)
        curand_init(clock64(), nid, 0, &seed[nid]);

    return;
}

struct States {
    double v;
    double n;
    double m;
    double h;
    double a;
    double b;
    double spike;
    double v1;
    double v2;
    double refactory;
};

struct Derivatives {
    double v;
    double n;
    double m;
    double h;
    double a;
    double b;
    double refactory;
};


__device__ void clip(States &states)
{
    states.v = fmax(states.v, V_MIN);
    states.v = fmin(states.v, V_MAX);
    states.n = fmax(states.n, N_MIN);
    states.n = fmin(states.n, N_MAX);
    states.m = fmax(states.m, M_MIN);
    states.m = fmin(states.m, M_MAX);
    states.h = fmax(states.h, H_MIN);
    states.h = fmin(states.h, H_MAX);
    states.a = fmax(states.a, A_MIN);
    states.a = fmin(states.a, A_MAX);
    states.b = fmax(states.b, B_MIN);
    states.b = fmin(states.b, B_MAX);
}

__device__ void forward(
    States &states,
    Derivatives &gstates,
    double dt
)
{
    states.v += dt * gstates.v;
    states.n += dt * gstates.n;
    states.m += dt * gstates.m;
    states.h += dt * gstates.h;
    states.a += dt * gstates.a;
    states.b += dt * gstates.b;
    states.refactory += dt * gstates.refactory;
}

__device__ int ode(
    States &states,
    Derivatives &gstates,
    double MS,
    double NS,
    double HS,
    double GNA,
    double GK,
    double GL,
    double GA,
    double ENA,
    double EK,
    double EL,
    double EA,
    double SIGMA,
    double REFPERIOD,
    double &stimulus,
    curandStateXORWOW_t  &seed
)
{
    double alpha;
    double beta;
    double n_inf;
    double tau_n;
    double m_inf;
    double tau_m;
    double h_inf;
    double tau_h;
    double a_inf;
    double tau_a;
    double b_inf;
    double tau_b;
    double i_na;
    double i_k;
    double i_l;
    double i_a;

    alpha = (exp(((-((states.v + 50.0) + NS)) / 10.0)) - 1.0);
    if ((abs(alpha) <= 1e-07)) {
        alpha = 0.1;
    } else {
        alpha = ((-0.01 * ((states.v + 50.0) + NS)) / alpha);
    }
    beta = (0.125 * exp(((-((states.v + 60.0) + NS)) / 80.0)));
    n_inf = (alpha / (alpha + beta));
    tau_n = (2.0 / (3.8 * (alpha + beta)));
    alpha = (exp(((-((states.v + 35.0) + MS)) / 10.0)) - 1.0);
    if ((abs(alpha) <= 1e-07)) {
        alpha = 1.0;
    } else {
        alpha = ((-0.1 * ((states.v + 35.0) + MS)) / alpha);
    }
    beta = (4.0 * exp(((-((states.v + 60.0) + MS)) / 18.0)));
    m_inf = (alpha / (alpha + beta));
    tau_m = (1.0 / (3.8 * (alpha + beta)));
    alpha = (0.07 * exp(((-((states.v + 60.0) + HS)) / 20.0)));
    beta = (1.0 / (1.0 + exp(((-((states.v + 30.0) + HS)) / 10.0))));
    h_inf = (alpha / (alpha + beta));
    tau_h = (1.0 / (3.8 * (alpha + beta)));
    a_inf = cbrt(((0.0761 * exp(((states.v + 94.22) / 31.84))) / (1.0 + exp(((states.v + 1.17) / 28.93)))));
    tau_a = (0.3632 + (1.158 / (1.0 + exp(((states.v + 55.96) / 20.12)))));
    b_inf = pow((1 / (1 + exp(((states.v + 53.3) / 14.54)))), 4.0);
    tau_b = (1.24 + (2.678 / (1 + exp(((states.v + 50) / 16.027)))));
    i_na = (((GNA * pow(states.m, 3)) * states.h) * (states.v - ENA));
    i_k = ((GK * pow(states.n, 4)) * (states.v - EK));
    i_l = (GL * (states.v - EL));
    i_a = (((GA * pow(states.a, 3)) * states.b) * (states.v - EA));
    gstates.v = ((((stimulus - i_na) - i_k) - i_l) - i_a);
    gstates.n = (((n_inf - states.n) / tau_n) + (0.0+SIGMA*curand_normal(&seed)));
    gstates.m = (((m_inf - states.m) / tau_m) + (0.0+SIGMA*curand_normal(&seed)));
    gstates.h = (((h_inf - states.h) / tau_h) + (0.0+SIGMA*curand_normal(&seed)));
    gstates.a = (((a_inf - states.a) / tau_a) + (0.0+SIGMA*curand_normal(&seed)));
    gstates.b = (((b_inf - states.b) / tau_b) + (0.0+SIGMA*curand_normal(&seed)));
    gstates.refactory = (states.refactory < 0);
    return 0;
}


/* post processing */
__device__ int post(
    States &states,
    double MS,
    double NS,
    double HS,
    double GNA,
    double GK,
    double GL,
    double GA,
    double ENA,
    double EK,
    double EL,
    double EA,
    double SIGMA,
    double REFPERIOD
)
{

    states.spike = (((states.v1 < states.v2) * (states.v < states.v2)) * (states.v > -30.));
    states.v1 = states.v2;
    states.v2 = states.v;
    states.spike = ((states.spike > 0.0) * (states.refactory >= 0));
    states.refactory = (states.refactory - ((states.spike > 0.0) * REFPERIOD));
    return 0;
}

__global__ void NoisyConnorStevens (
    int num_thread,
    double dt,
    curandStateXORWOW_t *g_state,
    double *g_n,
    double *g_m,
    double *g_h,
    double *g_a,
    double *g_b,
    double *g_v1,
    double *g_v2,
    double *g_refactory,
    double *g_ms,
    double *g_ns,
    double *g_hs,
    double *g_gNa,
    double *g_gK,
    double *g_gL,
    double *g_ga,
    double *g_ENa,
    double *g_EK,
    double *g_EL,
    double *g_Ea,
    double *g_sigma,
    double *g_refperiod,
    double *g_stimulus,
    double *g_spike,
    double *g_v
)
{
    /* TODO: option for 1-D or 2-D */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int nid = tid; nid < num_thread; nid += total_threads) {

        States states;
        Derivatives gstates;

        /* import data */
        states.v = g_v2[nid];
        states.n = g_n[nid];
        states.m = g_m[nid];
        states.h = g_h[nid];
        states.a = g_a[nid];
        states.b = g_b[nid];
        states.spike = g_spike[nid];
        states.v1 = g_v1[nid];
        states.v2 = g_v2[nid];
        states.refactory = g_refactory[nid];
        double MS = g_ms[nid];
        double NS = g_ns[nid];
        double HS = g_hs[nid];
        double GNA = g_gNa[nid];
        double GK = g_gK[nid];
        double GL = g_gL[nid];
        double GA = g_ga[nid];
        double ENA = g_ENa[nid];
        double EK = g_EK[nid];
        double EL = g_EL[nid];
        double EA = g_Ea[nid];
        double SIGMA = g_sigma[nid]/sqrt(dt/1000.0);
        double REFPERIOD = g_refperiod[nid];
        double stimulus = g_stimulus[nid];
        curandStateXORWOW_t localstate = g_state[nid];

        /* compute gradient */
        ode(states, gstates, MS, NS, HS, GNA, GK, GL, GA, ENA, EK, EL, EA, SIGMA, REFPERIOD, stimulus, localstate);

        /* solve ode */
        forward(states, gstates, dt);

        /* clip */
        clip(states);

        /* post processing */
        post(states, MS, NS, HS, GNA, GK, GL, GA, ENA, EK, EL, EA, SIGMA, REFPERIOD);

        /* export data */
        g_v[nid] = states.v;
        g_n[nid] = states.n;
        g_m[nid] = states.m;
        g_h[nid] = states.h;
        g_a[nid] = states.a;
        g_b[nid] = states.b;
        g_spike[nid] = states.spike;
        g_v1[nid] = states.v1;
        g_v2[nid] = states.v2;
        g_refactory[nid] = states.refactory;
        g_state[nid] = localstate;
    }

    return;
}

}
