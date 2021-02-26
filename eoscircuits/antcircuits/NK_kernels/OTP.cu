#define  V_MIN		0
#define  V_MAX		1000000000.0
#define  UH_MIN		0.0
#define  UH_MAX		50000.0
#define  X1_MIN		0.0
#define  X1_MAX		1.0
#define  X2_MIN		0.0
#define  X2_MAX		1.0
#define  X3_MIN		0.0
#define  X3_MAX		1000.0

struct States {
    double v;
    double I;
    double uh;
    double duh;
    double x1;
    double x2;
    double x3;
};

struct Derivatives {
    double uh;
    double duh;
    double x1;
    double x2;
    double x3;
};


__device__ void clip(States &states)
{
    states.v = fmax(states.v, V_MIN);
    states.v = fmin(states.v, V_MAX);
    states.uh = fmax(states.uh, UH_MIN);
    states.uh = fmin(states.uh, UH_MAX);
    states.x1 = fmax(states.x1, X1_MIN);
    states.x1 = fmin(states.x1, X1_MAX);
    states.x2 = fmax(states.x2, X2_MIN);
    states.x2 = fmin(states.x2, X2_MAX);
    states.x3 = fmax(states.x3, X3_MIN);
    states.x3 = fmin(states.x3, X3_MAX);
}

__device__ void forward(
    States &states,
    Derivatives &gstates,
    double dt
)
{
    states.uh += dt * gstates.uh;
    states.duh += dt * gstates.duh;
    states.x1 += dt * gstates.x1;
    states.x2 += dt * gstates.x2;
    states.x3 += dt * gstates.x3;
}

__device__ int ode(
    States &states,
    Derivatives &gstates,
    double BR,
    double DR,
    double GAMMA,
    double A1,
    double B1,
    double A2,
    double B2,
    double A3,
    double B3,
    double KAPPA,
    double P,
    double C,
    double IMAX,
    double &stimulus
)
{
    double f;

    gstates.x1 = (((BR * states.v) * (1.0 - states.x1)) - (DR * states.x1));
    f = (cbrt((states.x2 * states.x2)) * cbrt((states.x3 * states.x3)));
    gstates.x2 = ((((A2 * states.x1) * (1.0 - states.x2)) - (B2 * states.x2)) - (KAPPA * f));
    gstates.x3 = ((A3 * states.x2) - (B3 * states.x3));
    states.I = ((IMAX * states.x2) / (states.x2 + C));
    gstates.uh = states.duh;
    gstates.duh = ((((-2 * A1) * B1) * states.duh) + ((A1 * A1) * (stimulus - states.uh)));
    states.v = (states.uh + (GAMMA * states.duh));
    return 0;
}



__global__ void OTP (
    int num_thread,
    double dt,
    double *g_v,
    double *g_uh,
    double *g_duh,
    double *g_x1,
    double *g_x2,
    double *g_x3,
    double *g_br,
    double *g_dr,
    double *g_gamma,
    double *g_a1,
    double *g_b1,
    double *g_a2,
    double *g_b2,
    double *g_a3,
    double *g_b3,
    double *g_kappa,
    double *g_p,
    double *g_c,
    double *g_Imax,
    double *g_stimulus,
    double *g_I
)
{
    /* TODO: option for 1-D or 2-D */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int nid = tid; nid < num_thread; nid += total_threads) {

        States states;
        Derivatives gstates;

        /* import data */
        states.v = g_v[nid];
        states.I = g_I[nid];
        states.uh = g_uh[nid];
        states.duh = g_duh[nid];
        states.x1 = g_x1[nid];
        states.x2 = g_x2[nid];
        states.x3 = g_x3[nid];
        double BR = g_br[nid];
        double DR = g_dr[nid];
        double GAMMA = g_gamma[nid];
        double A1 = g_a1[nid];
        double B1 = g_b1[nid];
        double A2 = g_a2[nid];
        double B2 = g_b2[nid];
        double A3 = g_a3[nid];
        double B3 = g_b3[nid];
        double KAPPA = g_kappa[nid];
        double P = g_p[nid];
        double C = g_c[nid];
        double IMAX = g_Imax[nid];
        double stimulus = g_stimulus[nid];

        /* compute gradient */
        ode(states, gstates, BR, DR, GAMMA, A1, B1, A2, B2, A3, B3, KAPPA, P, C, IMAX, stimulus);

        /* solve ode */
        forward(states, gstates, dt);

        /* clip */
        clip(states);        

        /* export data */
        g_v[nid] = states.v;
        g_I[nid] = states.I;
        g_uh[nid] = states.uh;
        g_duh[nid] = states.duh;
        g_x1[nid] = states.x1;
        g_x2[nid] = states.x2;
        g_x3[nid] = states.x3;
        // printf("(%f, %f), (%f,%f), (%.f, %f)\n", gstates.x1, g_x1[nid], gstates.x2, g_x2[nid], gstates.x3, g_x3[nid]);
    }

    return;
}