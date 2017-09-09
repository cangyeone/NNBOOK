
/*
 *
 * Canyge@hotmail.com
 *
 *
 * */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "nn.h"

matrix
init_matrix(int n, int m) {
    double **weight;
    double *heap;
    matrix out;

    /* Allocate continuously */
    weight = (double **)malloc(sizeof(double *) * n);

    heap = (double *)malloc(sizeof(double) * n * m);

    /* Fill with random numbers */
    for(int i = 0; i < n; i++) {
        weight[i] = heap + i * n;
        for (int j = 0; j < m; j++) {
            weight[i][j] = ((float)rand() / (float)(RAND_MAX / 2.0)) - 1.0;
        }
    }
    out.data=weight;
    out.m=n;
    out.n=m;
    return out;
}

void
free_matrix(matrix mt){
    free(mt.data[0]);
    free(mt.data);
    mt.data[0]=NULL;
}

matrix
matmul(matrix A, matrix B){
    int m, n;
    matrix C;
    m = A.m;
    n = B.n;
    C = init_matrix(m, n);
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            C.data[i][j]=0;
            for(int k=0;k<A.n;k++){
                C.data[i][j]+=A.data[i][k]*B.data[k][j];
            }
        }
    }
	return C;
}
void
matcop(matrix A, matrix B){
    memcpy(B.data[0], A.data[0], sizeof(double)*A.m*A.n);
}

matrix
matadd(matrix A, matrix B){
    int m, n;
    matrix C;
    m = A.m;
    n = A.n;
    C = init_matrix(m, n);
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            C.data[i][j]=A.data[i][j]+B.data[i][j];
        }
    }
    return C;
}

matrix
matsub(matrix A, matrix B){
    int m, n;
    matrix C;
    m = A.m;
    n = A.n;
    C = init_matrix(m, n);
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            C.data[i][j]=A.data[i][j]-B.data[i][j];
        }
    }
    return C;
}


matrix
matdot(matrix A, matrix B){
    int m, n;
    matrix C;
    m = A.m;
    n = A.n;
    C = init_matrix(m, n);
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            C.data[i][j]=A.data[i][j]*B.data[i][j];
        }
    }
    return C;
}

matrix
matcst(matrix A,matrix B,double eta){
    int m, n;
    matrix C;
    m = A.m;
    n = A.n;
    C = init_matrix(m, n);
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            C.data[i][j]=A.data[i][j]+B.data[i][j]*eta;
        }
    }
    return C;
}

matrix
mattra(matrix A){
    int m, n;
    matrix C;
    m = A.m;
    n = A.n;
    C = init_matrix(n, m);
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            C.data[j][i]=A.data[i][j];
        }
    }
    return C;
}



static double
nn_sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

void
sigmoid(matrix x){
    for(int i=0;i<x.m;i++){
        for(int j=0;j<x.n;i++)
        {
            x.data[i][j]=nn_sigmoid(x.data[i][j]);
        }
    }
}

static double
nn_d_sigmoid(double x){
    return exp(-x)/(1+exp(-x))/(1+exp(-x));
}

void
d_sigmoid(matrix x){
    for(int i=0;i<x.m;i++){
        for(int j=0;j<x.n;i++)
        {
            x.data[i][j]=nn_d_sigmoid(x.data[i][j]);
        }
    }
}

void 
nn_forward(struct nn *nn, matrix input) {
    matrix temp_y,temp_v,temp_vb;
    matcop(input, nn->y[0]);
    temp_y=init_matrix(nn->y[0].m,nn->y[0].n);
    matcop(input, temp_y);
    for(int i=0; i<nn->nlayer-1;i++){
        temp_v = matmul(temp_y,nn->weigh[i]);
        temp_vb = matadd(temp_v, nn->bias[i]);
        matcop(temp_vb, nn->y[i+1]);
        matcop(temp_vb, nn->dv[i+1]);
        sigmoid(nn->y[i+1]);
        d_sigmoid(nn->dv[i+1]);
        matcop(nn->y[i+1], temp_y);
        free_matrix(temp_v);
        free_matrix(temp_vb);
    }
}

matrix
mat2vv(matrix A){
    int m, n;
    matrix C;
    m = A.m;
    n = A.n;
    C = init_matrix(n, m);
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            C.data[j][i]=A.data[i][j];
        }
    }
    return C;
}


static void
nn_back_forward(struct nn *nn, matrix output) {
    
    int layer=nn->nlayer;
    matrix temp_delta,temp,ta,tb,sigm_delta;
    matrix vva;
    temp = matsub(output, nn->y[layer-1]);
    matcop(temp, nn->e[layer-1]);
    temp_delta=matdot(temp, nn->dv[layer-1]);
    free_matrix(temp);
    ta = mattra(nn->y[layer-2]);
    tb = mattra(temp_delta);
    temp = matmul(ta,tb);
    matcop(temp,nn->dw[layer-2]);
    matcop(temp_delta,nn->db[layer-2]);
    free_matrix(ta);
    free_matrix(tb);
    free_matrix(temp);
    for(int i=layer-2;i>0;i--){
        sigm_delta=matmul(nn->weigh[i],temp_delta);
        vva=mat2vv(nn->dv[i]);
        free_matrix(temp_delta);
        temp_delta=matdot(sigm_delta,vva);
        free_matrix(vva);
        ta=mat2vv(nn->y[i-1]);
        tb=mattra(temp_delta);
        vva=matmul(ta,tb);
        matcop(vva,nn->dw[i-1]);
        matcop(tb,nn->db[i-1]);
        free_matrix(ta);
        free_matrix(tb);
        free_matrix(vva);
    }

    

    matcop(nn->dw[layer-2],ta);
    

}

void
init(struct nn *nn,int *layer, int nlayer) {

    nn->nlayer = nlayer;
    nn->layer=layer;
    nn->weigh = (matrix *)malloc((nlayer-1)*sizeof(matrix));
    nn->bias = (matrix *)malloc((nlayer-1)*sizeof(matrix));
    nn->dw = (matrix *)malloc((nlayer-1)*sizeof(matrix));
    nn->db = (matrix *)malloc((nlayer-1)*sizeof(matrix));
    nn->e = (matrix *)malloc((nlayer)*sizeof(matrix));
    nn->y = (matrix *)malloc((nlayer)*sizeof(matrix));
    nn->v = (matrix *)malloc((nlayer)*sizeof(matrix));
    nn->dv = (matrix *)malloc((nlayer)*sizeof(matrix));
    for(int i = 0; i < nlayer-1; i++){
        nn->weigh[i] = init_matrix(layer[i], layer[i+1]);
        nn->bias[i] = init_matrix(1, layer[i+1]);
        nn->dw[i] = init_matrix(layer[i], layer[i+1]);
        nn->db[i] = init_matrix(1, layer[i+1]);
    }
    for(int i = 0; i < nlayer; i++){
        nn->e[i] = init_matrix(1, layer[i+1]);
        nn->y[i] = init_matrix(1, layer[i+1]);
        nn->v[i] = init_matrix(1, layer[i+1]);
        nn->dv[i] = init_matrix(1, layer[i+1]);
    }
}

void trian(struct nn *nn,matrix input,matrix output,double eta){
    int N=input.m;
    matrix in;
    matrix out;
    matrix tw;
    matrix tb;
    in = init_matrix(1,input.n);
    out= init_matrix(1,output.n);
    for(int i=0;i<N;i++){
        memcpy(in.data[0],input.data[i],sizeof(double)*input.n);
        memcpy(out.data[0],output.data[i],sizeof(double)*output.n);
        nn_forward(nn,in);
        nn_back_forward(nn,out);
        for(int j=0;j<nn->nlayer-1;j++){
            tw=matcst(nn->weigh[j],nn->dw[j],eta);
            tb=matcst(nn->bias[j],nn->db[j],eta);
            matcop(tw,nn->weigh[j]);
            matcop(tb,nn->bias[j]);
            free_matrix(tw);
            free_matrix(tb);
        }

    }
}



static void
nn_one_freep(void *arg) {
    void **ptr = (void **)arg;
    if (*ptr) {
        free(*ptr);
        *ptr = NULL;
    }
}

static double *
nn_one_act_alloc(int n) {
    double *act;

    act = (double *)malloc(sizeof(double) * n);
    if (!act) {
        return NULL;
    }

    for (int i = 0; i < n; i++) {
        act[i] = 1.0;
    }

    return act;
}

static double **
nn_one_weight_alloc(int n, int m) {
    double **weight;
    double *heap;

    /* Allocate continuously */
    weight = (double **)malloc(sizeof(double *) * n);
    if (!weight) {
        return NULL;
    }

    heap = (double *)malloc(sizeof(double) * n * m);
    if (!heap) {
        nn_one_freep(&weight);
        return NULL;
    }

    /* Fill with random numbers */
    for(int i = 0; i < n; i++) {
        weight[i] = heap + i * n;
        for (int j = 0; j < m; j++) {
            weight[i][j] = ((float)rand() / (float)(RAND_MAX / 2.0)) - 1.0;
        }
    }

    return weight;
}

static double
nn_one_sigmoid(double x) {
    return 1.0 / (1.0 + exp( -x));
}

static void
nn_one_update(struct nn_one_layer *nn, int *inputs) {
    for (int i = 0; i < nn->ni; i++) {
        nn->ai[i] = inputs[i];
    }

    for (int i = 0; i < nn->nh; i++) {
        double sum = 0.0;

        for (int j = 0; j < nn->ni; j++) {
            sum += nn->ai[j] * nn->wh[j][i];
        }

        nn->ah[i] = nn_one_sigmoid(sum);
    }

    for (int i = 0; i < nn->no; i++) {
        double sum = 0.0;

        for (int j = 0; j < nn->nh; j++) {
            sum += nn->ah[j] * nn->wo[j][i];
        }

        nn->ao[i] = nn_one_sigmoid(sum);
    }
}

static void
nn_one_back_propagate(struct nn_one_layer *nn, int *targets) {
    /* Output deltas */
    for (int i = 0; i < nn->no; i++) {
        double error;

        error = targets[i] - nn->ao[i];
        nn->od[i] = nn->ao[i] * (1 - nn->ao[i]) * error;
    }

    /* Hidden deltas */
    for (int i = 0; i < nn->nh; i++) {
        double error = 0.0;

        for (int j = 0; j < nn->no; j++) {
            error += nn->od[j] * nn->wo[i][j];
        }

        nn->hd[i] = nn->ah[i] * (1 - nn->ah[i]) * error;
    }

    /* Update output weights */
    for (int i = 0; i < nn->nh; i++) {
        for (int j = 0; j < nn->no; j++) {
            double change;

            change = nn->od[j] * nn->ah[i];
            nn->wo[i][j] += 1. * change + 0.1 * nn->co[i][j];
            nn->co[i][j] = change;
        }
    }

    /* Update hidden weights */
    for (int i = 0; i < nn->ni; i++) {
        for (int j = 0; j < nn->nh; j++) {
            double change;

            change = nn->hd[j] * nn->ai[i];
            nn->wh[i][j] += 1.0 * change + 0.1 * nn->ch[i][j];
            nn->ch[i][j] = change;
        }
    }
}

int
nn_one_del(struct nn_one_layer *nn) {
    nn_one_freep(&nn->ai);
    nn_one_freep(&nn->ah);
    nn_one_freep(&nn->ao);
    nn_one_freep(&nn->hd);
    nn_one_freep(&nn->od);

    if (nn->wh) {
        nn_one_freep(&nn->wh[0]);
        nn_one_freep(&nn->wh);
    }

    if (nn->wo) {
        nn_one_freep(&nn->wo[0]);
        nn_one_freep(&nn->wo);
    }

    if (nn->ch) {
        nn_one_freep(&nn->ch[0]);
        nn_one_freep(&nn->ch);
    }

    if (nn->co) {
        nn_one_freep(&nn->co[0]);
        nn_one_freep(&nn->co);
    }

    return 1;
}

int
nn_one_init(struct nn_one_layer *nn, int ni, int nh, int no) {
    nn->ni = ni;
    nn->nh = nh;
    nn->no = no;

    /* Assign NULL so that nn_one_freep won't fail */
    nn->ai = NULL;
    nn->ah = NULL;
    nn->ao = NULL;
    nn->wh = NULL;
    nn->wo = NULL;
    nn->ch = NULL;
    nn->co = NULL;
    nn->hd = NULL;
    nn->od = NULL;

    nn->ai = nn_one_act_alloc(ni);
    if (!nn->ai) {
        return nn_one_del(nn);
    }

    nn->ah = nn_one_act_alloc(nh);
    if (!nn->ah) {
        return nn_one_del(nn);
    }

    nn->ao = nn_one_act_alloc(no);
    if (!nn->ao) {
        return nn_one_del(nn);
    }

    nn->hd = nn_one_act_alloc(nh);
    if (!nn->hd) {
        return nn_one_del(nn);
    }

    nn->od = nn_one_act_alloc(no);
    if (!nn->od) {
        return nn_one_del(nn);
    }

    nn->wh = nn_one_weight_alloc(ni, nh);
    if (!nn->wh) {
        return nn_one_del(nn);
    }

    nn->wo = nn_one_weight_alloc(nh, no);
    if (!nn->wo) {
        return nn_one_del(nn);
    }

    nn->ch = nn_one_weight_alloc(ni, nh);
    if (!nn->ch) {
        return nn_one_del(nn);
    }

    nn->co = nn_one_weight_alloc(nh, no);
    if (!nn->co) {
        return nn_one_del(nn);
    }

    return 0;
}



void
nn_one_train(struct nn_one_layer *nn, int n, int inputs[][nn->ni], int targets[][nn->no]) {
    for (int i = 0; i < 100000; i++) {
        for (int j = 0; j < n; j++) {
            nn_one_update(nn, inputs[j]);
            nn_one_back_propagate(nn, targets[j]);
        }
    }
}

void
nn_one_test(struct nn_one_layer *nn, int n, int inputs[][nn->ni], int targets[][nn->no]) {
    for (int i = 0; i < n; i++) {
        nn_one_update(nn, inputs[i]);

        /* Print inputs */
        printf("%.2f", nn->ai[0]);
        for (int j = 1; j < nn->ni; j++) {
            printf(", %.2f", nn->ai[j]);
        }

        /* Print delimiter */
        printf(" -> ");

        /* Print outputs */
        printf("%.2f", nn->ao[0]);
        for (int j = 1; j < nn->no; j++) {
            printf(", %.2f", nn->ao[j]);
        }

        /* Print new line */
        printf("\n");
    }
}
