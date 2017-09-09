
/*
 *
 * Canyge@hotmail.com
 *
 *
 * */

#ifndef NN_H
#define NN_H

struct vect{
    int n;
    double *data;
} vect;

struct matrix{
    int m;
    int n;
    double **data;
};
typedef struct matrix matrix;
struct nn {
    /* Number of input, hidden and output nodes */
    int nlayer;

    int *layer;

    /* Activations for nodes */
    matrix *y;

    /* Weight matrices */
    matrix *weigh;
    matrix *bias;

    matrix *dw;
    matrix *db;
    /* Last changes */

    matrix *e;
    matrix *v;
    matrix *dv;
};

void init(struct nn *, int *, int);
void trian(struct nn *nn,matrix input,matrix output,double eta);

struct nn_one_layer {
    /* Number of input, hidden and output nodes */
    int ni;
    int nh;
    int no;

    /* Activations for nodes */
    double *ai;
    double *ah;
    double *ao;

    /* Weight matrices */
    double **wh;
    double **wo;

    /* Last changes */
    double **ch;
    double **co;

    /* Deltas */
    double *hd; /* Hidden deltas */
    double *od; /* Output deltas */
};

int nn_init(struct nn_one_layer *, int, int, int);
int nn_del(struct nn_one_layer *);
void nn_train(struct nn_one_layer *nn, int, int [][nn->ni], int [][nn->no]);
void nn_test(struct nn_one_layer *nn, int, int [][nn->ni], int [][nn->no]);

#endif /* NN_H */
