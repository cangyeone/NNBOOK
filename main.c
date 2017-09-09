/*
 *
 * Canyge@hotmail.com
 *
 *
 * */

#include <stdio.h>

#include "nn.h"

int
main(int argc, char *argv[]) {
    struct nn nn;
    int layers[3]={2, 3, 1};

    init(&nn,layers,3);

    return 0;
}
