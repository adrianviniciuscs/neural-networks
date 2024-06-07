#include "../include/utils.h"

double dot_product(const double *vec1, const double *vec2, double *bias, int size) {
    double result = 0.0;
    for (int i = 0; i < size; i++) {
        result += vec1[i] * vec2[i];
    }
    result += *bias;
    return result;
}

void layer_output(double *input,double *weights,double *bias,int input_size,double *outputs,int output_size){
    int i = 0;
    int offset = 0;
    for(i = 0; i < output_size; i++){
        outputs[i] = dot_product(input,weights + offset,&bias[i],input_size);
        offset+=input_size;
    }
}

