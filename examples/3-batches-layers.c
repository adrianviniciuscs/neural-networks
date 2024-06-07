#include <stdio.h>
#include <stdlib.h>
#include "../include/nn.h"

#define RAND_HIGH_RANGE (0.10)
#define RAND_MIN_RANGE (-0.10)
#define INIT_BIASES (0.0)

#define NET_BATCH_SIZE 3
#define NET_INPUT_LAYER_1_SIZE 4 
#define NET_HIDDEN_LAYER_2_SIZE 5 
#define NET_OUTPUT_LAYER_SIZE 2 


int main()
{

    //seed the random values
    srand(0);

    int i = 0;
    int j = 0;
    layer_dense_t X;
    layer_dense_t layer1;
    layer_dense_t layer2;
    double X_input[NET_BATCH_SIZE][NET_INPUT_LAYER_1_SIZE] = {
        {1.0,2.0,3.0,2.5},
        {2.0,5.0,-1.0,2.0},
        {-1.5,2.7,3.3,-0.8}
    };


    layer_init(&layer1,NET_INPUT_LAYER_1_SIZE,NET_HIDDEN_LAYER_2_SIZE);
    layer_init(&layer2,NET_HIDDEN_LAYER_2_SIZE,NET_OUTPUT_LAYER_SIZE);

    for(i = 0; i < NET_BATCH_SIZE;i++){
        X.output = &X_input[i][0];

        forward(&X,&layer1);

        printf("batch: %d layerX_output: ",i);
        for(j = 0; j < layer1.output_size; j++){
            printf("%f ",layer1.output[j]);
        }
        printf("\n");

        forward(&layer1,&layer2);
        printf("batch: %d layerY_output: ",i);
        for(j = 0; j < layer2.output_size; j++){
            printf("%f ",layer2.output[j]);
        }
        printf("\n");
    }


    dealloc_layer(&layer1);
    dealloc_layer(&layer2);

    return 0;
}
