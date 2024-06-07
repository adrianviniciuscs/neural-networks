#include <stdio.h>
#include <stdlib.h>
#include "../include/nn.h"

#define NET_BATCH_SIZE 300
#define NET_INPUT_LAYER_1_SIZE 2 // Can be replaced with (sizeof(var)/sizeof(double))
#define NET_OUTPUT_LAYER_SIZE 5 // Can be replaced with (sizeof(var)/sizeof(double))

int main(void)
{

    //seed the random values
    srand(0);

    int i = 0;
    int j = 0;
    spiral_data_t X_data;
    layer_dense_t X;
    layer_dense_t layer1;


    spiral_data(100,3,&X_data);
    if(X_data.x == NULL){
        printf("data null\n");
        return 0;
    }

    X.callback = NULL;
    layer1.callback = activation_ReLU_callback;

    layer_init(&layer1,NET_INPUT_LAYER_1_SIZE,NET_OUTPUT_LAYER_SIZE);

    for(i = 0; i < NET_BATCH_SIZE;i++){
        X.output = &X_data.x[i*2];
        forward(&X,&layer1);

        printf("batch: %d layer1_output: ",i);
        for(j = 0; j < layer1.output_size; j++){
            printf("%f ",layer1.output[j]);
        }
        printf("\n");
    }

    dealloc_layer(&layer1);
    dealloc_spiral(&X_data);
    return 0;
}
