#include <stdio.h>
#include <stdlib.h>
#include "../include/nn.h"

#define RAND_HIGH_RANGE (0.10)
#define RAND_MIN_RANGE (-0.10)
#define INIT_ (0.0)

#define NET_BATCH_SIZE 3
#define NET_INPUT_LAYER_1_SIZE 4 
#define NET_HIDDEN_LAYER_2_SIZE 5 
#define NET_OUTPUT_LAYER_SIZE 2 

int main() {
    // Seed the random values
    srand(0);

    // Define input values for the batches
    double input_batches[NET_BATCH_SIZE][NET_INPUT_LAYER_1_SIZE] = {
        {1.0, 2.0, 3.0, 2.5},
        {2.0, 5.0, -1.0, 2.0},
        {-1.5, 2.7, 3.3, -0.8}
    };

    // Initialize layers
    layer_dense_t input_layer, hidden_layer, output_layer;
    layer_init(&hidden_layer, NET_INPUT_LAYER_1_SIZE, NET_HIDDEN_LAYER_2_SIZE);
    layer_init(&output_layer, NET_HIDDEN_LAYER_2_SIZE, NET_OUTPUT_LAYER_SIZE);

    // Process each batch
    for (int batch = 0; batch < NET_BATCH_SIZE; batch++) {
        input_layer.output = input_batches[batch];

        // Forward pass through the hidden layer
        forward(&input_layer, &hidden_layer);
        printf("Batch %d, Hidden Layer Output: ", batch);
        for (int j = 0; j < hidden_layer.output_size; j++) {
            printf("%f ", hidden_layer.output[j]);
        }
        printf("\n");

        // Forward pass through the output layer
        forward(&hidden_layer, &output_layer);
        printf("Batch %d, Output Layer Output: ", batch);
        for (int j = 0; j < output_layer.output_size; j++) {
            printf("%f ", output_layer.output[j]);
        }
        printf("\n");
    }

    // Deallocate layers
    dealloc_layer(&hidden_layer);
    dealloc_layer(&output_layer);

    return 0;
}
