#include <stdio.h>
#include <stdlib.h>
#include "../include/nn.h"

#define NET_BATCH_SIZE 300
#define NET_INPUT_LAYER_1_SIZE 2
#define NET_OUTPUT_LAYER_SIZE 5

void print_layer_output(layer_dense_t *layer, int batch_idx);

int main(void)
{
    srand(0); // Seed the random number generator

    int i, j;
    spiral_data_t X_data;
    layer_dense_t X;
    layer_dense_t layer1;

    // Generate spiral data
    spiral_data(100, 3, &X_data);
    if (X_data.x == NULL) {
        printf("Failed to generate data.\n");
        return 0;
    }

    X.callback = NULL;
    layer1.callback = activation_ReLU_callback;
    layer_init(&layer1, NET_INPUT_LAYER_1_SIZE, NET_OUTPUT_LAYER_SIZE);

    // Process batches
    for (i = 0; i < NET_BATCH_SIZE; i++) {
        X.output = &X_data.x[i * 2];
        forward(&X, &layer1);

        // Print layer output
        printf("Batch: %d, Layer1 Output: ", i);
        print_layer_output(&layer1, i);
        printf("\n");
    }

    // Clean up
    dealloc_layer(&layer1);
    dealloc_spiral(&X_data);

    return 0;
}

// Function to print layer output
void print_layer_output(layer_dense_t *layer, int batch_idx) {
    int j;
    for (j = 0; j < layer->output_size; j++) {
        printf("%f ", layer->output[j]);
    }
}

