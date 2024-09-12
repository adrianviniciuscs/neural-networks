#include <stdio.h>
#include <stdlib.h>
#include "../include/nn.h"
#include "../include/dataset.h"

#define NET_BATCH_SIZE 300
#define NET_INPUT_LAYER_1_SIZE 2
#define NET_OUTPUT_LAYER_SIZE 5

// Function to print layer output
void print_layer_output(layer_dense_t *layer) {
    for (int j = 0; j < layer->output_size; j++) {
        printf("%f ", layer->output[j]);
    }
}

int main(void) {
    srand(0); // Seed the random number generator

    spiral_data_t X_data;
    layer_dense_t input_layer, hidden_layer;

    // Generate spiral data
    spiral_data(100, 3, &X_data);
    if (X_data.x == NULL) {
        printf("Failed to generate data.\n");
        return 0;
    }

    input_layer.callback = NULL;
    hidden_layer.callback = activation_ReLU_callback;
    layer_init(&hidden_layer, NET_INPUT_LAYER_1_SIZE, NET_OUTPUT_LAYER_SIZE);

    // Process batches
    for (int i = 0; i < NET_BATCH_SIZE; i++) {
        input_layer.output = &X_data.x[i * 2];
        forward(&input_layer, &hidden_layer);

        // Print layer output
        printf("Batch: %d, Layer1 Output: ", i);
        print_layer_output(&hidden_layer);
        printf("\n");
    }

    // Clean up
    dealloc_layer(&hidden_layer);
    dealloc_spiral(&X_data);

    return 0;
}
