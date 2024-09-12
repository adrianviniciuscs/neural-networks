#include "../include/dataset.h"
#include "../include/nn.h"
#include <stdio.h>
#include <stdlib.h>

#define NET_BATCH_SIZE 300
#define NET_INPUT_LAYER_1_SIZE 2
#define NET_INPUT_LAYER_2_SIZE 3
#define NET_OUTPUT_LAYER_SIZE 3

// Function to print layer output
void print_layer_output(layer_dense_t *layer, const char *layer_name) {
    printf("%s -> ", layer_name);
    for (int j = 0; j < layer->output_size; j++) {
        printf("%f \t ", layer->output[j]);
    }
    printf("\n");
}

int main() {
    srand(0); // Seed the random number generator

    spiral_data_t X_data;
    layer_dense_t input_layer, dense1, dense2;

    // Generate spiral data
    spiral_data(100, 3, &X_data);
    if (X_data.x == NULL) {
        printf("data null\n");
        return 0;
    }

    input_layer.callback = NULL;
    dense1.callback = activation_ReLU_callback;
    dense2.callback = NULL;

    layer_init(&dense1, NET_INPUT_LAYER_1_SIZE, NET_INPUT_LAYER_2_SIZE);
    layer_init(&dense2, NET_INPUT_LAYER_2_SIZE, NET_OUTPUT_LAYER_SIZE);

    // Process batches
    for (int i = 0; i < NET_BATCH_SIZE; i++) {
        input_layer.output = &X_data.x[i * 2];
        forward(&input_layer, &dense1);

        printf("Batch: %d\n", i);
        print_layer_output(&dense1, "\tLayer1");

        forward(&dense1, &dense2);
        print_layer_output(&dense2, "\tLayer2");

        activation_softmax(&dense2);
        print_layer_output(&dense2, "\tSoftmax (Layer 2)");

        printf("\tLayer 2 Normalized Sum: %f\n\n", sum_softmax_layer_output(&dense2));
    }

    // Clean up
    dealloc_layer(&dense1);
    dealloc_layer(&dense2);
    dealloc_spiral(&X_data);

    return 0;
}
