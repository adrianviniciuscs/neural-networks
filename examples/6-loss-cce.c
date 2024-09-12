#include "../include/dataset.h"
#include "../include/nn.h"
#include "../include/losscce.h"
#include <stdio.h>
#include <stdlib.h>

#define NET_BATCH_SIZE 100
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
        printf("Data is null\n");
        return 0;
    }

    input_layer.callback = NULL;
    dense1.callback = activation_ReLU_callback;
    dense2.callback = NULL;

    // Initialize layers
    layer_init(&dense1, NET_INPUT_LAYER_1_SIZE, NET_INPUT_LAYER_2_SIZE);
    layer_init(&dense2, NET_INPUT_LAYER_2_SIZE, NET_OUTPUT_LAYER_SIZE);

    // Loss function initialization
    Loss_CCE *loss_function = Create_LCCE();

    // Process batches
    for (int i = 0; i < NET_BATCH_SIZE; i++) {
        input_layer.output = &X_data.x[i * 2];

        // Forward pass through layer 1
        forward(&input_layer, &dense1);
        printf("Batch: %d\n", i);
        print_layer_output(&dense1, "\tLayer 1 Output");

        // Forward pass through layer 2
        forward(&dense1, &dense2);
        print_layer_output(&dense2, "\tLayer 2 Output");

        // Apply softmax activation to layer 2
        activation_softmax(&dense2);
        print_layer_output(&dense2, "\tSoftmax (Layer 2)");

        // Calculate loss
        int yValue = (int) *(X_data.y) + 1; // type sorcery 
        double loss = calculate_loss(loss_function, dense2.output, &yValue, 1, NET_OUTPUT_LAYER_SIZE);
        printf("\tLoss: %f\n", loss);

        // Print normalized sum of layer 2 output
        printf("\tLayer 2 Normalized Sum: %f\n\n", sum_softmax_layer_output(&dense2));
    }

    // Deallocate memory
    dealloc_layer(&dense1);
    dealloc_layer(&dense2);
    dealloc_spiral(&X_data);

    // Free loss function
    free(loss_function);

    return 0;
}
