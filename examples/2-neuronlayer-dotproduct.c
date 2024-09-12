#include <stdio.h>
#include "../include/nn.h"

// Define sizes for input and output layers
#define NET_INPUT_LAYER_SIZE 4 
#define NET_OUTPUT_LAYER_SIZE 3 

int main(void)
{
    // Define input values
    double input[NET_INPUT_LAYER_SIZE] = {1.0, 2.0, 3.0, 2.5};
    
    // Define weights for each output neuron
    double weights[NET_OUTPUT_LAYER_SIZE][NET_INPUT_LAYER_SIZE] = {
        {0.2, 0.8, -0.5, 1.0},
        {0.5, -0.91, 0.26, -0.5},
        {-0.26, -0.27, 0.17, 0.87}
    };
    
    // Define biases for each output neuron
    double biases[NET_OUTPUT_LAYER_SIZE] = {2.0, 3.0, 0.5};

    // Define array to store output values
    double output[NET_OUTPUT_LAYER_SIZE] = {0.0};

    // Calculate the output of the neural network layer
    for (int i = 0; i < NET_OUTPUT_LAYER_SIZE; i++) {
        output[i] = biases[i];  // Initialize output with the bias term
        for (int j = 0; j < NET_INPUT_LAYER_SIZE; j++) {
            output[i] += input[j] * weights[i][j];  // Add the contribution from each input
        }
    }

    // Print the output values
    printf("Network output: %f %f %f\n", output[0], output[1], output[2]);

    return 0;
}
