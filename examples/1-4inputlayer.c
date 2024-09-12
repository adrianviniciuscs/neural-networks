#include "../include/nn.h"
#include <stdio.h>

int main() {
    // Define the input values
    double inputs[] = {1.0, 2.0, 3.0, 2.5};

    // Define the weights for each neuron
    double weights[][4] = {
        {0.2, 0.8, -0.5, 1.0},
        {0.5, -0.91, 0.26, -0.5},
        {-0.26, -0.27, 0.17, 0.87}
    };

    // Define the biases for each neuron
    double biases[] = {2.0, 3.0, 0.5};

    // Calculate the output for each neuron
    double output[3];
    for (int i = 0; i < 3; i++) {
        output[i] = biases[i];  // Initialize output with the bias term
        for (int j = 0; j < 4; j++) {
            output[i] += inputs[j] * weights[i][j];  // Add the contribution from each input
        }
    }

    // Print the outputs
    printf("[%f, %f, %f]\n", output[0], output[1], output[2]);

    return 0;
}
