#include "../include/nn.h"
#include <stdio.h>

int main() {
  // Define the input values
  double inputs[] = {1.0, 2.0, 3.0, 2.5};

  // Define the weights for each neuron
  double weights1[] = {0.2, 0.8, -0.5, 1.0};
  double weights2[] = {0.5, -0.91, 0.26, -0.5};
  double weights3[] = {-0.26, -0.27, 0.17, 0.87};

  // Define the biases for each neuron
  double bias1 = 2.0;
  double bias2 = 3.0;
  double bias3 = 0.5;

  // Calculate the output for each neuron
  double output[] = {
      inputs[0] * weights1[0] + inputs[1] * weights1[1] +
          inputs[2] * weights1[2] + inputs[3] * weights1[3] + bias1,

      inputs[0] * weights2[0] + inputs[1] * weights2[1] +
          inputs[2] * weights2[2] + inputs[3] * weights2[3] + bias2,

      inputs[0] * weights3[0] + inputs[1] * weights3[1] +
          inputs[2] * weights3[2] + inputs[3] * weights3[3] + bias3
  };

  // Print the outputs
  printf("[%f, %f, %f]\n", output[0], output[1], output[2]);

  return 0;
}

