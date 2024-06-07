#include <stdio.h>

int main() {
  // Define the input values
  double inputs[] = {1.0, 2.0, 3.0};

  // Define the corresponding weights for the inputs
  double weights[] = {3.1, 2.1, 8.7};

  // Define the bias term
  double bias = 3.0;

  // Compute the weighted sum of inputs and add the bias
  double output = inputs[0] * weights[0] +  // Contribution from the first input
                  inputs[1] * weights[1] +  // Contribution from the second input
                  inputs[2] * weights[2] +  // Contribution from the third input
                  bias;                     // Add the bias term

  // Print the output
  printf("%f\n", output);

  return 0;
}
