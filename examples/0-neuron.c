#include <stdio.h>

int main() {
  // Define the input values
  double inputs[] = {1.0, 2.0, 3.0};

  // Define the corresponding weights for the inputs
  double weights[] = {3.1, 2.1, 8.7};

  // Define the bias term
  double bias = 3.0;

  // Compute the weighted sum of inputs and add the bias
  double output = bias; // Initialize output with the bias term
  for (int i = 0; i < 3; i++) {
    output += inputs[i] * weights[i]; // Add the contribution from each input
  }

  // Print the output
  printf("%f\n", output);
  return 0;
}
