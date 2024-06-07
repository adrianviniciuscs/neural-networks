#include <stdio.h>

// Calculate the dot product of two arrays
double dot_product(double inputs[], double weights[], int length) {
    double result = 0.0;
    for (int i = 0; i < length; i++) {
        result += inputs[i] * weights[i];
    }
    return result;
}

// Calculate the output of a neuron
double neuron_output(double inputs[], double weights[], int length, double bias) {
    double dot = dot_product(inputs, weights, length);
    return dot + bias;
}

int main() {
    double inputs[] = {1.2, 3.2, 3.4, 2.5};
    int length = sizeof(inputs) / sizeof(inputs[0]);


    // 4 inputs, so 4 weights
    double weights1[] = {0.2, 1.0, -0.5, 0.3};
    double bias1 = 2.0;

    double weights2[] = {0.5, -0.91, 0.26, 0.64};
    double bias2 = 1.5;

    double weights3[] = {-0.26, -0.27, 0.17, 0.23};
    double bias3 = 0.5;

    // Calculate the outputs for each set of weights and biases
    double output1 = neuron_output(inputs, weights1, length, bias1);
    double output2 = neuron_output(inputs, weights2, length, bias2);
    double output3 = neuron_output(inputs, weights3, length, bias3);

    // Print the results
    printf("Output with weights1 and bias1: %f\n", output1);
    printf("Output with weights2 and bias2: %f\n", output2);
    printf("Output with weights3 and bias3: %f\n", output3);

    return 0;
}

