#include "../include/nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

double rand_range(double min, double max) {
  double range = max - min;
  double div = RAND_MAX / range;
  return min + (rand() / div);
}

void layer_init(layer_dense_t *layer, int input_size, int output_size) {
  layer->input_size = input_size;
  layer->output_size = output_size;

  // Allocate memory for weights, biases, and output
  layer->weights = malloc(sizeof(double) * input_size * output_size);
  if (layer->weights == NULL) {
    fprintf(stderr, "Error: Unable to allocate memory for weights\n");
    exit(1);
  }

  layer->biases = malloc(sizeof(double) * output_size);
  if (layer->biases == NULL) {
    fprintf(stderr, "Error: Unable to allocate memory for biases\n");
    free(layer->weights);
    exit(1);
  }

  layer->output = malloc(sizeof(double) * output_size);
  if (layer->output == NULL) {
    fprintf(stderr, "Error: Unable to allocate memory for output\n");
    free(layer->weights);
    free(layer->biases);
    exit(1);
  }

  // Initialize biases with INIT_BIASES
  for (int i = 0; i < output_size; i++) {
    layer->biases[i] = INIT_BIASES;
  }

  // Initialize weights with random values
  for (int i = 0; i < input_size * output_size; i++) {
    layer->weights[i] = rand_range(RAND_MIN_RANGE, RAND_HIGH_RANGE);
  }
}

void dealloc_layer(layer_dense_t *layer) {
  free(layer->weights);
  free(layer->biases);
  free(layer->output);
}

void forward(layer_dense_t *previous_layer, layer_dense_t *next_layer) {
  layer_output(previous_layer->output, next_layer->weights, next_layer->biases,
               next_layer->input_size, next_layer->output,
               next_layer->output_size, next_layer->callback);
}

double dot_product(double *input, double *weights, double *bias, int input_size,
                   activation_callback callback) {
  double output = 0.0;
  for (int i = 0; i < input_size; i++) {
    output += input[i] * weights[i];
  }

  // Apply activation callback if provided
  if (callback != NULL) {
    callback(&output);
  }

  output += *bias;
  return output;
}

void layer_output(double *input, double *weights, double *bias, int input_size,
                  double *outputs, int output_size,
                  activation_callback callback) {
  for (int i = 0; i < output_size; i++) {
    outputs[i] = dot_product(input, weights + i * input_size, &bias[i],
                             input_size, callback);
  }
}

double activation_sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

double activation_ReLU(double x) { return x < 0.0 ? 0.0 : x; }

double sum_softmax_layer_output(layer_dense_t *output_layer) {
  double sum = 0.0;
  for (int i = 0; i < output_layer->output_size; i++) {
    sum += output_layer->output[i];
  }
  return sum;
}

void activation_softmax(layer_dense_t *output_layer) {
  double sum = 0.0;
  double max_value = output_layer->output[0];

  // Find the maximum value in the output array
  for (int i = 1; i < output_layer->output_size; i++) {
    if (output_layer->output[i] > max_value) {
      max_value = output_layer->output[i];
    }
  }

  // Apply softmax function
  for (int i = 0; i < output_layer->output_size; i++) {
    output_layer->output[i] = exp(output_layer->output[i] - max_value);
    sum += output_layer->output[i];
  }

  // Normalize the output
  for (int i = 0; i < output_layer->output_size; i++) {
    output_layer->output[i] /= sum;
  }
}

// ReLU activation callback
void activation_ReLU_callback(double *output) {
  *output = activation_ReLU(*output);
}

double uniform_distribution(double rangeLow, double rangeHigh) {
  double rng = rand() / (1.0 + RAND_MAX);
  double range = rangeHigh - rangeLow + 1;
  return (rng * range) + rangeLow;
}
