#include "../include/nn.h"

double rand_range(double min, double max) {
    double range = (max - min);
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

void layer_init(layer_dense_t *layer, int input_size, int output_size) {
    layer->input_size = input_size;
    layer->output_size = output_size;

    // Allocate memory for weights
    layer->weights = malloc(sizeof(double) * input_size * output_size);
    if (layer->weights == NULL) {
        printf("Error allocating memory for weights\n");
        return;
    }

    // Allocate memory for biases
    layer->biase = malloc(sizeof(double) * output_size);
    if (layer->biase == NULL) {
        printf("Error allocating memory for biases\n");
        free(layer->weights);
        return;
    }

    // Allocate memory for output
    layer->output = malloc(sizeof(double) * output_size);
    if (layer->output == NULL) {
        printf("Error allocating memory for output\n");
        free(layer->weights);
        free(layer->biase);
        return;
    }

    // Initialize biases with zeros
    for (int i = 0; i < output_size; i++) {
        layer->biase[i] = INIT_BIASES;
    }

    // Initialize weights with random values
    for (int i = 0; i < input_size * output_size; i++) {
        layer->weights[i] = rand_range(RAND_MIN_RANGE, RAND_HIGH_RANGE);
    }
}

void dealloc_layer(layer_dense_t *layer) {
    if (layer->weights != NULL) {
        free(layer->weights);
    }
    if (layer->biase != NULL) {
        free(layer->biase);
    }
    if (layer->output != NULL) {
        free(layer->output);
    }
}

void forward(layer_dense_t *previous_layer, layer_dense_t *next_layer) {
    layer_output(previous_layer->output, next_layer->weights, next_layer->biase, next_layer->input_size, next_layer->output, next_layer->output_size);
}
