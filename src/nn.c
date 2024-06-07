#include "../include/nn.h"

double rand_range(double min, double max)
{
    double range = (max - min);
    double div = RAND_MAX / range;
    return min + (rand() / div);
}
void layer_init(layer_dense_t *layer, int input_size, int output_size)
{
    layer->input_size = input_size;
    layer->output_size = output_size;

    // Allocate memory for weights, biases, and output
    layer->weights = malloc(sizeof(double) * input_size * output_size);
    if (layer->weights == NULL) {
        fprintf(stderr, "Error: Unable to allocate memory for weights\n");
        return;
    }

    layer->biase = malloc(sizeof(double) * output_size);
    if (layer->biase == NULL) {
        fprintf(stderr, "Error: Unable to allocate memory for biases\n");
        free(layer->weights);
        return;
    }

    layer->output = malloc(sizeof(double) * output_size);
    if (layer->output == NULL) {
        fprintf(stderr, "Error: Unable to allocate memory for output\n");
        free(layer->weights);
        free(layer->biase);
        return;
    }

    // Initialize biases with INIT_BIASES
    for (int i = 0; i < output_size; i++) {
        layer->biase[i] = INIT_BIASES;
    }

    // Initialize weights with random values
    for (int i = 0; i < (input_size * output_size); i++) {
        layer->weights[i] = rand_range(RAND_MIN_RANGE, RAND_HIGH_RANGE);
    }
}

void dealloc_layer(layer_dense_t *layer)
{
    if (layer->weights != NULL) {
        free(layer->weights);
        layer->weights = NULL;
    }
    if (layer->biase != NULL) {
        free(layer->biase);
        layer->biase = NULL;
    }
    if (layer->output != NULL) {
        free(layer->output);
        layer->output = NULL;
    }
}

void forward(layer_dense_t *previous_layer, layer_dense_t *next_layer)
{
    layer_output(previous_layer->output, next_layer->weights, next_layer->biase,
                 next_layer->input_size, next_layer->output, next_layer->output_size, next_layer->callback);
}

double dot_product(double *input, double *weights, double *bias, int input_size, activation_callback callback)
{
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

void layer_output(double *input, double *weights, double *bias, int input_size, double *outputs, int output_size, activation_callback callback)
{
    int offset = 0;
    for (int i = 0; i < output_size; i++) {
        outputs[i] = dot_product(input, weights + offset, &bias[i], input_size, callback);
        offset += input_size;
    }
}

double activation_sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double activation_ReLU(double x)
{
    if (x < 0.0) {
        return 0.0;
    }
    return x;
}

// ReLU activation callback
void activation_ReLU_callback(double *output)
{
    *output = activation_ReLU(*output);
}

double uniform_distribution(double rangeLow, double rangeHigh)
{
    double rng = rand() / (1.0 + RAND_MAX);
    double range = rangeHigh - rangeLow + 1;
    double rng_scaled = (rng * range) + rangeLow;
    return rng_scaled;
}

void spiral_data(int points, int classes, spiral_data_t *data)
{
    data->x = (double*)malloc(sizeof(double) * points * classes * 2);
    if (data->x == NULL) {
        fprintf(stderr, "Error: Unable to allocate memory for data.x\n");
        return;
    }

    data->y = (double*)malloc(sizeof(double) * points * classes);
    if (data->y == NULL) {
        fprintf(stderr, "Error: Unable to allocate memory for data.y\n");
        free(data->x);
        return;
    }

    int ix = 0;
    int iy = 0;

    for (int class_number = 0; class_number < classes; class_number++) {
        double r = 0;
        double t = class_number * 4;

        while (r <= 1 && t <= (class_number + 1) * 4) {
            // Adding some randomness to t
            double random_t = t + uniform_distribution(-1.0, 1.0) * 0.2;

            // Converting from polar to Cartesian coordinates
            data->x[ix] = r * sin(random_t * 2.5);
            data->x[ix + 1] = r * cos(random_t * 2.5);

            data->y[iy] = class_number;

            // The below two statements achieve linspace-like functionality
            r += 1.0 / (points - 1);
            t += 4.0 / (points - 1);

            iy++;
            ix += 2; // increment index
        }
    }
}

void dealloc_spiral(spiral_data_t *data)
{
    if (data->x != NULL) {
        free(data->x);
        data->x = NULL;
    }

    if (data->y != NULL) {
        free(data->y);
        data->y = NULL;
    }
}

