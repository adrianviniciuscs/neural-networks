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

void spiral_data(int points, int classes, spiral_data_t *data) {
    int ix = 0; // Index for x coordinates
    int iy = 0; // Index for y coordinates

    // Seed the random number generator
    srand(time(NULL));

    // Loop through each class to generate spiral data
    for (int class_number = 0; class_number < classes; class_number++) {
        double r = 0; // Radius
        double t = class_number * 4; // Angle

        // Generate points on the spiral for the current class
        while (r <= 1 && t <= (class_number + 1) * 4) {
            // Add some randomness to t to introduce variation
            double random_t = t + uniform_distribution(-1.0, 1.0) * 0.2;

            // Convert polar coordinates to Cartesian coordinates
            data->x[ix] = r * sin(random_t * 2.5); // x = r * sin(theta)
            data->x[ix + 1] = r * cos(random_t * 2.5); // y = r * cos(theta)

            // Assign the class number as the label
            data->y[iy] = class_number;

            // Increase r and t to move to the next point on the spiral
            r += 1.0 / (points - 1); // Increment r to move along the radius
            t += 4.0 / (points - 1); // Increment t to move along the angle

            // Increment indices to move to the next set of coordinates
            iy++; // Increment y index
            ix += 2; // Increment x index (2 elements per x,y pair)
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

