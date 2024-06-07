#ifndef NN_H
#define NN_H

#include <stdlib.h>
#include <stdio.h>
#include "utils.h"

#define RAND_HIGH_RANGE (0.10)
#define RAND_MIN_RANGE (-0.10)
#define INIT_BIASES (0.0)

/**
 * @brief Structure representing a dense layer in a neural network.
 */
typedef struct {
    double *weights;    /**< Pointer to the weights array */
    double *biase;      /**< Pointer to the biases array */
    double *output;     /**< Pointer to the output array */
    int input_size;     /**< Size of the input layer */
    int output_size;    /**< Size of the output layer */
} layer_dense_t;

/** 
 * @brief Generate a random floating point number within a specified range.
 *
 * @param min Minimum value of the range.
 * @param max Maximum value of the range.
 * @return Random number within the specified range.
 */
double rand_range(double min, double max);

/** 
 * @brief Setup a layer with random weights and biases, and allocate memory for the storage buffers.
 *
 * @param layer Pointer to an empty layer structure.
 * @param input_size Size of the input layer.
 * @param output_size Size of the output layer.
 */
void layer_init(layer_dense_t *layer, int input_size, int output_size);

/** 
 * @brief Free the memory allocated by a layer.
 *
 * @param layer Pointer to a layer structure.
 */
void dealloc_layer(layer_dense_t *layer);

/** 
 * @brief Perform a forward pass in the network from one layer to the next.
 *
 * @param previous_layer Pointer to the previous layer struct.
 * @param next_layer Pointer to the next layer struct.
 */
void forward(layer_dense_t *previous_layer, layer_dense_t *next_layer);

#endif /* NN_H */


