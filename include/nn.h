/**
 * @file nn.h
 * @brief Header file for neural network functions and data structures.
 */

#ifndef NN_H
#define NN_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define RAND_HIGH_RANGE (0.10)  /**< Upper limit for random weight initialization */
#define RAND_MIN_RANGE (-0.10)  /**< Lower limit for random weight initialization */
#define INIT_BIASES (0.0)       /**< Initial value for biases */

/**
 * @brief Activation function callback type definition.
 */
typedef void (*activation_callback)(double *output);

/**
 * @brief Structure representing a dense layer in a neural network.
 */
typedef struct {
    double *weights;             /**< Pointer to the weights array */
    double *biases;               /**< Pointer to the biases array */
    double *output;              /**< Pointer to the output array */
    int input_size;              /**< Size of the input layer */
    int output_size;             /**< Size of the output layer */
    activation_callback callback;/**< Pointer to the activation callback function */
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

/** 
 * @brief Calculate the dot product of a neuron and add the bias, optionally applying an activation callback.
 *
 * @param input Pointer to the input data.
 * @param weights Pointer to the weights of the neuron.
 * @param bias Pointer to the bias of the neuron.
 * @param input_size Number of neurons in the input layer.
 * @param callback Pointer to the activation callback function.
 * @return Output of the neuron after activation.
 */
double dot_product(double *input, double *weights, double *bias, int input_size, activation_callback callback);

/** 
 * @brief Calculate the dot products of each neuron in a layer and add the bias, storing them in an output array.
 *
 * @param input Pointer to the input data.
 * @param weights Pointer to the weights of the neurons.
 * @param bias Pointer to the biases of the neurons.
 * @param input_size Number of neurons in the input layer.
 * @param outputs Pointer to the output array.
 * @param output_size Number of neurons in the output layer.
 * @param callback Pointer to the activation callback function.
 */
void layer_output(double *input, double *weights, double *bias, int input_size, double *outputs, int output_size, activation_callback callback);

/** 
 * @brief Sigmoid activation function.
 *
 * @param x Input value.
 * @return Output of the sigmoid function.
 */
double activation_sigmoid(double x);

/** 
 * @brief ReLU activation function.
 *
 * @param x Input value.
 * @return Output of the ReLU function.
 */
double activation_ReLU(double x);

/** 
 * @brief Apply ReLU activation function to the output of a node.
 *
 * @param output Pointer to the output value.
 */
void activation_ReLU_callback(double *output);

/**
 * @brief Apply softmax activation function to the output layer.
 *
 * Softmax function normalizes the output of a layer into a probability distribution
 * over the classes.
 *
 * @param output_layer Pointer to the dense layer structure containing the output to be normalized.
 */
void activation_softmax(layer_dense_t *output_layer);

/**
 * @brief Calculate the sum of the output values of a dense layer.
 *
 * @param output_layer Pointer to the dense layer structure containing the output values.
 * @return Sum of the output values.
 */
double sum_softmax_layer_output(layer_dense_t *output_layer);


/** 
 * @brief Generate a random number within a uniform distribution range.
 *
 * @param rangeLow Lower bound of the range.
 * @param rangeHigh Upper bound of the range.
 * @return Random number within the specified range.
 */
double uniform_distribution(double rangeLow, double rangeHigh);
#endif /* NN_H */

