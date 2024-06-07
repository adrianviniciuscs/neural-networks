#ifndef UTILS_H
#define UTILS_H

/**
 * @brief Calculate the dot product of two vectors and add a bias.
 *
 * This function calculates the dot product of two vectors of the same size,
 * and then adds a bias to the result.
 *
 * @param vec1 Pointer to the first vector.
 * @param vec2 Pointer to the second vector.
 * @param bias Pointer to the bias value to add.
 * @param size Size of the vectors.
 * @return Dot product of the two vectors plus the bias.
 */
double dot_product(const double *vec1, const double *vec2, double *bias, int size);

/**
 * @brief Perform a layer output calculation.
 *
 * This function calculates the output of a layer given inputs, weights, biases,
 * and stores the result in an outputs array.
 *
 * @param input Pointer to the input array.
 * @param weights Pointer to the weights array.
 * @param bias Pointer to the bias array.
 * @param input_size Size of each input vector.
 * @param outputs Pointer to the outputs array.
 * @param output_size Number of outputs to calculate.
 */
void layer_output(double *input, double *weights, double *bias, int input_size,
                  double *outputs, int output_size);

#endif // UTILS_H
