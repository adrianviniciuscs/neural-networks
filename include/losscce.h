/**
 * @file categorical_crossentropy.h
 * @brief Functions and structures for categorical cross entropy loss calculation.
 */

#ifndef CATEGORICAL_CROSSENTROPY_H
#define CATEGORICAL_CROSSENTROPY_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
/**
 * @brief Structure representing the Categorical Cross Entropy Loss function.
 */
typedef struct {
    double* (*forward)(double *, int *, int, int); ///< Function pointer to forward function
} Loss_CCE;


/**
 * @brief Calculates the categorical cross entropy loss.
 *
 * This function calculates the categorical cross entropy loss given the predicted
 * probabilities (`y_pred`), true labels (`y_true`), number of samples (`samples`),
 * and number of classes (`num_classes`).
 *
 * @param y_pred Predicted probabilities, a 1D array of size `samples * num_classes`.
 * @param y_true True labels, a 1D array of size `samples`.
 * @param samples Number of samples.
 * @param num_classes Number of classes.
 * @return A dynamically allocated array of size `samples` containing the negative
 * log-likelihoods for each sample.
 *
 * @note The caller is responsible for freeing the memory allocated for the returned array.
 */
double* loss_categorical_crossentropy(double *y_pred, int *y_true, int samples, int num_classes);

/**
 * @brief Creates an instance of Loss_CategoricalCrossentropy.
 *
 * This function allocates memory for Loss_CCE structure and initializes the
 * function pointer to loss_categorical_crossentropy.
 *
 * @return Pointer to the allocated Loss_CCE structure.
 *
 * @note The caller is responsible for freeing the memory allocated for the returned structure.
 */
Loss_CCE* Create_LCCE();

/**
 * @brief Calculates the mean of an array.
 *
 * This function calculates the mean of the elements in the array.
 *
 * @param arr Array of double values.
 * @param length Length of the array.
 * @return Mean value of the array.
 */
double mean(double *arr, int length);

/**
 * @brief Calculates the loss using the provided loss function.
 *
 * This function calculates the loss using the loss function pointed by `loss->forward`.
 *
 * @param loss Loss function instance.
 * @param output Predicted outputs, a 1D array of size `samples * num_classes`.
 * @param y True labels, a 1D array of size `samples`.
 * @param samples Number of samples.
 * @param num_classes Number of classes.
 * @return Mean loss value.
 */
double calculate_loss(Loss_CCE *loss, double *output, int *y, int samples, int num_classes);

#endif /* CATEGORICAL_CROSSENTROPY_H */
