/**
 * @file dataset.h
 * @brief Header file for generating data for training and testing of neural networks.
 */
#ifndef DATASET_H
#define DATASET_H

#include <cstddef> // for size_t

/**
 * @brief Structure representing spiral data for classification.
 */
typedef struct {
    double *x; /* Holds the x y axis data. Data is formatted as x y x y x y */
    double *y; /* Holds the group the data belongs to. Two steps of x is a single step of y */
} spiral_data_t;

/**
 * @brief Generate spiral data for classification.
 *
 * @param points Number of points to generate per class.
 * @param classes Number of classes to generate.
 * @param data Pointer to the spiral data structure to hold the generated data.
 */
void spiral_data(int points, int classes, spiral_data_t *data);

/**
 * @brief Free the memory allocated for spiral data.
 *
 * @param data Pointer to the spiral data structure.
 */
void dealloc_spiral(spiral_data_t *data);

#endif // DATASET_H
