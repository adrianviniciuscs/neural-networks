#include "../include/losscce.h"
#include <math.h>
#include <stdlib.h>

// Function to calculate categorical cross entropy
double* loss_categorical_crossentropy(double *y_pred, int *y_true, int samples, int num_classes) {
    double *negative_log_likelihoods = (double *)malloc(samples * sizeof(double));
    double *y_pred_clipped = (double *)malloc(num_classes * sizeof(double));

    for (int i = 0; i < samples; ++i) {
        for (int j = 0; j < num_classes; ++j) {
            y_pred_clipped[j] = fmax(1e-7, fmin(y_pred[i * num_classes + j], 1.0 - 1e-7));
        }

        double correct_confidence = 0.0;
        if (y_true[i] >= 0 && y_true[i] < num_classes) {
            correct_confidence = y_pred_clipped[y_true[i]];
        }

        negative_log_likelihoods[i] = -log(correct_confidence);
    }

    free(y_pred_clipped);
    return negative_log_likelihoods;
}

Loss_CCE* Create_LCCE() {
    Loss_CCE *loss = (Loss_CCE *)malloc(sizeof(Loss_CCE));
    loss->forward = loss_categorical_crossentropy;
    return loss;
}

double mean(double *arr, int length) {
    double sum = 0.0;
    for (int i = 0; i < length; ++i) {
        sum += arr[i];
    }
    return sum / length;
}

double calculate_loss(Loss_CCE *loss, double *output, int *y, int samples, int num_classes) {
    double *sample_losses = loss->forward(output, y, samples, num_classes);
    double data_loss = mean(sample_losses, samples);
    free(sample_losses);
    return data_loss;
}
