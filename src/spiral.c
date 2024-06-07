#include "../include/dataset.h"

// Code for generating a spiral dataset 
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


