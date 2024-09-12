#include "../include/dataset.h"
#include "../include/nn.h"
#include <time.h>
#include <stdlib.h>
#include <math.h>

void spiral_data(int points, int classes, spiral_data_t *data) {
    int total_points = points * classes;
    data->x = (double *)malloc(total_points * 2 * sizeof(double)); // Allocate memory for x
    data->y = (double *)malloc(total_points * sizeof(double)); // Allocate memory for y

    if (data->x == NULL || data->y == NULL) {
        printf("Memory allocation failed.\n");
        return;
    }

    int index_x = 0; // Index for x coordinates
    int index_y = 0; // Index for y coordinates

    // Seed the random number generator
    srand(time(NULL));

    // Loop through each class to generate spiral data
    for (int class_num = 0; class_num < classes; class_num++) {
        double radius = 0; // Radius
        double angle = class_num * 4; // Angle

        // Generate points on the spiral for the current class
        while (radius <= 1 && angle <= (class_num + 1) * 4) {
            // Add some randomness to angle to introduce variation
            double random_angle = angle + uniform_distribution(-1.0, 1.0) * 0.2;

            // Convert polar coordinates to Cartesian coordinates
            data->x[index_x] = radius * sin(random_angle * 2.5); // x = r * sin(theta)
            data->x[index_x + 1] = radius * cos(random_angle * 2.5); // y = r * cos(theta)

            // Assign the class number as the label
            data->y[index_y] = class_num;

            // Increase radius and angle to move to the next point on the spiral
            radius += 1.0 / (points - 1); // Increment radius to move along the radius
            angle += 4.0 / (points - 1); // Increment angle to move along the angle

            // Increment indices to move to the next set of coordinates
            index_y++; // Increment y index
            index_x += 2; // Increment x index (2 elements per x,y pair)
        }
    }
}

void dealloc_spiral(spiral_data_t *data) {
    if (data->x != NULL) {
        free(data->x);
        data->x = NULL;
    }

    if (data->y != NULL) {
        free(data->y);
        data->y = NULL;
    }
}
