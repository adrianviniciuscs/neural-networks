#include "../include/dataset.h"
#include "../include/nn.h"
#include "../include/losscce.h"  
#include <stdio.h>
#include <stdlib.h>

#define NET_BATCH_SIZE 100
#define NET_INPUT_LAYER_1_SIZE 2
#define NET_INPUT_LAYER_2_SIZE 3
#define NET_OUTPUT_LAYER_SIZE 3

int main() {

  int i, j;
  spiral_data_t X_data;
  layer_dense_t X, dense1, dense2;
  
  // Generate spiral data
  spiral_data(100, 3, &X_data);
  if (X_data.x == NULL) {
    printf("Data is null\n");
    return 0;
  }

  X.callback = NULL;
  dense1.callback = activation_ReLU_callback;
  dense2.callback = NULL;

  // Initialize layers
  layer_init(&dense1, NET_INPUT_LAYER_1_SIZE, NET_INPUT_LAYER_2_SIZE);
  layer_init(&dense2, NET_INPUT_LAYER_2_SIZE, NET_OUTPUT_LAYER_SIZE);

  // Loss function initialization
  Loss_CCE *loss_function = Create_LCCE();

  for (i = 0; i < NET_BATCH_SIZE; i++) {
    X.output = &X_data.x[i * 2];

    // Forward pass through layer 1
    forward(&X, &dense1);
    printf("Batch: %d\n", i);
    printf("\tLayer 1 Output: ");
    for (j = 0; j < dense1.output_size; j++) {
      printf("%f\t", dense1.output[j]);
    }
    printf("\n");

    // Forward pass through layer 2
    forward(&dense1, &dense2);
    printf("\tLayer 2 Output: ");
    for (j = 0; j < dense2.output_size; j++) {
      printf("%f\t", dense2.output[j]);
    }
    printf("\n");

    // Apply softmax activation to layer 2
    activation_softmax(&dense2);
    printf("\tSoftmax (Layer 2): ");
    for (j = 0; j < dense2.output_size; j++) {
      printf("%f\t", dense2.output[j]);
    }
    printf("\n");

    // Calculate loss
    int yValue = (int) *(X_data.y) + 1; // type sorcery 
    double loss = calculate_loss(loss_function, dense2.output, &yValue, 1, NET_OUTPUT_LAYER_SIZE);
    printf("\tLoss: %f\n", loss);

    // Print normalized sum of layer 2 output
    printf("\tLayer 2 Normalized Sum: %f\n", sum_softmax_layer_output(&dense2));
    printf("\n");
  }

  // Deallocate memory
  dealloc_layer(&dense1);
  dealloc_layer(&dense2);
  dealloc_spiral(&X_data);

  // Free loss function
  free(loss_function);

  return 0;
}

