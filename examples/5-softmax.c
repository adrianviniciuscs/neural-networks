
#include "../include/dataset.h"
#include "../include/nn.h"
#include <stdio.h>
#include <stdlib.h>

#define NET_BATCH_SIZE 300
#define NET_INPUT_LAYER_1_SIZE 2
#define NET_INPUT_LAYER_2_SIZE 3
#define NET_OUTPUT_LAYER_SIZE 3

int main() {

  // seed the random values
  srand(0);

  int i = 0;
  int j = 0;
  spiral_data_t X_data;
  layer_dense_t X;
  layer_dense_t dense1;
  layer_dense_t dense2;

  spiral_data(100, 3, &X_data);
  if (X_data.x == NULL) {
    printf("data null\n");
    return 0;
  }

  X.callback = NULL;

  dense1.callback = activation_ReLU_callback;

  dense2.callback = NULL;

  layer_init(&dense1, NET_INPUT_LAYER_1_SIZE, NET_INPUT_LAYER_2_SIZE);
  layer_init(&dense2, NET_INPUT_LAYER_2_SIZE, NET_OUTPUT_LAYER_SIZE);

  for (i = 0; i < NET_BATCH_SIZE; i++) {
    X.output = &X_data.x[i * 2];
    forward(&X, &dense1);

    printf("Batch: %d \n \t Layer1 -> ", i);
    for (j = 0; j < dense1.output_size; j++) {
      printf("%f \t ", dense1.output[j]);
    }
    printf("\n");

    forward(&dense1, &dense2);

    printf("\t Layer2 -> ");
    for (j = 0; j < dense2.output_size; j++) {
      printf("%f \t ", dense2.output[j]);
    }
    printf("\n");

    activation_softmax(&dense2);

    printf("\t Softmax (Layer 2) -> ");
    for (j = 0; j < dense2.output_size; j++) {
      printf("%f \t ", dense2.output[j]);
    }
    printf("\n");

    printf("\tLayer 2 Normalized Sum: %f\n", sum_softmax_layer_output(&dense2));
    printf("\n");
  }

  dealloc_layer(&dense1);
  dealloc_layer(&dense2);
  dealloc_spiral(&X_data);
  return 0;
}
