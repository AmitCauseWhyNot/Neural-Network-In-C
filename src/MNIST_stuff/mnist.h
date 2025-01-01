#ifndef MNIST_H
#define MNIST_H

#include <stdio.h>
#include <stdint.h>

uint32_t convert_to_32int(uint8_t *buff);

double *get_image(char *path, int num);

int get_label(char *path, int num);

#endif