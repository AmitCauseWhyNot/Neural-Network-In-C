#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "mnist.h"

uint32_t convert_to_32int(uint8_t *buff)
{
    return (buff[0] << 24) | (buff[1] << 16) | (buff[2] << 8) | buff[3];
}

double *get_image(char *path, int num)
{
    uint8_t buffer[4]; // set a 4 size buffer to read magic number.
    int magic_num, num_images, buff_length = 4, img_buff_length = 784;
    FILE *f;

    f = fopen(path, "rb");
    if (!f)
    {
        perror("OPEN FILE ERROR");
        return NULL;
    }

    fread(buffer, sizeof(uint8_t), buff_length, f); // MAGIC NUMBER.
    magic_num = convert_to_32int(buffer);

    if (magic_num != 2051)
    {
        perror("error reading file (magic num error)");
        fclose(f);
        return NULL;
    }

    fread(buffer, sizeof(uint8_t), buff_length, f); // NUM IMAGES.
    num_images = convert_to_32int(buffer);

    if (num > num_images || !num_images > num)
    {
        perror("num out of bounds");
        return NULL;
    }

    int skip_length = buff_length * 2 + img_buff_length * num;
    if (fseek(f, skip_length, SEEK_CUR) != 0)
    {
        perror("fseek error, get_image");
        fclose(f);
        return NULL;
    }

    uint8_t *image_buff = malloc(sizeof(uint8_t) * img_buff_length);
    if (fread(image_buff, sizeof(uint8_t), img_buff_length, f) != img_buff_length)
    {
        perror("requested image fread error");
        fclose(f);
        return NULL;
    }
    fclose(f);

    double *norm_image_buff = malloc(sizeof(double) * img_buff_length);

    for (int i = 0; i < img_buff_length; i++)
    {
        norm_image_buff[i] = image_buff[i] / 255.0;
    }

    return norm_image_buff;
}

int get_label(char *path, int num)
{
    uint8_t buffer[4], *label = malloc(sizeof(uint8_t));
    int magic_number, num_labels, buff_length = 4;
    FILE *f;

    f = fopen(path, "rb");
    if (!f)
    {
        perror("Error opening file");
        return -1;
    }

    fread(buffer, sizeof(uint8_t), buff_length, f);
    magic_number = convert_to_32int(buffer);

    fread(buffer, sizeof(uint8_t), buff_length, f);
    num_labels = convert_to_32int(buffer);

    if (num > num_labels || !num_labels > num)
    {
        perror("num out of bounds error");
        fclose(f);
        return -1;
    }

    int requested_label_num = 8 + num;
    if (fseek(f, requested_label_num, SEEK_SET) != 0)
    {
        perror("seek error label");
        fclose(f);
        return -1;
    }

    if (fread(label, sizeof(uint8_t), 1, f) != 1)
    {
        perror("error reading label");
        fclose(f);
        return -1;
    }

    int label_value = *label;
    free(label);
    fclose(f);
    return label_value;
}