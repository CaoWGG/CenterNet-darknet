//
// Created by yoloCao on 19-8-26.
//

#ifndef CTDET_LAYER_H
#define CTDET_LAYER_H
#include "darknet.h"
#include "layer.h"
#include "network.h"

layer make_ctdet_layer(int batch, int w, int h , int classes, int size ,int stride,int padding);
void forward_ctdet_layer(const layer l, network net);
void backward_ctdet_layer(const layer l, network net);
void resize_ctdet_layer(layer *l, int w, int h);
int ctdet_num_detections(layer l, float thresh);
int get_ctdet_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets);
#ifdef GPU
void forward_ctdet_layer_gpu(const layer l, network net);
void forward_ctdet_loss_layer_gpu( layer l, network net);
void forward_ctdet_maxpool_layer_gpu(layer l, network net);
void backward_ctdet_layer_gpu(layer l, network net);
#endif

#endif //CTDET_LAYER_H
