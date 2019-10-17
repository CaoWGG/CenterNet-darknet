//
// Created by yoloCao on 19-8-26.
//

//
// Created by yoloCao on 19-8-14.
//

#include "ctdet_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

layer make_ctdet_layer(int batch, int w, int h , int classes, int size ,int stride,int padding)
{

    layer l = {0};
    l.type = CTDET;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = (4+classes);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.cost = calloc(1, sizeof(float));
    l.outputs = h*w*(4+classes);
    l.inputs = l.outputs;
    l.truths = h*w*(4+classes);
    l.delta = calloc(batch*l.outputs, sizeof(float));
    l.output = calloc(batch*l.outputs, sizeof(float));

    l.pad = padding;
    l.size = size;
    l.stride = stride;
    l.indexes = calloc(batch*l.outputs, sizeof(int));

    l.num_detection = calloc(1, sizeof(int));
    l.forward = forward_ctdet_layer;
    l.backward = backward_ctdet_layer;
#ifdef GPU
    l.num_detection_gpu = cuda_make_int_array(0,1);
    l.forward_gpu = forward_ctdet_layer_gpu;
    l.backward_gpu = backward_ctdet_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
    l.indexes_gpu = cuda_make_int_array(l.indexes, batch*l.outputs);
#endif

    fprintf(stderr, "ctdet \n");
    srand(0);

    return l;
}

void resize_ctdet_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->outputs = h*w*(4+l->classes);
    l->inputs = l->outputs;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta = realloc(l->delta, l->batch*l->outputs*sizeof(float));
    l->indexes = realloc(l->indexes, l->batch*l->outputs*sizeof(int));
#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);
    cudaError_t status = cudaFree(l->indexes_gpu);
    check_error(status);
    l->delta_gpu =     cuda_make_array(l->delta, l->batch*l->outputs);
    l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
    l->indexes_gpu =  cuda_make_int_array(l->indexes, l->batch*l->outputs);
#endif
}

box get_ctdet_box(float *x, int index, int i, int j, int lw, int lh ,int stride)
{
    box b;
    b.x = (x[index + 0*stride]+i) / lw;
    b.y = (x[index + 1*stride]+j) / lh;
    b.w = x[index + 2*stride] / lw;
    b.h = x[index + 3*stride] / lh;
    return b;
}

float delta_ctdet_box(box truth, float *x, int index, int i, int j, int lw, int lh, float *delta, float scale, int stride)
{
    box pred = get_ctdet_box(x, index, i, j, lw, lh,stride);
    float iou = box_iou(pred, truth);

    float tx = truth.x*lw - i;
    float ty = truth.y*lh - j;
    float tw = truth.w*lw;
    float th = truth.h*lh;

    delta[index + 0*stride] = ((tx - x[index + 0*stride])>=0  ? 1:-1);
    delta[index + 1*stride] = ((ty - x[index + 1*stride])>=0  ? 1:-1);
    delta[index + 2*stride] = ((tw - x[index + 2*stride])>=0  ? 1:-1);
    delta[index + 3*stride] = ((th - x[index + 3*stride])>=0  ? 1:-1);
    return iou;
}


static int entry_index(layer l, int batch, int location, int entry)
{
    int n =   location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*(4+l.classes) + entry*l.w*l.h + loc;
}

void forward_ctdet_layer(const layer l, network net)
{
    int i,j,b,cl;
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));
#ifndef GPU

#endif
    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    if(!net.train){
        return;
    }
    float avg_iou = 0;
    float recall = 0;
    float recall75 = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    *(l.cost) = 0;
    for (b = 0; b < l.batch; ++b) {
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i){
                for(cl = 0;cl <l.classes;++cl) {
                    int obj_index = entry_index(l, b, j * l.w + i,cl);
                    float label=net.truth[obj_index];
                    if(label<1) {
                        l.delta[obj_index] = -pow(l.output[obj_index], 3) +
                                             2 * log(1-l.output[obj_index]) *(1 - l.output[obj_index])*
                                             pow(l.output[obj_index], 2) * pow(1-label,4);
                        avg_anyobj += l.output[obj_index];

                    }else if (label==1){
                        l.delta[obj_index] = pow(1 - l.output[obj_index], 3) -
                                             2 * log(l.output[obj_index]) * l.output[obj_index] *
                                             pow(1 - l.output[obj_index], 2);
                        int box_index = entry_index(l, b, j * l.w + i,l.classes);
                        box truth = float_to_box(net.truth +box_index, l.w*l.h);
                        float iou = delta_ctdet_box(truth, l.output, box_index, i, j, l.w, l.h, l.delta, (2-truth.w*truth.h), l.w*l.h);
                        avg_obj += l.output[obj_index];
                        avg_iou+=iou;
                        if(iou > .5) recall += 1;
                        if(iou > .75) recall75 += 1;
                        ++count;
                    }
                }
            }
        }
    }


    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    printf("Region %d Avg IOU: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f,  count: %d\n", net.index , avg_iou/count, avg_obj/count, avg_anyobj/(l.classes*l.w*l.h*l.batch), recall/count, recall75/count, count);
}

void backward_ctdet_layer(const layer l, network net)
{
    axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
}

void correct_ctdet_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i){
        box b = dets[i].bbox;
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw);
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth);
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}

int ctdet_num_detections(layer l, float thresh)
{
    int i;
    int count = 0;
    for (i = 0; i < *(l.num_detection); ++i){
        if (l.output[l.indexes[i]] >= thresh) {
            ++count;
        }
    }
    return count;
}

int get_ctdet_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets)
{
    int i,id,obj_index,x,y,k;
    float *predictions = l.output;
    int count = 0;
    for (i = 0; i < *(l.num_detection); ++i){
        obj_index = l.indexes[i];
        id = obj_index;
        x = id % l.w;
        id /= l.w;
        y = id % l.h;
        id /= l.h;
        k = id % l.classes;
        float objectness = predictions[obj_index];
        if (objectness <= thresh) continue;
        int box_index = entry_index(l, 0, y * l.w + x, l.classes );
        dets[count].bbox = get_ctdet_box(predictions, box_index, x, y, l.w, l.h, l.w * l.h);
        dets[count].objectness = objectness;
        dets[count].classes = l.classes;
        memset(dets[count].prob,0,l.classes* sizeof(float));
        dets[count].prob[k]=objectness;
        ++count;
    }
    correct_ctdet_boxes(dets, count, w, h, netw, neth, relative);
    return count;
}

#ifdef GPU

void forward_ctdet_layer_gpu(const layer l, network net)
{
    copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
    int b ;
    for (b = 0; b < l.batch; ++b){
        int index = entry_index(l, b, 0, 0);
        activate_array_gpu(l.output_gpu + index, l.classes*l.w*l.h, LOGISTIC);
    }
    cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
    if(!net.train || l.onlyforward){
        forward_ctdet_maxpool_layer_gpu(l,net);
        cuda_pull_int_array(l.num_detection_gpu,l.num_detection,1);
        cuda_pull_int_array(l.indexes_gpu,l.indexes,*l.num_detection);
        return;
    }
    cuda_push_array(net.truth_gpu,net.truth,net.truths*net.batch);
    forward_ctdet_loss_layer_gpu(l,net);
}

void backward_ctdet_layer_gpu(const layer l, network net)
{
    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif

