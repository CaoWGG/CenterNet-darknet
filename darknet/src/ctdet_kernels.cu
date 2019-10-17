#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "ctdet_layer.h"
#include "blas.h"
#include "cuda.h"
#include "utils.h"
}


__global__ void forward_ctdet_loss_layer_kernel(int n, int in_h, int in_w,int classes,float *output, float *delta, float *truth,
        float hm_weight, float off_weight, float wh_weight,float * mertic)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;
    int i = id % in_w;
    id /= in_w;
    int j = id % in_h;
    id /= in_w;
    int k = id % classes;
    id /= classes;
    int b = id;
    float label;
    int stride = in_w*in_h;
    float x,y,w,h,tx,ty,tw,th,predx,predy,predx1,predy1,inter,uino,iou;
    int index = b*in_w*in_h*(classes+4) + k*in_w*in_h + j*in_w+i;
    label=truth[index];
    float alpha = 0.5;
    float pt = output[index] + 0.000000000000001F;
    float grad = -(1 - pt) * (2 * pt*logf(pt) + pt - 1);
    if(label< 1)
    {
        delta[index] = (-output[index])*grad*pow(1-label,4)*hm_weight;
    } else if(label == 1)
    {
        delta[index] = alpha*(1-output[index])*grad*hm_weight;
        int box_index = b*in_w*in_h*(classes+4) + classes*in_w*in_h + j*in_w+i;
        x = truth[box_index+0];
        y = truth[box_index+1*stride];
        w = truth[box_index+2*stride];
        h = truth[box_index+3*stride];
        tx = (x*in_w - i);
        ty = (y*in_h - j);
        tw = w*in_w;
        th = h*in_h;
        delta[box_index + 0*stride] = off_weight*((tx - output[box_index + 0*stride]) >=0 ? 1:-1);
        delta[box_index + 1*stride] = off_weight*((ty - output[box_index + 1*stride]) >=0 ? 1:-1);
        delta[box_index + 2*stride] = wh_weight*((tw - output[box_index + 2*stride]) >=0 ? 1:-1);
        delta[box_index + 3*stride] = wh_weight*((th - output[box_index + 3*stride]) >=0 ? 1:-1);
        predx = output[box_index + 0*stride] +i -output[box_index + 2*stride]/2;
        predy = output[box_index + 1*stride] +j -output[box_index + 3*stride]/2;
        predx1 = predx+output[box_index + 2*stride];
        predy1 = predy+output[box_index + 3*stride];
        uino = tw*th + output[box_index + 2*stride]*output[box_index + 3*stride];
        x = x*in_w-tw/2;y = y*in_h - th/2; w = x+tw;h=y+th;
        inter = (fmin(predx1,w)-fmax(predx,x))*(fmin(predy1,h)-fmax(predy,y));
        iou = inter / (uino - inter +1e-10);
        atomicAdd(mertic,output[index]);
        atomicAdd(mertic+1,iou);
        atomicAdd(mertic+2,1.);
    }
}

extern "C" void forward_ctdet_loss_layer_gpu( layer l, network net)
{

    int h = l.out_h;
    int w = l.out_w;
    size_t n = h*w*l.classes*l.batch;
    float show[3]={0};
    float *show_gpu=cuda_make_array(show,3);
    cudaMemset(l.delta_gpu,0,l.batch*l.outputs* sizeof(float));
    forward_ctdet_loss_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, l.h, l.w,l.classes ,l.output_gpu,l.delta_gpu,net.truth_gpu,l.hm_weight,l.off_weight,l.wh_weight,show_gpu);
    cuda_pull_array(show_gpu,show,3);
    cuda_pull_array(l.delta_gpu,l.delta,l.batch*l.outputs);
    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    printf("Region %d   Obj: %f, Avg IOU: %f, count: %d\n", net.index , show[0]/show[2],show[1]/show[2],(int)show[2]);
    cudaFree(show_gpu);
    check_error(cudaPeekAtLastError());
}

__global__ void forward_ctdet_maxpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride, int size, int pad, float *input,int *indexes,int* numdet)
{
    int h = (in_h + pad - size)/stride + 1;
    int w = (in_w + pad - size)/stride + 1;
    int c = in_c;
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;
    int i = id % w;
    id /= w;
    int j = id % h;
    id /= h;
    int k = id % c;
    id /= c;
    int b = id;
    int w_offset = -pad/2;
    int h_offset = -pad/2;
    int out_index = b*in_w*in_h*(c+4) + k*in_w*in_h + j*in_w+i;
    float max = -INFINITY;
    int max_i = -1;
    int l, m;
    for(l = 0; l < size; ++l){
        for(m = 0; m < size; ++m){
            int cur_h = h_offset + j*stride + l;
            int cur_w = w_offset + i*stride + m;
            int index = b*in_w*in_h*(c+4) + k*in_w*in_h + cur_h*in_w+cur_w;
            int valid = (cur_h >= 0 && cur_h < in_h &&
                         cur_w >= 0 && cur_w < in_w);
            float val = (valid != 0) ? input[index] : -INFINITY;
            max_i = (val > max) ? index : max_i;
            max   = (val > max) ? val   : max;
        }
    }
    if (max_i == out_index && max > 0.001){
        int resCount = (int)atomicAdd(numdet,1);
        indexes[resCount]=out_index;
    }
}


extern "C" void forward_ctdet_maxpool_layer_gpu(layer l, network net)
{
    int h = l.out_h;
    int w = l.out_w;
    size_t n = h*w*l.classes*l.batch;
    cudaError_t status=cudaMemset(l.num_detection_gpu,0, sizeof(int));
    check_error(status);
    forward_ctdet_maxpool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, l.h, l.w, l.classes, l.stride, l.size, l.pad, l.output_gpu, l.indexes_gpu ,l.num_detection_gpu);
    check_error(cudaPeekAtLastError());
}