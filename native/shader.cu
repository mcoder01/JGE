#include <stdio.h>
#include <iostream>

#include "com_mcoder_jge_g3d_render_Pipeline.h"
#include "device_util.h"

using namespace std;

__global__ void clip_triangle(Triangle *in_triangle, Plane *in_planes, int *num_planes, bool *clipProj, Triangle *out_triangles, int *out_size) {
    device_queue<Triangle> *currQueue = new device_queue<Triangle>();

    currQueue->enqueue(*in_triangle);
    for (int i = 0; i < *num_planes; i++) {
        device_queue<Triangle> *nextQueue = new device_queue<Triangle>();
        while(currQueue->size() > 0)
            clip(currQueue->dequeue(), in_planes[i], nextQueue, *clipProj);

        delete currQueue;
        currQueue = nextQueue;
    }

    *out_size = currQueue->size();
    for (int i = 0; i < *out_size; i++)
        out_triangles[i] = currQueue->dequeue();

    delete currQueue;
}

JNIEXPORT jobjectArray JNICALL Java_com_mcoder_jge_g3d_render_Pipeline_clipTriangle(JNIEnv *env, jobject thisObj, jobject in_triangle, jobjectArray in_planes, jboolean clipProj) {
    cudaDeviceSetLimit(cudaLimitStackSize, 8192);
    initClasses(env);
    
    Triangle triangle = parseTriangle(env, in_triangle, clipProj);
    Plane *planes = parsePlaneArray(env, in_planes);

    int planes_num = env->GetArrayLength(in_planes);
    const int max_clipped_triangles = pow(2, planes_num);

    /*int size;
    Triangle* clipped = (Triangle*) malloc(sizeof(Triangle)*max_clipped_triangles);
    clip_triangle(&triangle, planes, &planes_num, (bool*) &clipProj, clipped, &size);*/

    Triangle *gpu_triangle, *gpu_clipped;
    Plane *gpu_planes;
    bool *gpu_clipProj;
    int *gpu_planes_num, *gpu_out_size;
    cudaMalloc((void**) &gpu_triangle, sizeof(Triangle));
    cudaMalloc((void**) &gpu_planes, sizeof(Plane)*planes_num);
    cudaMalloc((void**) &gpu_planes_num, sizeof(int));
    cudaMalloc((void**) &gpu_clipProj, sizeof(bool));
    cudaMalloc((void**) &gpu_clipped, sizeof(Triangle)*max_clipped_triangles);
    cudaMalloc((void**) &gpu_out_size, sizeof(int));

    cudaMemcpy(gpu_triangle, &triangle, sizeof(Triangle), cudaMemcpyDefault);
    cudaMemcpy(gpu_planes, planes, sizeof(Plane)*planes_num, cudaMemcpyDefault);
    cudaMemcpy(gpu_planes_num, &planes_num, sizeof(int), cudaMemcpyDefault);
    cudaMemcpy(gpu_clipProj, &clipProj, sizeof(jboolean), cudaMemcpyDefault);

    clip_triangle<<<1,1>>>(gpu_triangle, gpu_planes, gpu_planes_num, gpu_clipProj, gpu_clipped, gpu_out_size);
    cudaDeviceSynchronize();

    int size;
    cudaMemcpy(&size, gpu_out_size, sizeof(int), cudaMemcpyDefault);

    int arraySize = size*sizeof(Triangle);
    Triangle clipped[size];
    cudaMemcpy(clipped, gpu_clipped, arraySize, cudaMemcpyDefault);

    cudaFree(gpu_triangle);
    cudaFree(gpu_planes);
    cudaFree(gpu_planes_num);
    cudaFree(gpu_clipped);
    cudaFree(gpu_out_size);

    return triangleArrayToJobject(env, clipped, size);
}