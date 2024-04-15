#ifndef DEVICE_VECTOR_H
#define DEVICE_VECTOR_H

#include <stdlib.h>
#include "util.h"

template <class T>
class device_vector {
private:
    static const int inc = 10;

    T* values;
    int capacity, length;

    __host__ __device__ void expand();

public:
    __host__ __device__ device_vector(int initial_capacity=inc);
    __host__ __device__ ~device_vector();
    __host__ __device__ void push_back(T);
    __host__ __device__ T pop_back();
    __host__ __device__ T at(int);
    __host__ __device__ int size();
    __host__ __device__ T* get_values();
    __host__ __device__ int get_capacity();
};

extern template __host__ __device__ class device_vector<Triangle>;
extern template __host__ __device__ class device_vector<Vertex>;

#endif