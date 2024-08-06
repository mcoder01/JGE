#ifndef DEVICE_QUEUE_H
#define DEVICE_QUEUE_H

#include "util.h"

template <class T>
class queue_node {
private:
    T data;
    queue_node<T> *next;

public:
    __host__ __device__ queue_node(T data):data(data),next(nullptr) {}
    __host__ __device__ ~queue_node();
    __host__ __device__ T get();
    __host__ __device__ queue_node<T> *get_next();
    __host__ __device__ void set_next(queue_node<T>*);
};

extern template __host__ __device__ class queue_node<Triangle>;

template <class T>
class device_queue {
private:
    queue_node<T> *head, *tail;
    int length;

public:
    __host__ __device__ device_queue():head(nullptr),tail(nullptr),length(0) {}
    __host__ __device__ ~device_queue();
    __host__ __device__ void enqueue(T);
    __host__ __device__ T dequeue();
    __host__ __device__ int size();
};

extern template __host__ __device__ class device_queue<Triangle>;

#endif