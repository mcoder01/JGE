#include "device_queue.h"

template <class T> __host__ __device__ queue_node<T>::~queue_node() {
    if (next) delete next;
}

template <class T> __host__ __device__ T queue_node<T>::get() {
    return data;
}

template <class T> __host__ __device__ queue_node<T>* queue_node<T>::get_next() {
    return next;
}

template <class T> __host__ __device__ void queue_node<T>::set_next(queue_node<T>* next) {
    this->next = next;
}

template __host__ __device__ class queue_node<Triangle>;

template <class T> __host__ __device__ device_queue<T>::~device_queue() {
    delete head;
}

template <class T> __host__ __device__ void device_queue<T>::enqueue(T data) {
    queue_node<T>* node = new queue_node<T>(data);
    if (tail) {
        tail->set_next(node);
        tail = node;
    } else head = tail = node;
    length++;
}

template <class T> __host__ __device__ T device_queue<T>::dequeue() {
    queue_node<T> *node = head;
    head = node->get_next();
    T value = node->get();
    length--;
    delete node;
    return value;
}

template <class T> __host__ __device__ int device_queue<T>::size() {
    return length;
}

template __host__ __device__ class device_queue<Triangle>;