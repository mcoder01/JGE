#include "device_vector.h"

template <class T> __host__ __device__ void device_vector<T>::expand() {
    capacity += inc;
    T* new_values = (T*) malloc(sizeof(T)*capacity);
    for (int i = 0; i < length; i++)
        new_values[i] = values[i];

    delete values;
    values = new_values;
}

template <class T> __host__ __device__ device_vector<T>::device_vector(int initial_capacity) {
    values = (T*) malloc(sizeof(T)*initial_capacity);
    capacity = initial_capacity;
    length = 0;
}

template <class T> __host__ __device__ device_vector<T>::~device_vector() {
    free(values);
}
    
template <class T> __host__ __device__ void device_vector<T>::push_back(T value) {
    if (length == capacity)
        expand();
    values[length++] = value;
}

template <class T> __host__ __device__ T device_vector<T>::pop_back() {
    return values[--length];
}
    
template <class T> __host__ __device__ T device_vector<T>::at(int index) {
    return values[index];
}

template <class T> __host__ __device__ int device_vector<T>::size() {
    return length;
}

template <class T> __host__ __device__ T* device_vector<T>::get_values() {
    return values;
}

template <class T> __host__ __device__ int device_vector<T>::get_capacity() {
    return capacity;
}

template __host__ __device__ class device_vector<Triangle>;
template __host__ __device__ class device_vector<Vertex>;