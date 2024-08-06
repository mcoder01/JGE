#ifndef VECTOR_UTIL_H
#define VECTOR_UTIL_H

namespace geom2d {
    typedef struct {
        double x, y;
    } Vec2d;

    __host__ __device__ Vec2d add(Vec2d, Vec2d);
    __host__ __device__ Vec2d subtract(Vec2d, Vec2d);
    __host__ __device__ Vec2d scale(Vec2d, double);
    __host__ __device__ double dot(Vec2d, Vec2d);
    __host__ __device__ Vec2d lerp(Vec2d, Vec2d, double);
};

namespace geom3d {
    typedef struct {
        double x, y, z;
    } Vec3d;

    __host__ __device__ Vec3d add(Vec3d, Vec3d);
    __host__ __device__ Vec3d subtract(Vec3d, Vec3d);
    __host__ __device__ Vec3d scale(Vec3d, double);
    __host__ __device__ double dot(Vec3d, Vec3d);
    __host__ __device__ Vec3d lerp(Vec3d, Vec3d, double);
};

#endif