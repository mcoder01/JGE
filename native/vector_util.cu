#include "vector_util.h"

using namespace geom2d;
using namespace geom3d;

__host__ __device__ Vec2d geom2d::add(Vec2d v1, Vec2d v2) {
    v1.x += v2.x;
    v1.y += v2.y;
    return v1;
}

__host__ __device__ Vec2d geom2d::subtract(Vec2d v1, Vec2d v2) {
    return add(v1, scale(v2, -1));
}

__host__ __device__ Vec2d geom2d::scale(Vec2d v, double m) {
    v.x *= m;
    v.y *= m;
    return v;
}

__host__ __device__ double geom2d::dot(Vec2d v1, Vec2d v2) {
    return v1.x*v2.x+v1.y*v2.y;
}

__host__ __device__ Vec2d geom2d::lerp(Vec2d start, Vec2d end, double t) {
    return add(scale(subtract(end, start), t), start);
}

__host__ __device__ Vec3d geom3d::add(Vec3d v1, Vec3d v2) {
    v1.x += v2.x;
    v1.y += v2.y;
    v1.z += v2.z;
    return v1;
}

__host__ __device__ Vec3d geom3d::subtract(Vec3d v1, Vec3d v2) {
    return add(v1, scale(v2, -1));
}

__host__ __device__ Vec3d geom3d::scale(Vec3d v, double m) {
    v.x *= m;
    v.y *= m;
    v.z *= m;
    return v;
}

__host__ __device__ double geom3d::dot(Vec3d v1, Vec3d v2) {
    return v1.x*v2.x+v1.y*v2.y+v1.z*v2.z;
}

__host__ __device__ Vec3d geom3d::lerp(Vec3d start, Vec3d end, double t) {
    return add(scale(subtract(end, start), t), start);
}