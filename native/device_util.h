#ifndef DEVICE_UTIL_H
#define DEVICE_UTIL_H

#include "device_queue.h"
#include "device_vector.h"
#include "vector_util.h"
#include "util.h"

using namespace geom2d;
using namespace geom3d;

__host__ __device__ double planeToPointDistance(Plane, Vec3d);
__host__ __device__ double planeLineIntersection(Plane, Vec3d, Vec3d);
__host__ __device__ void clip(Triangle, Plane, device_queue<Triangle>*, bool);

#endif