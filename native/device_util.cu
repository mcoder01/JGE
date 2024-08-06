#include "device_util.h"

__host__ __device__ double planeToPointDistance(Plane plane, Vec3d point) {
    return dot(plane.normal, point) - dot(plane.normal, plane.pos);
}

__host__ __device__ double planeLineIntersection(Plane plane, Vec3d line_start, Vec3d line_end) {
    double pd = dot(plane.pos, plane.normal);
    double ad = dot(line_start, plane.normal);
    double bd = dot(line_end, plane.normal);
    return (pd-ad)/(bd-ad);
}

__host__ __device__ void clip(Triangle triangle, Plane plane, device_queue<Triangle> *triangles, bool clipProj) {
    device_vector<Vertex> inside(3), outside(3);

    for (Vertex v : triangle.vertices) {  
        double dist = planeToPointDistance(plane, v.pos);
        if (dist >= 0) inside.push_back(v);
        else outside.push_back(v);
    }

    if (inside.size() == 3) triangles->enqueue(triangle);
    else if (inside.size() == 1 && outside.size() == 2) {
        double t1 = planeLineIntersection(plane, inside.at(0).pos, outside.at(0).pos);
        Vec3d point1 = lerp(inside.at(0).pos, outside.at(0).pos, t1);
        Vec2d tp1 = lerp(inside.at(0).texCoords, outside.at(0).texCoords, t1);

        double t2 = planeLineIntersection(plane, inside.at(0).pos, outside.at(1).pos);
        Vec3d point2 = lerp(inside.at(0).pos, outside.at(1).pos, t2);
        Vec2d tp2 = lerp(inside.at(0).texCoords, outside.at(1).texCoords, t2);

        Vec2d sp1 = lerp(inside.at(0).screenPos, outside.at(0).screenPos, t1);
        Vec2d sp2 = lerp(inside.at(0).screenPos, outside.at(1).screenPos, t2);

        Triangle clipped;
        clipped.vertices[0] = inside.at(0);
        clipped.vertices[1] = {point1, outside.at(0).normal, tp1, sp1};
        clipped.vertices[2] = {point2, outside.at(1).normal, tp2, sp2};
        clipped.isProjected = clipProj;
        triangles->enqueue(clipped);
    } else if (inside.size() == 2 && outside.size() == 1) {
        double t1 = planeLineIntersection(plane, inside.at(0).pos, outside.at(0).pos);
        Vec3d point1 = lerp(inside.at(0).pos, outside.at(0).pos, t1);
        Vec2d tp1 = lerp(inside.at(0).texCoords, outside.at(0).texCoords, t1);

        double t2 = planeLineIntersection(plane, inside.at(1).pos, outside.at(0).pos);
        Vec3d point2 = lerp(inside.at(1).pos, outside.at(0).pos, t2);
        Vec2d tp2 = lerp(inside.at(1).texCoords, outside.at(0).texCoords, t2);

        Vec2d sp1 = lerp(inside.at(0).screenPos, outside.at(0).screenPos, t1);
        Vec2d sp2 = lerp(inside.at(1).screenPos, outside.at(0).screenPos, t2);

        Vertex common = {point1, outside.at(0).normal, tp1, sp1};
        Triangle clipped1;
        clipped1.vertices[0] = inside.at(0);
        clipped1.vertices[1] = inside.at(1);
        clipped1.vertices[2] = common;
        clipped1.isProjected = clipProj;
        triangles->enqueue(clipped1);

        Triangle clipped2;
        clipped2.vertices[0] = inside.at(1);
        clipped2.vertices[1] = common;
        clipped2.vertices[2] = {point2, outside.at(0).normal, tp2, sp2};
        clipped2.isProjected = clipProj;
        triangles->enqueue(clipped2);
    }
}