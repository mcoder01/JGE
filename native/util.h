#ifndef SHADER_H
#define SHADER_H

#include <jni.h>
#include <vector>
#include <stdlib.h>
#include "vector_util.h"

using namespace geom2d;
using namespace geom3d;

typedef struct {
    int pointIndex;
    int texIndex;
} OBJIndex;

typedef struct {
    Vec3d pos, normal;
    Vec2d texCoords, screenPos;
} Vertex;

typedef struct {
    Vec3d pos, normal;
} Plane;

typedef struct {
    Vertex vertices[3];
    bool isProjected;
} Triangle;

void initClasses(JNIEnv*);

Vec2d parseVec2d(JNIEnv*, jobject);
Vec3d parseVec3d(JNIEnv*, jobject);
Vertex parseVertex(JNIEnv*, jobject);
Triangle parseTriangle(JNIEnv*, jobject, bool);
Plane parsePlane(JNIEnv*, jobject);

Vertex* parseVertexArray(JNIEnv*, jobjectArray, bool);
Plane* parsePlaneArray(JNIEnv*, jobjectArray);

jobject vec2dToJobject(JNIEnv*, Vec2d);
jobject vec3dToJobject(JNIEnv*, Vec3d);
jobject vertexToJobject(JNIEnv*, Vertex, bool);
jobject triangleToJobject(JNIEnv*, Triangle);
jobjectArray triangleArrayToJobject(JNIEnv*, Triangle*, int);

#endif