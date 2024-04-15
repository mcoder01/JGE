#include "util.h"

struct {
    jclass vec2dClass;
    jclass vec3dClass;
    jclass vertexClass;
    jclass planeClass;
    jclass triangleClass;
} classes;

void initClasses(JNIEnv *env) {
    classes.vec2dClass = env->FindClass("com/mcoder/jge/math/Vector2D");
    classes.vec3dClass = env->FindClass("com/mcoder/jge/math/Vector3D");
    classes.vertexClass = env->FindClass("com/mcoder/jge/g3d/core/Vertex");
    classes.planeClass = env->FindClass("com/mcoder/jge/g3d/geom/Plane");
    classes.triangleClass = env->FindClass("com/mcoder/jge/g3d/geom/Triangle");
}

Vec2d parseVec2d(JNIEnv *env, jobject obj) {
    Vec2d v;
    if (obj) {
        v.x = env->CallDoubleMethod(obj, env->GetMethodID(classes.vec2dClass, "getX", "()D"));
        v.y = env->CallDoubleMethod(obj, env->GetMethodID(classes.vec2dClass, "getY", "()D"));
    }

    return v;
}

Vec3d parseVec3d(JNIEnv *env, jobject obj) {
    return {
        env->CallDoubleMethod(obj, env->GetMethodID(classes.vec3dClass, "getX", "()D")),
        env->CallDoubleMethod(obj, env->GetMethodID(classes.vec3dClass, "getY", "()D")),
        env->CallDoubleMethod(obj, env->GetMethodID(classes.vec3dClass, "getZ", "()D"))
    };
}

Vertex parseVertex(JNIEnv *env, jobject obj) {
    return {
        parseVec3d(env, env->CallObjectMethod(obj, 
            env->GetMethodID(classes.vertexClass, "getPosition", "()Lcom/mcoder/jge/math/Vector3D;"))),
        parseVec3d(env, env->CallObjectMethod(obj, 
            env->GetMethodID(classes.vertexClass, "getNormal", "()Lcom/mcoder/jge/math/Vector3D;"))),
        parseVec2d(env, env->CallObjectMethod(obj, 
            env->GetMethodID(classes.vertexClass, "getTexCoords", "()Lcom/mcoder/jge/math/Vector2D;"))),
        parseVec2d(env, env->CallObjectMethod(obj, 
            env->GetMethodID(classes.vertexClass, "getScreenPosition", "()Lcom/mcoder/jge/math/Vector2D;")))
    };
}

Triangle parseTriangle(JNIEnv *env, jobject obj, bool isProjected) {
    Triangle triangle;
    jobjectArray vertices = (jobjectArray) env->CallObjectMethod(obj, 
        env->GetMethodID(classes.triangleClass, "vertices", "()[Lcom/mcoder/jge/g3d/core/Vertex;"));
    for (int i = 0; i < env->GetArrayLength(vertices); i++)
        triangle.vertices[i] = parseVertex(env, env->GetObjectArrayElement(vertices, i));
    triangle.isProjected = isProjected;
    return triangle;
}

Plane parsePlane(JNIEnv *env, jobject obj) {
    return {
        parseVec3d(env, env->CallObjectMethod(obj, env->GetMethodID(classes.planeClass, "pos", "()Lcom/mcoder/jge/math/Vector3D;"))),
        parseVec3d(env, env->CallObjectMethod(obj, env->GetMethodID(classes.planeClass, "normal", "()Lcom/mcoder/jge/math/Vector3D;")))
    };
}

Vertex* parseVertexArray(JNIEnv *env, jobjectArray array, bool areProjected) {
    int length = env->GetArrayLength(array);
    Vertex *vertices = new Vertex[length];
    for (int i = 0; i < length; i++)
        vertices[i] = parseVertex(env, env->GetObjectArrayElement(array, i));
    return vertices;
}

Plane* parsePlaneArray(JNIEnv *env, jobjectArray array) {
    int length = env->GetArrayLength(array);
    Plane *planes = new Plane[length];
    for (int i = 0; i < length; i++)
        planes[i] = parsePlane(env, env->GetObjectArrayElement(array, i));
    return planes;
}

jobject vec2dToJobject(JNIEnv *env, Vec2d v) {
    return env->NewObject(classes.vec2dClass, env->GetMethodID(
        classes.vec2dClass, "<init>", "(DD)V"), v.x, v.y);
}

jobject vec3dToJobject(JNIEnv *env, Vec3d v) {
    return env->NewObject(classes.vec3dClass, env->GetMethodID(
        classes.vec3dClass, "<init>", "(DDD)V"), v.x, v.y, v.z);
}

jobject vertexToJobject(JNIEnv *env, Vertex v, bool hasScreenPos) {
    if (hasScreenPos)
        return env->NewObject(classes.vertexClass, env->GetMethodID(classes.vertexClass, "<init>", 
            "(Lcom/mcoder/jge/math/Vector3D;Lcom/mcoder/jge/math/Vector2D;Lcom/mcoder/jge/math/Vector3D;Lcom/mcoder/jge/math/Vector2D;)V"),
            vec3dToJobject(env, v.pos), vec2dToJobject(env, v.texCoords), vec3dToJobject(env, v.normal), vec2dToJobject(env, v.screenPos));

    return env->NewObject(classes.vertexClass, env->GetMethodID(classes.vertexClass, "<init>", 
        "(Lcom/mcoder/jge/math/Vector3D;Lcom/mcoder/jge/math/Vector2D;Lcom/mcoder/jge/math/Vector3D;)V"),
        vec3dToJobject(env, v.pos), vec2dToJobject(env, v.texCoords), vec3dToJobject(env, v.normal));    
}

jobject triangleToJobject(JNIEnv *env, Triangle triangle) {
    jobjectArray vertices = env->NewObjectArray(3, classes.vertexClass, nullptr);
    for (int i = 0; i < 3; i++)
        env->SetObjectArrayElement(vertices, i, vertexToJobject(env, triangle.vertices[i], triangle.isProjected));
    return env->NewObject(classes.triangleClass, env->GetMethodID(classes.triangleClass, 
        "<init>", "([Lcom/mcoder/jge/g3d/core/Vertex;)V"), vertices);
}

jobjectArray triangleArrayToJobject(JNIEnv *env, Triangle *triangles, int length) {
    jobjectArray array = env->NewObjectArray(length, classes.triangleClass, nullptr);
    for (int i = 0; i < length; i++)
        env->SetObjectArrayElement(array, i, triangleToJobject(env, triangles[i]));
    return array;
}