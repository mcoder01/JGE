package com.mcoder.jge.g3d.core;

import com.mcoder.jge.math.Vector;
import com.mcoder.jge.math.Vector2D;
import com.mcoder.jge.math.Vector3D;

public class Vertex {
    private final Vector[] data;

    public Vertex(Vector3D position, Vector2D texCoords, Vector3D normal, Vector2D screenPosition) {
        data = new Vector[] {position, texCoords, normal, screenPosition};
    }

    public Vertex(Vector3D position, Vector2D texCoords, Vector3D normal) {
        this(position, texCoords, normal, null);
    }

    public Vertex() {
        this(null, null, null, null);
    }

    public Vertex copy() {
        Vertex vertex = new Vertex();
        for (int i = 0; i < data.length; i++)
            if (data[i] != null)
                vertex.data[i] = data[i].copy();
        return vertex;
    }

    public Vector3D getPosition() {
        return (Vector3D) data[0];
    }

    public void setPosition(Vector3D position) {
        data[0] = position;
    }

    public Vector2D getTexCoords() {
        return (Vector2D) data[1];
    }

    public void setTexCoords(Vector2D texCoords) {
        data[1] = texCoords;
    }

    public Vector3D getNormal() {
        return (Vector3D) data[2];
    }

    public void setNormal(Vector3D normal) {
        data[2] = normal;
    }

    public Vector2D getScreenPosition() {
        return (Vector2D) data[3];
    }

    public void setScreenPosition(Vector2D screenPosition) {
        data[3] = screenPosition;
    }

    public static Vertex lerp(Vertex v1, Vertex v2, double t) {
        Vertex result = new Vertex();
        for (int i = 0; i < v1.data.length; i++)
            if (v1.data[i] != null)
                result.data[i] = Vector.lerp(v1.data[i], v2.data[i], t);
        return result;
    }
}
