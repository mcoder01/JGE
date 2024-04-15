package com.mcoder.jge.g3d.core;

import com.mcoder.jge.math.Vector2D;
import com.mcoder.jge.math.Vector3D;

public class Vertex {
    private Vector3D position, normal;
    private Vector2D texCoords, screenPosition;

    public Vertex(Vector3D position, Vector2D texCoords, Vector3D normal, Vector2D screenPosition) {
        this.position = position;
        this.texCoords = texCoords;
        this.normal = normal;
        this.screenPosition = screenPosition;
    }

    public Vertex(Vector3D position, Vector2D texCoords, Vector3D normal) {
        this(position, texCoords, normal, null);
    }

    public Vertex() {}

    public Vertex copy() {
        Vertex copy = new Vertex(position.copy(), texCoords.copy(), normal.copy());
        if (screenPosition != null)
            copy.setScreenPosition(screenPosition.copy());
        return copy;
    }

    public Vector3D getPosition() {
        return position;
    }

    public void setPosition(Vector3D position) {
        this.position = position;
    }

    public Vector2D getTexCoords() {
        return texCoords;
    }

    public void setTexCoords(Vector2D texCoords) {
        this.texCoords = texCoords;
    }

    public Vector3D getNormal() {
        return normal;
    }

    public void setNormal(Vector3D normal) {
        this.normal = normal;
    }

    public Vector2D getScreenPosition() {
        return screenPosition;
    }

    public void setScreenPosition(Vector2D screenPosition) {
        this.screenPosition = screenPosition;
    }
}
