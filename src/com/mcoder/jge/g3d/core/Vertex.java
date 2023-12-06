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

    public static Vertex lerp(Vertex a, Vertex b, double t) {
        Vector3D pos = Vector3D.lerp(a.getPosition(), b.getPosition(), t);
        Vector2D texCoords = Vector2D.lerp(a.getTexCoords(), b.getTexCoords(), t);
        Vector3D normal = Vector3D.lerp(a.getNormal(), b.getNormal(), t);
        Vector2D screenPos = Vector2D.lerp(a.getScreenPosition(), b.getScreenPosition(), t);
        return new Vertex(pos, texCoords, normal, screenPos);
    }

    public static double[] inverseLerp(Vertex a, Vertex b, Vertex c) {
        Vertex v1 = new Vertex();
        v1.setPosition(Vector3D.sub(c.getPosition(), a.getPosition()));
        v1.setTexCoords(Vector2D.sub(c.getTexCoords(), a.getTexCoords()));
        v1.setNormal(Vector3D.sub(c.getNormal(), a.getNormal()));
        v1.setScreenPosition(Vector2D.sub(c.getScreenPosition(), a.getScreenPosition()));

        Vertex v2 = new Vertex();
        v2.setPosition(Vector3D.sub(b.getPosition(), a.getPosition()).normalize());
        v2.setTexCoords(Vector2D.sub(b.getTexCoords(), a.getTexCoords()).normalize());
        v2.setNormal(Vector3D.sub(b.getNormal(), a.getNormal()).normalize());
        v2.setScreenPosition(Vector2D.sub(b.getScreenPosition(), a.getScreenPosition()).normalize());

        return new double[] {
                v2.getPosition().dot(v1.getPosition()),
                v2.getTexCoords().dot(v1.getTexCoords()),
                v2.getNormal().dot(v1.getNormal()),
                v2.getScreenPosition().dot(v1.getScreenPosition())
        };
    }

    public static Vertex delta(Vertex a, Vertex b, int steps) {
        Vector3D pos = Vector3D.sub(b.getPosition(), a.getPosition()).div(steps);
        Vector2D texCoords = Vector2D.sub(b.getTexCoords(), a.getTexCoords()).div(steps);
        Vector3D normal = Vector3D.sub(b.getNormal(), a.getNormal()).div(steps);
        Vector2D screenPos = Vector2D.sub(b.getScreenPosition(), a.getScreenPosition()).div(steps);
        return new Vertex(pos, texCoords, normal, screenPos);
    }
}
