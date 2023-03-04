package com.mcoder.jge.g3d.geom.plane;

import com.mcoder.jge.math.Vector;

public class Plane {
    private final Vector pos, normal;

    public Plane(Vector pos, Vector normal) {
        this.pos = pos;
        this.normal = normal;
    }

    public double distanceToPoint(Vector point) {
        return normal.dot(point)-normal.dot(pos);
    }

    public Vector getPos() {
        return pos;
    }

    public Vector getNormal() {
        return normal;
    }
}
