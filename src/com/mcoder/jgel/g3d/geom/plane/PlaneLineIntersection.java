package com.mcoder.jgel.g3d.geom.plane;

import com.mcoder.jgel.math.Vector;

public class PlaneLineIntersection {
    private final Vector point;
    private final double factor;

    private PlaneLineIntersection(Vector point, double factor) {
        this.point = point;
        this.factor = factor;
    }

    public Vector applyToLine(Vector lineStart, Vector lineEnd) {
        return Vector.sub(lineEnd, lineStart).mult(factor).add(lineStart);
    }

    public static PlaneLineIntersection compute(Plane plane, Vector lineStart, Vector lineEnd) {
        double pd = -plane.getNormal().dot(plane.getPos());
        double ad = lineStart.dot(plane.getNormal());
        double bd = lineEnd.dot(plane.getNormal());
        double factor = (-pd - ad) / (bd - ad);
        Vector lineStartToEnd = Vector.sub(lineEnd, lineStart);
        Vector lineToIntersect = Vector.mult(lineStartToEnd, factor);
        Vector point = Vector.add(lineStart, lineToIntersect);
        return new PlaneLineIntersection(point, factor);
    }

    public Vector getPoint() {
        return point;
    }
}
