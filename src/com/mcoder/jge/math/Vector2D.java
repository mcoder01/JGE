package com.mcoder.jge.math;

public class Vector2D extends Vector {
    protected Vector2D(double... data) {
        super(data);
    }

    public Vector2D(double x, double y) {
        super(x, y);
    }

    public Vector2D() {
        this(0, 0);
    }

    public Vector2D add(Vector2D v) {
        return matToVec2D(super.add(v));
    }

    public Vector2D sub(Vector2D v) {
        return matToVec2D(super.sub(v));
    }

    public Vector2D scale(double m) {
        return matToVec2D(super.scale(m));
    }

    public Vector2D setMag(double mag) {
        return matToVec2D(super.setMag(mag));
    }

    public Vector2D normalize() {
        return matToVec2D(super.normalize());
    }

    public double heading() {
        return Math.atan2(getY(), getX());
    }

    public Vector2D copy() {
        return new Vector2D(getX(), getY());
    }

    public double getX() {
        return data[0];
    }

    public void setX(double x) {
        data[0] = x;
    }

    public double getY() {
        return data[1];
    }

    public void setY(double y) {
        data[1] = y;
    }

    public static Vector2D lerp(Vector2D v1, Vector2D v2, double t) {
        return matToVec2D(Vector.lerp(v1, v2, t));
    }

    private static Vector2D matToVec2D(Matrix m) {
        Vector2D result = new Vector2D();
        result.data = m.data;
        return result;
    }
}
