package com.mcoder.jge.math;

public class Vector2D extends Vector {
    protected Vector2D(double... values) {
        super(values);
    }

    public Vector2D(double x, double y) {
        super(x, y);
    }

    public Vector2D() {
        this(0, 0);
    }

    public Vector2D set(double x, double y) {
        return (Vector2D) super.set(x, y);
    }

    public Vector2D add(Vector2D v) {
        return (Vector2D) super.add(v);
    }

    public Vector2D sub(Vector2D v) {
        return (Vector2D) super.sub(v);
    }

    public Vector2D scale(double m) {
        return (Vector2D) super.scale(m);
    }

    public Vector2D scale(Vector2D v) { return (Vector2D) super.scale(v); }

    public Vector2D div(double d) {
        return (Vector2D) super.div(d);
    }

    public Vector2D setMag(double mag) {
        return (Vector2D) super.setMag(mag);
    }

    public Vector2D normalize() {
        return (Vector2D) super.normalize();
    }

    public double heading() {
        return Math.atan2(getY(), getX());
    }

    public Vector2D copy() {
        return new Vector2D(getX(), getY());
    }

    public double getX() {
        return values[0];
    }

    public void setX(double x) {
        values[0] = x;
    }

    public double getY() {
        return values[1];
    }

    public void setY(double y) {
        values[1] = y;
    }

    public static Vector2D add(Vector2D v1, Vector2D v2) {
        return (Vector2D) Vector.add(v1, v2);
    }

    public static Vector2D sub(Vector2D v1, Vector2D v2) {
        return (Vector2D) Vector.sub(v1, v2);
    }

    public static Vector2D scale(Vector2D v, double m) {
        return (Vector2D) Vector.scale(v, m);
    }

    public static Vector2D scale(Vector2D v1, Vector2D v2) { return (Vector2D) Vector.scale(v1, v2); }

    public static Vector2D div(Vector2D v, double d) {
        return (Vector2D) Vector.div(v, d);
    }

    public static Vector2D lerp(Vector2D v1, Vector2D v2, double t) {
        return (Vector2D) Vector.lerp(v1, v2, t);
    }
}
