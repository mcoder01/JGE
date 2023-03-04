package com.mcoder.jge.math;

public class Vector {
    protected double x, y, z;

    public Vector(double x, double y, double z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    public Vector(double x, double y) {
        this(x, y, 0);
    }

    public Vector() {
        this(0, 0, 0);
    }

    @Override
    public String toString() {
        return "{x=" + x + ", y=" + y + ", z=" + z + "}";
    }

    public Vector set(Vector v) {
        setX(v.x);
        setY(v.y);
        setZ(v.z);
        return this;
    }

    public Vector add(Vector v) {
        x += v.x;
        y += v.y;
        z += v.z;
        return this;
    }

    public Vector sub(Vector v) {
        return add(mult(v, -1));
    }

    public Vector mult(double m) {
        x *= m;
        y *= m;
        z *= m;
        return this;
    }

    public Vector div(double d) {
        return mult(1/d);
    }

    public double dot(Vector v) {
        return x*v.x+y*v.y+z*v.z;
    }

    public Vector cross(Vector v) {
        double x = this.y*v.z-this.z*v.y;
        double y = this.z*v.x-this.x*v.z;
        double z = this.x*v.y-this.y*v.x;
        return new Vector(x, y, z);
    }

    public double mag() {
        return Math.sqrt(x*x+y*y+z*z);
    }

    public Vector setMag(double mag) {
        normalize();
        return mult(mag);
    }

    public Vector normalize() {
        double mag = mag();
        if (mag > 0) {
            x /= mag;
            y /= mag;
            z /= mag;
        }

        return this;
    }

    public double heading() {
        return Math.atan2(y, x);
    }

    public Vector copy() {
        return new Vector(x, y, z);
    }

    public double getX() {
        return x;
    }

    public void setX(double x) {
        this.x = x;
    }

    public double getY() {
        return y;
    }

    public void setY(double y) {
        this.y = y;
    }

    public double getZ() {
        return z;
    }

    public void setZ(double z) {
        this.z = z;
    }

    public static Vector add(Vector v1, Vector v2) {
        return v1.copy().add(v2);
    }

    public static Vector sub(Vector v1, Vector v2) {
        return v1.copy().sub(v2);
    }

    public static Vector mult(Vector v, double m) {
        return v.copy().mult(m);
    }

    public static Vector div(Vector v, double d) {
        return v.copy().div(d);
    }

    public static double dist(Vector v1, Vector v2) {
        return Vector.sub(v2, v1).mag();
    }

    public static Vector fromAngle(double angle) {
        return new Vector(Math.cos(angle), Math.sin(angle));
    }
}
