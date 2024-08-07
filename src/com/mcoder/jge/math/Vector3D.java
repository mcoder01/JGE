package com.mcoder.jge.math;

public class Vector3D extends Vector2D {
    public Vector3D(double x, double y, double z) {
        super(x, y, z);
    }

    public Vector3D() {
        this(0, 0, 0);
    }

    public Vector3D set(double x, double y, double z) {
        return (Vector3D) super.set(x, y, z);
    }

    public Vector3D add(Vector3D v) {
        return (Vector3D) super.add(v);
    }

    public Vector3D sub(Vector3D v) {
        return (Vector3D) super.sub(v);
    }

    public Vector3D scale(double m) {
        return (Vector3D) super.scale(m);
    }

    public Vector3D scale(Vector3D v) { return (Vector3D) super.scale(v); }

    public Vector3D div(double d) {
        return (Vector3D) super.div(d);
    }

    public Vector3D cross(Vector3D v) {
        double x = getY()*v.getZ()-getZ()*v.getY();
        double y = getZ()*v.getX()-getX()*v.getZ();
        double z = getX()*v.getY()-getY()*v.getX();
        return new Vector3D(x, y, z);
    }

    public Vector3D setMag(double mag) {
        return (Vector3D) super.setMag(mag);
    }

    public Vector3D normalize() {
        return (Vector3D) super.normalize();
    }

    public Vector3D copy() {
        return new Vector3D(getX(), getY(), getZ());
    }

    @Override
    public boolean equals(Object obj) {
        if (obj instanceof Vector3D v)
            return v == this || (getX() == v.getX() && getY() == v.getY() && getZ() == v.getZ());
        return false;
    }

    public double getZ() {
        return values[2];
    }

    public void setZ(double z) {
        values[2] = z;
    }

    public static Vector3D add(Vector3D v1, Vector3D v2) {
        return (Vector3D) Vector.add(v1, v2);
    }

    public static Vector3D sub(Vector3D v1, Vector3D v2) {
        return (Vector3D) Vector.sub(v1, v2);
    }

    public static Vector3D scale(Vector3D v, double m) {
        return (Vector3D) Vector.scale(v, m);
    }

    public static Vector3D scale(Vector3D v1, Vector3D v2) {
        return (Vector3D) Vector.scale(v1, v2);
    }

    public static Vector3D div(Vector3D v, double d) {
        return (Vector3D) Vector.div(v, d);
    }

    public static Vector3D lerp(Vector3D v1, Vector3D v2, double t) {
        return (Vector3D) Vector.lerp(v1, v2, t);
    }

    public static Vector3D rgbToVec(int rgb) {
        return new Vector3D(rgb >> 16 & 0xff, rgb >> 8 & 0xff, rgb & 0xff);
    }

    public static int vecToRGB(Vector3D v) {
        double max = Math.max(v.getX(), v.getY());
        max = Math.max(max, v.getZ());
        if (max > 255)
            v.div(max).scale(255);
        return (int) v.getX() << 16 | (int) v.getY() << 8 | (int) v.getZ();
    }
}
