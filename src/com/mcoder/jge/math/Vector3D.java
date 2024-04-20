package com.mcoder.jge.math;

public class Vector3D extends Vector2D {
    public Vector3D(double x, double y, double z) {
        super(x, y, z);
    }

    public Vector3D() {
        this(0, 0, 0);
    }

    public Vector3D add(Vector3D v) {
        return matToVec3D(super.add(v));
    }

    public Vector3D sub(Vector3D v) {
        return matToVec3D(super.sub(v));
    }

    public Vector3D scale(double m) {
        return matToVec3D(super.scale(m));
    }

    public Vector3D cross(Vector3D v) {
        double x = getY()*v.getZ()-getZ()*v.getY();
        double y = getZ()*v.getX()-getX()*v.getZ();
        double z = getX()*v.getY()-getY()*v.getX();
        return new Vector3D(x, y, z);
    }

    public Vector3D setMag(double mag) {
        return matToVec3D(super.setMag(mag));
    }

    public Vector3D normalize() {
        return matToVec3D(super.normalize());
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
        return data[2];
    }

    public void setZ(double z) {
        data[2] = z;
    }

    public static Vector3D lerp(Vector3D v1, Vector3D v2, double t) {
        return matToVec3D(Vector.lerp(v1, v2, t));
    }

    public static Vector3D rgbToVec(int rgb) {
        return new Vector3D(rgb >> 16 & 0xff, rgb >> 8 & 0xff, rgb & 0xff);
    }

    public static int vecToRGB(Vector3D v) {
        double max = Math.max(v.getX(), v.getY());
        max = Math.max(max, v.getZ());
        if (max > 255) v.scale(255/max);
        return (int) v.getX() << 16 | (int) v.getY() << 8 | (int) v.getZ();
    }

    private static Vector3D matToVec3D(Matrix m) {
        Vector3D result = new Vector3D();
        result.data = m.data;
        return result;
    }
}
