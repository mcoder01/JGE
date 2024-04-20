package com.mcoder.jge.math;

import java.util.Arrays;

public class Vector extends Matrix {
    protected Vector(double... data) {
        super(data.length, 1, data);
    }

    public Vector(int length) {
        super(length, 1);
    }

    @Override
    public String toString() {
        return Arrays.toString(data);
    }

    public void set(Vector v) {
        if (length() != v.length())
            throw new RuntimeException("Error in set operation: vectors must have the same size!");
        System.arraycopy(v.data, 0, data, 0, length());
    }

    public Vector add(Vector v) {
        return matToVec(super.add(v));
    }

    public Vector sub(Vector v) {
        return matToVec(super.sub(v));
    }

    public Vector scale(double factor) {
        return matToVec(super.scale(factor));
    }

    public double dot(Vector v) {
        return times(v).data[0];
    }

    public double mag() {
        Vector transposed = matToVec(transpose());
        return Math.sqrt(transposed.dot(this));
    }

    public Vector setMag(double mag) {
        return matToVec(normalize().scale(mag));
    }

    public Vector normalize() {
        return matToVec(scale(1/mag()));
    }

    public int length() {
        return data.length;
    }

    public static Vector lerp(Vector v1, Vector v2, double t) {
        return matToVec(v2.sub(v1).scale(t).add(v1));
    }

    private static Vector matToVec(Matrix m) {
        Vector result = new Vector(m.data.length);
        result.data = m.data;
        return result;
    }
}
