package com.mcoder.jge.math;

import java.util.Arrays;

public class Vector {
    protected final double[] values;

    public Vector(double... values) {
        this.values = values;
    }

    public Vector(int size) {
        values = new double[size];
    }

    public Vector(Vector v) {
        this(v.size());
        set(v.values);
    }

    @Override
    public String toString() {
        return Arrays.toString(values);
    }

    public Vector set(Vector v) {
        if (size() != v.size())
            throw new RuntimeException("Error in set operation: vectors must have the same size!");

        System.arraycopy(v.values, 0, values, 0, size());
        return this;
    }

    public Vector set(double... values) {
        return set(new Vector(values));
    }

    public Vector add(Vector v) {
        if (v.size() != size())
            throw new RuntimeException("Error in add operation: vectors must have the same size!");

        for (int i = 0; i < size(); i++)
            values[i] += v.values[i];
        return this;
    }

    public Vector sub(Vector v) {
        return add(mult(v, -1));
    }

    public Vector mult(double m) {
        for (int i = 0; i < size(); i++)
            values[i] *= m;
        return this;
    }

    public Vector div(double d) {
        return mult(1.0/d);
    }

    public double dot(Vector v) {
        if (size() != v.size())
            throw new RuntimeException("Dot operation error: vectors must have the same size!");

        double sum = 0;
        for (int i = 0; i < size(); i++)
            sum += values[i]*v.values[i];
        return sum;
    }

    public double mag() {
        return Math.sqrt(dot(this));
    }

    public Vector setMag(double mag) {
        normalize();
        return mult(mag);
    }

    public Vector normalize() {
        return div(mag());
    }

    public Vector copy() {
        return new Vector(this);
    }

    public int size() {
        return values.length;
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

    public static Vector lerp(Vector v1, Vector v2, double t) {
        return Vector.sub(v2, v1).mult(t).add(v1);
    }
}
