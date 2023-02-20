package com.mcoder.jgel.g3d.geom;

public class Slope {
    private final double stepSize;
    private double value;

    public Slope(double start, double end, int steps) {
        value = start;
        stepSize = (end-start)/steps;
    }

    @Override
    public String toString() {
        return "{value=" + value + ", stepSize=" + stepSize + "}";
    }

    public void advance() {
        advance(1);
    }

    public void advance(int steps) {
        value += stepSize*steps;
    }

    public double getValue() {
        return value;
    }
}
