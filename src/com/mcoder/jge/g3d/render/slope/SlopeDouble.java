package com.mcoder.jge.g3d.render.slope;

public class SlopeDouble implements Slope {
    private final double start, stepSize;
    private double value;

    public SlopeDouble(double start, double end, int steps) {
        this.start = start;
        value = start;
        stepSize = (end-start)/steps;
    }

    @Override
    public void advance(int steps) {
        value += stepSize*steps;
    }

    @Override
    public void advance() {
        advance(1);
    }

    public Double getValue() {
        return value;
    }

    @Override
    public void stepAt(int step) {
        value = start+stepSize*step;
    }
}
