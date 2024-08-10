package com.mcoder.jge.g3d.render.slope;

import com.mcoder.jge.math.Vector;

public class SlopeVector implements Slope {
    private final Vector start, stepSize;
    private Vector value;

    public SlopeVector(Vector start, Vector end, int steps) {
        this.start = start;
        value = start.copy();
        stepSize = Vector.sub(end, start).div(steps);
    }

    @Override
    public void advance(int steps) {
        value.add(Vector.scale(stepSize, steps));
    }

    @Override
    public void advance() {
        advance(1);
    }

    public Vector getValue() {
        return value;
    }

    @Override
    public void stepAt(int step) {
        value = Vector.add(start, Vector.scale(stepSize, step));
    }
}
