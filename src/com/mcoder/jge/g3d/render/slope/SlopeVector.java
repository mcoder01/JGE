package com.mcoder.jge.g3d.render.slope;

import com.mcoder.jge.math.Vector;

public class SlopeVector implements Slope {
    private final Vector step;
    private Vector value;

    public SlopeVector(Vector start, Vector end, int steps) {
        value = start.copy();
        step = Vector.sub(end, start).div(steps);
    }

    @Override
    public void advance(int steps) {
        value = Vector.add(value, Vector.mult(step, steps));
    }

    @Override
    public void advance() {
        advance(1);
    }

    public Vector getValue() {
        return value;
    }
}
