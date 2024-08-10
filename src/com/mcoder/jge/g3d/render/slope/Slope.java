package com.mcoder.jge.g3d.render.slope;

public interface Slope {
    void advance(int steps);
    void advance();
    Object getValue();
    void stepAt(int step);
}
