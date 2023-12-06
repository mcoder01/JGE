package com.mcoder.jge.screen;

import java.awt.*;

public interface Drawable {
    void tick(double deltaTime);
    void show(Graphics2D g2d);
}
