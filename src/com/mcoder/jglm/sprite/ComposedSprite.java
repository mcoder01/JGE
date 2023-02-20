package com.mcoder.jglm.sprite;

import com.mcoder.jglm.scene.View;
import java.awt.*;
import java.io.Serializable;
import java.util.Stack;

public class ComposedSprite extends Stack<Sprite> implements View, Serializable {
    @Override
    public void update() {
        for (Sprite s : this)
            s.update();
    }

    @Override
    public void show(Graphics2D g2d) {
        for (Sprite s : this)
            s.show(g2d);
    }
}
