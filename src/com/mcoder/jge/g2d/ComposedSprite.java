package com.mcoder.jge.g2d;

import com.mcoder.jge.screen.View;
import java.awt.*;
import java.io.Serializable;
import java.util.Stack;

public class ComposedSprite extends View implements Serializable {
    private final Stack<Sprite> sprites;

    public ComposedSprite() {
        sprites = new Stack<>();
    }

    public void push(Sprite sprite) {
        sprites.push(sprite);
    }

    public Sprite pop() {
        return sprites.pop();
    }

    @Override
    public void tick(double deltaTime) {
        for (Sprite s : sprites)
            s.tick(deltaTime);
    }

    @Override
    public void show(Graphics2D g2d) {
        for (Sprite s : sprites)
            s.show(g2d);
    }
}
