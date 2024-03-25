package com.mcoder.jge.test;

import com.mcoder.jge.g2d.Sprite;
import com.mcoder.jge.screen.GameLoop;
import com.mcoder.jge.screen.Screen;
import com.mcoder.jge.util.Texture;

public class G2DTest {
    public static void main(String[] args) {
        Screen screen = Screen.createWindow("Test G2D", 1280, 720);

        Texture texture = new Texture("textures/cobblestone.png");
        Sprite sprite = new Sprite(texture, 0, 0, 16, 16);
        screen.addView(sprite);
        GameLoop gameLoop = new GameLoop(screen);
        gameLoop.start();
    }
}
