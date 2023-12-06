package com.mcoder.jge.test;

import com.mcoder.jge.g3d.scene.World;
import com.mcoder.jge.screen.GameLoop;
import com.mcoder.jge.screen.Screen;

public class Test {
    public static void main(String[] args) {
        System.setProperty("sun.java2d.opengl", "true");
        Screen screen = Screen.createWindow("Test", 1280, 720);
        World world = new World();
        screen.addView(world);
        GameLoop gameLoop = new GameLoop(screen);
        gameLoop.start();
    }
}
