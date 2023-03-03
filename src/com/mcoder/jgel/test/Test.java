package com.mcoder.jgel.test;

import com.mcoder.jgel.g3d.scene.World;
import com.mcoder.jgel.scene.GameLoop;
import com.mcoder.jgel.scene.Screen;

public class Test {
    public static void main(String[] args) {
        Screen.getInstance().createWindow("Test", 1280, 720);
        Screen.getInstance().addDrawer(World.getInstance());
        GameLoop gameLoop = new GameLoop(Screen.getInstance());
        gameLoop.start();
    }
}
