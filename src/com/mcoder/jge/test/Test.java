package com.mcoder.jge.test;

import com.mcoder.jge.g3d.scene.World;
import com.mcoder.jge.screen.GameLoop;
import com.mcoder.jge.screen.Screen;

public class Test {
    public static void main(String[] args) {
        Screen.getInstance().createWindow("Test", 1280, 720);
        Screen.getInstance().addDrawer(World.getInstance());
        GameLoop gameLoop = new GameLoop(Screen.getInstance());
        gameLoop.start();
    }
}
