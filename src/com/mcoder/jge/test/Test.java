package com.mcoder.jge.test;

import com.mcoder.jge.g3d.scene.World;
import com.mcoder.jge.screen.GameLoop;
import com.mcoder.jge.screen.Screen;

public class Test {
    public static void main(String[] args) {
        Screen screen = Screen.createWindow("Test", 1280, 720);
        screen.addView(new World());
        GameLoop gameLoop = new GameLoop(screen);
        gameLoop.start();
    }
}
