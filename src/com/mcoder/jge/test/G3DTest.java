package com.mcoder.jge.test;

import com.mcoder.jge.g3d.scene.World;
import com.mcoder.jge.screen.GameLoop;
import com.mcoder.jge.screen.Screen;

public class G3DTest {
    public static void main(String[] args) {
        Screen screen = Screen.createWindow("Test G3D", 1280, 720);
        screen.addView(new World());
        new GameLoop(screen).start();
    }
}
