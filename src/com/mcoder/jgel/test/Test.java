package com.mcoder.jgel.test;

import com.mcoder.jgel.g3d.scene.World;
import com.mcoder.jgel.scene.Screen;

import javax.swing.*;

public class Test {
    public static void main(String[] args) {
        JFrame frame = new JFrame("Test");
        Screen.getInstance().createCanvas(540, 540);
        frame.add(Screen.getInstance());
        frame.pack();
        frame.setResizable(false);
        frame.setLocationRelativeTo(null);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);
        
        Screen.getInstance().addDrawer(World.getInstance());
        Screen.getInstance().start();
    }
}
