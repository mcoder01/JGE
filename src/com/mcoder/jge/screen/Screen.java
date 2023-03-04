package com.mcoder.jge.screen;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferStrategy;
import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.LinkedList;

public class Screen extends Canvas {
    private static Screen instance;

    private BufferedImage screen;
    private int[] pixels;
    private double[] zbuffer;
    private int fov;

    private final LinkedList<Display> drawers;
    private Display toRemove, toAdd;

    private Screen() {
        super();
        setFocusable(true);
        setFocusTraversalKeysEnabled(false);
        drawers = new LinkedList<>();
    }

    public JFrame createWindow(String title, int width, int height) {
        setSize(width, height);
        screen = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        fov = width;

        JFrame frame = new JFrame(title);
        frame.add(this);
        frame.pack();
        frame.setResizable(false);
        frame.setLocationRelativeTo(null);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);
        return frame;
    }

    private void checkDrawers() {
        if (toRemove != null) {
            Display last = drawers.getLast();
            drawers.remove(toRemove);
            toRemove.onFocusLost();
            if (toRemove == last && drawers.size() > 0)
                drawers.getLast().onFocus();

            toRemove = null;
        }

        if (toAdd != null) {
            for (Display drawer : drawers)
                drawer.onFocusLost();

            drawers.add(toAdd);
            toAdd.onFocus();
            toAdd = null;
        }
    }

    public void tick() {
        checkDrawers();
        for (Display drawer : drawers)
            drawer.update();
    }

    public void draw() {
        BufferStrategy bs = getBufferStrategy();
        if (bs == null) {
            createBufferStrategy(2);
            return;
        }

        getPixels();
        zbuffer = new double[getWidth()*getHeight()];
        Arrays.fill(pixels, Color.BLACK.getRGB());
        Graphics2D screenGraphics = screen.createGraphics();
        for (Display drawer : drawers)
            drawer.show(screenGraphics);
        screenGraphics.dispose();
        updatePixels();

        Graphics2D g2d = (Graphics2D) bs.getDrawGraphics();
        g2d.drawImage(screen, 0, 0, this);
        g2d.dispose();
        bs.show();
    }

    public int[] getPixels() {
        if (pixels == null)
            pixels = screen.getRGB(0, 0, getWidth(), getHeight(), null, 0, getWidth());
        return pixels;
    }

    public void updatePixels() {
        if (pixels != null)
            screen.setRGB(0, 0, getWidth(), getHeight(), pixels, 0, getWidth());
    }

    public double[] getZBuffer() {
        return zbuffer;
    }

    public void addDrawer(Display drawer) {
        toAdd = drawer;
    }

    public void removeDrawer(Display drawer) {
        toRemove = drawer;
    }

    public int getWidth() {
        return screen.getWidth();
    }

    public int getHeight() {
        return screen.getHeight();
    }

    public int getFOV() {
        return fov;
    }

    public static Screen getInstance() {
        if (instance == null)
            instance = new Screen();
        return instance;
    }
}
