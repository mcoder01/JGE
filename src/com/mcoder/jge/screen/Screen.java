package com.mcoder.jge.screen;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferStrategy;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.util.Arrays;
import java.util.LinkedList;

public class Screen extends Canvas {
    private static Screen instance;

    private BufferedImage image;
    public int[] pixels;
    public double[] zbuffer;
    private int fov;

    private final LinkedList<Display> drawers;
    private Display toRemove, toAdd;

    protected Screen() {
        super();
        setFocusable(true);
        setFocusTraversalKeysEnabled(false);
        drawers = new LinkedList<>();
    }

    private void setupCanvas(int width, int height) {
        setSize(width, height);
        setFOV(height);
        image = new BufferedImage(getWidth(), getHeight(), BufferedImage.TYPE_INT_RGB);
        pixels = ((DataBufferInt) image.getRaster().getDataBuffer()).getData();
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

    public final void update() {
        checkDrawers();
        for (Display drawer : drawers)
            drawer.tick();

        zbuffer = new double[getWidth()*getHeight()];
    }

    public final void draw() {
        BufferStrategy bs = getBufferStrategy();
        if (bs == null) {
            createBufferStrategy(2);
            return;
        }

        Arrays.fill(pixels, Color.BLACK.getRGB());
        Graphics2D screenGraphics = image.createGraphics();
        for (Display drawer : drawers)
            drawer.show(screenGraphics);
        screenGraphics.dispose();

        Graphics2D g2d = (Graphics2D) bs.getDrawGraphics();
        g2d.drawImage(image, 0, 0, this);
        g2d.dispose();
        bs.show();
    }

    public void addDrawer(Display drawer) {
        toAdd = drawer;
    }

    public void removeDrawer(Display drawer) {
        toRemove = drawer;
    }

    public int getFOV() {
        return fov;
    }

    public void setFOV(int fov) {
        this.fov = fov;
    }

    public static Screen getInstance() {
        if (instance == null)
            instance = new Screen();
        return instance;
    }

    public static JFrame createWindow(String title, int width, int height) {
        getInstance().setupCanvas(width, height);
        JFrame frame = new JFrame(title);
        frame.add(getInstance());
        frame.pack();
        frame.setResizable(false);
        frame.setLocationRelativeTo(null);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);
        return frame;
    }
}
