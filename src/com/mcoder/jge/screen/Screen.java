package com.mcoder.jge.screen;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferStrategy;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.awt.image.ImageObserver;
import java.util.Arrays;
import java.util.LinkedList;

public class Screen extends Canvas {
    private final JFrame window;
    private final String title;
    private final BufferedImage image;
    public final int[] pixels;
    public double[] zbuffer;
    private int fov;

    private GameLoop loop;
    private final LinkedList<View> views;
    private View toRemove, toAdd;

    public Screen(JFrame window, int width, int height) {
        super();
        this.window = window;
        title = window.getTitle();
        window.add(this);

        setFocusable(true);
        setFocusTraversalKeysEnabled(false);
        setSize(width, height);
        setFOV(height);

        image = new BufferedImage(getWidth(), getHeight(), BufferedImage.TYPE_INT_RGB);
        pixels = ((DataBufferInt) image.getRaster().getDataBuffer()).getData();
        views = new LinkedList<>();
    }

    private void checkViews() {
        if (toRemove != null) {
            View last = views.getLast();
            views.remove(toRemove);
            toRemove.onFocusLost();
            if (toRemove == last && views.size() > 0)
                views.getLast().onFocus();

            toRemove = null;
        }

        if (toAdd != null) {
            for (View drawer : views)
                drawer.onFocusLost();

            views.add(toAdd);
            toAdd.onFocus();
            toAdd = null;
        }
    }

    public final void update(double deltaTime) {
        checkViews();
        for (View view : views)
            view.tick(deltaTime);
    }

    public final void draw() {
        BufferStrategy bs = getBufferStrategy();
        if (bs == null) {
            createBufferStrategy(2);
            return;
        }

        zbuffer = new double[getWidth()*getHeight()];
        Arrays.fill(pixels, Color.BLACK.getRGB());
        Graphics2D screenGraphics = image.createGraphics();
        for (View view : views)
            view.show(screenGraphics);
        screenGraphics.dispose();

        Graphics2D g2d = (Graphics2D) bs.getDrawGraphics();
        g2d.drawImage(image, 0, 0, this);
        bs.show();
    }

    public void addView(View view) {
        toAdd = view;
        toAdd.setScreen(this);
    }

    public void removeView(View view) {
        toRemove = view;
    }

    public JFrame getWindow() {
        return window;
    }

    public String getTitle() {
        return title;
    }

    public int getFOV() {
        return fov;
    }

    public void setFOV(int fov) {
        this.fov = fov;
    }

    public GameLoop getLoop() {
        return loop;
    }

    public void setLoop(GameLoop loop) {
        this.loop = loop;
    }

    public static Screen createWindow(String title, int width, int height) {
        JFrame window = new JFrame(title);
        Screen screen = new Screen(window, width, height);
        window.pack();
        window.setResizable(false);
        window.setLocationRelativeTo(null);
        window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        window.setVisible(true);
        return screen;
    }
}
