package com.mcoder.jglm;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.util.LinkedList;

public class Screen extends JPanel {
	private static Screen instance;

	private BufferedImage canvas;
	private Graphics2D g2d;
	private int[] pixels;
	private double[] zbuffer;


	private boolean running;
	private int tickSpeed, frameRate;
	private final LinkedList<Display> drawers;
	private Display toRemove, toAdd;

	private Screen() {
		super();
		setFocusable(true);
		setFocusTraversalKeysEnabled(false);

		drawers = new LinkedList<>();

		tickSpeed = 60;
		frameRate = 120;
	}

	public void createCanvas(int width, int height) {
		setSize(width, height);
		canvas = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		g2d = canvas.createGraphics();
		pixels = ((DataBufferInt) canvas.getRaster().getDataBuffer()).getData();
	}

	public void start() {
		long lastTime = System.nanoTime();
		long unprocessedTicksTime = 0, unprocessedFramesTime = 0, totalTime = 0;

		int ticks = 0, frames = 0;

		running = true;
		while(running) {
			double timePerTick = (tickSpeed == 0) ? 0 : 1.0E9 / tickSpeed;
			double timePerFrame = (frameRate == 0) ? 0 : 1.0E9 / frameRate;
			long currTime = System.nanoTime();
			long passedTime = currTime - lastTime;
			totalTime += passedTime;
			unprocessedTicksTime += passedTime;
			unprocessedFramesTime += passedTime;
			lastTime = currTime;

			checkDrawers();
			while(unprocessedTicksTime >= timePerTick) {
				tick();
				ticks++;
				if (tickSpeed > 0)
					unprocessedTicksTime -= timePerTick;
				else {
					unprocessedTicksTime = 0;
					break;
				}
			}

			while(unprocessedFramesTime >= timePerFrame) {
				draw();
				frames++;
				if (frameRate > 0)
					unprocessedFramesTime -= timePerFrame;
				else {
					unprocessedFramesTime = 0;
					break;
				}
			}

			if (totalTime >= 1.0E9) {
				System.out.println("Ticks: " + ticks + ", FPS: " + frames);
				totalTime = ticks = frames = 0;
			}
		}
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

	private void tick() {
		for (Display drawer : drawers)
			drawer.update();
	}

	private void draw() {
		zbuffer = new double[getWidth()*getHeight()];
		for (Display drawer : drawers)
			drawer.show(g2d);
		g2d.dispose();
	}

	public void stop() {
		running = false;
	}

	public void addDrawer(Display drawer) {
		toAdd = drawer;
	}

	public void removeDrawer(Display drawer) {
		toRemove = drawer;
	}

	public int getPixel(int x, int y) {
		return getPixel(x+y*canvas.getWidth());
	}

	public int getPixel(int index) {
		return pixels[index];
	}

	public void setPixel(int index, int rgb) {
		pixels[index] = rgb;
	}

	public void setTickSpeed(int tickSpeed) {
		this.tickSpeed = tickSpeed;
	}

	public void setFrameRate(int frameRate) {
		this.frameRate = frameRate;
	}

	public double[] getZBuffer() {
		return zbuffer;
	}

	public static Screen getInstance() {
		if (instance == null)
			instance = new Screen();
		return instance;
	}
}
