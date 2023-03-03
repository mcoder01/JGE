package com.mcoder.jgel.scene;

import java.awt.*;
import java.awt.image.BufferStrategy;
import java.awt.image.BufferedImage;
import java.util.LinkedList;

public class Screen extends Canvas implements Runnable {
	private static Screen instance;

	private final Thread thread;
	private BufferedImage screen;
	private int[] pixels;
	private double[] zbuffer;

	private boolean running;
	private int tickSpeed, frameRate, ticks, frames;
	private final LinkedList<Display> drawers;
	private Display toRemove, toAdd;

	private Screen() {
		super();
		setFocusable(true);
		setFocusTraversalKeysEnabled(false);

		thread = new Thread(this);

		drawers = new LinkedList<>();

		tickSpeed = 60;
		frameRate = 60;
	}

	public void createCanvas(int width, int height) {
		setSize(width, height);
		screen = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
	}

	@Override
	public void run() {
		long lastTime = System.nanoTime();
		long totalTime = 0;
		while(running) {
			long currTime = System.nanoTime();
			totalTime += currTime-lastTime;
			lastTime = currTime;
			if (totalTime >= 1.0E9) {
				System.out.println("Ticks: " + ticks + ", FPS: " + frames);
				totalTime = ticks = frames = 0;
			}
		}
	}

	public void start() {
		running = true;
		thread.start();

		long lastTime = System.nanoTime();
		long unprocessedTicksTime = 0, unprocessedFramesTime = 0;

		while(running) {
			double timePerTick = (tickSpeed == 0) ? 0 : 1.0E9 / tickSpeed;
			double timePerFrame = (frameRate == 0) ? 0 : 1.0E9 / frameRate;
			long currTime = System.nanoTime();
			long passedTime = currTime - lastTime;

			unprocessedTicksTime += passedTime;
			unprocessedFramesTime += passedTime;
			lastTime = currTime;

			checkDrawers();
			if (unprocessedTicksTime >= timePerTick) {
				tick();
				ticks++;
				unprocessedTicksTime = 0;
			}

			if (unprocessedFramesTime >= timePerFrame) {
				draw();
				frames++;
				unprocessedFramesTime = 0;
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
		BufferStrategy bs = getBufferStrategy();
		if (bs == null) {
			createBufferStrategy(2);
			return;
		}

		zbuffer = new double[getWidth()*getHeight()];
		Graphics2D screenGraphics = screen.createGraphics();
		screenGraphics.setColor(Color.BLACK);
		screenGraphics.fillRect(0, 0, getWidth(), getHeight());
		screenGraphics.dispose();

		for (Display drawer : drawers) {
			pixels = screen.getRGB(0, 0, getWidth(), getHeight(), null, 0, getWidth());
			drawer.show(screenGraphics);
			screen.setRGB(0, 0, getWidth(), getHeight(), pixels, 0, getWidth());
		}
		screenGraphics.dispose();

		Graphics2D g2d = (Graphics2D) bs.getDrawGraphics();
		g2d.drawImage(screen, 0, 0, this);
		g2d.dispose();
		bs.show();
	}

	public void noLoop() {
		running = false;
	}

	public void addDrawer(Display drawer) {
		toAdd = drawer;
	}

	public void removeDrawer(Display drawer) {
		toRemove = drawer;
	}

	public boolean overlaps(int index, double z) {
		return zbuffer[index] == 0 || z < zbuffer[index];
	}

	public int getPixel(int x, int y) {
		return getPixel(x+y*getWidth());
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

	public double getZBuffer(int x, int y) {
		return getZBuffer(x+y*getWidth());
	}

	public double getZBuffer(int index) {
		return zbuffer[index];
	}

	public void setZBuffer(int index, double z) {
		zbuffer[index] = z;
	}

	public static Screen getInstance() {
		if (instance == null)
			instance = new Screen();
		return instance;
	}
}
