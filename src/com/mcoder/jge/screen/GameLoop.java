package com.mcoder.jge.screen;

public class GameLoop extends Thread {
	private final Screen screen;
	private int frameRate, frameCount;
	private double deltaTime;
	private boolean running;

	public GameLoop(Screen screen) {
		this.screen = screen;
		screen.setLoop(this);
		showFPS();
		frameRate = 0;
	}

	@Override
	public void run() {
		long lastTime = System.nanoTime(), totalTime = 0;
		double unprocessedFramesTime = 0;

		while(running) {
			double timePerFrame = (frameRate == 0) ? 0 : 1.0E9 / frameRate;
			long currTime = System.nanoTime();
			long passedTime = currTime - lastTime;
			deltaTime = passedTime/1.0E9;

			unprocessedFramesTime += passedTime;
			totalTime += passedTime;
			lastTime = currTime;

			if (unprocessedFramesTime >= timePerFrame) {
				screen.update();
				screen.draw();
				frameCount++;

				if (timePerFrame > 0)
					unprocessedFramesTime -= timePerFrame;
				else unprocessedFramesTime = 0;
			}

			if (totalTime >= 1.0E9) {
				showFPS();
				totalTime = frameCount = 0;
			}
		}
	}

	@Override
	public void start() {
		running = true;
		super.start();
	}

	@Override
	public void interrupt() {
		running = false;
		super.interrupt();
	}

	private void showFPS() {
		screen.getWindow().setTitle(screen.getTitle() + " | FPS: " + frameCount);
	}

	public void setFrameRate(int frameRate) {
		this.frameRate = frameRate;
	}

	public double getDeltaTime() {
		return deltaTime;
	}

	public int getFrameRate() {
		return frameRate;
	}

	public int getFrameCount() {
		return frameCount;
	}
}
