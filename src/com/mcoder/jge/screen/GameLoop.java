package com.mcoder.jge.screen;

public class GameLoop {
	private final Screen screen;
	private boolean running;
	private int tickSpeed, frameRate, ticks, frames;

	public GameLoop(Screen screen) {
		this.screen = screen;
		tickSpeed = 60;
		frameRate = 75;
	}

	public void start() {
		running = true;

		long lastTime = System.nanoTime(), totalTime = 0;
		long unprocessedTicksTime = 0, unprocessedFramesTime = 0;

		while(running) {
			double timePerTick = (tickSpeed == 0) ? 0 : 1.0E9 / tickSpeed;
			double timePerFrame = (frameRate == 0) ? 0 : 1.0E9 / frameRate;
			long currTime = System.nanoTime();
			long passedTime = currTime - lastTime;

			unprocessedTicksTime += passedTime;
			unprocessedFramesTime += passedTime;
			totalTime += passedTime;
			lastTime = currTime;

			if (unprocessedTicksTime >= timePerTick) {
				screen.update();
				ticks++;
				unprocessedTicksTime = 0;
			}

			if (unprocessedFramesTime >= timePerFrame) {
				screen.draw();
				frames++;
				unprocessedFramesTime = 0;
			}

			if (totalTime >= 1.0E9) {
				System.out.println("Ticks: " + ticks + ", FPS: " + frames);
				totalTime = ticks = frames = 0;
			}
		}
	}

	public void noLoop() {
		running = false;
	}

	public void setTickSpeed(int tickSpeed) {
		this.tickSpeed = tickSpeed;
	}

	public void setFrameRate(int frameRate) {
		this.frameRate = frameRate;
	}
}
