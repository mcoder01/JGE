package com.mcoder.jge.screen;

public class GameLoop extends Thread {
	private final Screen screen;
	private int tickSpeed, frameRate, ticks, frames;

	public GameLoop(Screen screen) {
		this.screen = screen;
		screen.setLoop(this);

		tickSpeed = 60;
		frameRate = 60;
	}

	@Override
	public void run() {
		long lastTime = System.nanoTime(), totalTime = 0;
		long unprocessedTicksTime = 0, unprocessedFramesTime = 0;

		while(true) {
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
				if (timePerTick > 0)
					unprocessedTicksTime -= timePerTick;
				else unprocessedTicksTime = 0;
			}

			if (unprocessedFramesTime >= timePerFrame) {
				screen.draw();
				frames++;
				if (timePerFrame > 0)
					unprocessedFramesTime -= timePerFrame;
				else unprocessedFramesTime = 0;
			}

			if (totalTime >= 1.0E9) {
				System.out.println("Ticks: " + ticks + ", FPS: " + frames);
				totalTime = ticks = frames = 0;
			}
		}
	}

	public void setTickSpeed(int tickSpeed) {
		this.tickSpeed = tickSpeed;
	}

	public void setFrameRate(int frameRate) {
		this.frameRate = frameRate;
	}
}
