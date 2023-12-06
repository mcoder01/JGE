package com.mcoder.jge.screen;

public class GameLoop extends Thread {
	private final Screen screen;
	private int frameRate, frames;

	public GameLoop(Screen screen) {
		this.screen = screen;
		screen.setLoop(this);
		showFPS();
		frameRate = 0;
	}

	@Override
	public void run() {
		long lastTime = System.nanoTime(), totalTime = 0;
		long unprocessedFramesTime = 0;

		while(true) {
			double timePerFrame = (frameRate == 0) ? 0 : 1.0E9 / frameRate;
			long currTime = System.nanoTime();
			long passedTime = currTime - lastTime;
			double deltaTime = passedTime/1.0E9;

			unprocessedFramesTime += passedTime;
			totalTime += passedTime;
			lastTime = currTime;

			if (unprocessedFramesTime >= timePerFrame) {
				screen.update(deltaTime);
				screen.draw();
				frames++;

				if (timePerFrame > 0)
					unprocessedFramesTime -= timePerFrame;
				else unprocessedFramesTime = 0;
			}

			if (totalTime >= 1.0E9) {
				showFPS();
				totalTime = frames = 0;
			}
		}
	}

	private void showFPS() {
		screen.getWindow().setTitle(screen.getTitle() + " | FPS: " + frames);
	}

	public void setFrameRate(int frameRate) {
		this.frameRate = frameRate;
	}
}
