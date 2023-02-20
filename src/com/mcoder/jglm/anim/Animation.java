package com.mcoder.jglm.anim;

import java.util.ArrayList;

public class Animation extends ArrayList<KeyFrame> {
	private static final int frameRate = 60;
	private final int duration;
	private Runnable onFinish;

	public Animation(int duration) {
		this.duration = duration;
	}

	@Override
	public boolean add(KeyFrame keyFrame) {
		if (keyFrame.getTime() <= duration)
			return super.add(keyFrame);
		return false;
	}

	public void setOnFinish(Runnable onFinish) {
		this.onFinish = onFinish;
	}

	public Runnable getOnFinish() {
		return onFinish;
	}

	public int getDuration() {
		return duration;
	}

	public static int getFrameRate() {
		return frameRate;
	}
}
