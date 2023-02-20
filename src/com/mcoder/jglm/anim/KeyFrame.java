package com.mcoder.jglm.anim;

import java.util.HashMap;

public class KeyFrame extends HashMap<Knob, Double> {
	private final int time;

	public KeyFrame(int time) {
		this.time = time;
	}

	public int getTime() {
		return time;
	}
}
