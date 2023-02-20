package com.mcoder.jgel.anim;

import java.util.HashMap;

public class Animator extends Thread {
	private final Animation animation;
	private final double timePerFrame;
	private final HashMap<Knob, Double> steps;
	private int keyframeIndex;

	public Animator(Animation animation) {
		super();
		this.animation = animation;

		timePerFrame = 1.0E3 / Animation.getFrameRate();
		steps = new HashMap<>();
		keyframeIndex = 0;
	}

	@Override
	public void run() {
		super.run();
		int frameNum = calcStep();
		int frames = 0;
		long prevTime = System.currentTimeMillis();

		while (true) {
			long currTime = System.currentTimeMillis();
			if (currTime - prevTime >= timePerFrame) {
				update();
				frames++;
				if (frames == frameNum) {
					keyframeIndex++;
					if (keyframeIndex == animation.size() - 1)
						break;
					frames = 0;
					frameNum = calcStep();
				}
				prevTime = currTime;
			}
		}

		if (animation.getOnFinish() != null)
			animation.getOnFinish().run();
	}

	private void update() {
		for (Knob knob : steps.keySet())
			knob.accept(steps.get(knob));
	}

	private int calcStep() {
		KeyFrame k1 = animation.get(keyframeIndex);
		KeyFrame k2 = animation.get(keyframeIndex + 1);
		double dt = (k2.getTime() - k1.getTime()) / timePerFrame;

		steps.clear();
		if (dt != 0)
			for (Knob knob : k1.keySet())
				if (k2.containsKey(knob))
					steps.put(knob, (k2.get(knob) - k1.get(knob)) / dt);
				else steps.put(knob, 0.0);
		return (int) dt;
	}

	public static void launch(Animation animation) {
		new Animator(animation).start();
	}
}
