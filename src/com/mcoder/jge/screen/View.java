package com.mcoder.jge.screen;

import java.awt.*;
import java.awt.event.KeyListener;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.event.MouseWheelListener;
import java.util.Collection;
import java.util.LinkedList;

public abstract class View extends LinkedList<View> implements Drawable {
	protected int width, height, frameRate, frameCount;
	protected double deltaTime;
	protected Screen screen;

	@Override
	public boolean add(View view) {
		view.setScreen(screen);
		return super.add(view);
	}

	@Override
	public boolean addAll(Collection<? extends View> views) {
		for (View view : views)
			if (!add(view))
				return false;
		return true;
	}

	public void setup() {
		for (View view : this)
			view.setup();
	}

	public final void update() {
		width = screen.getWidth();
		height = screen.getHeight();
		frameRate = screen.getLoop().getFrameRate();
		frameCount = screen.getLoop().getFrameCount();
		deltaTime = screen.getLoop().getDeltaTime();

		for (View v : this)
			v.update();
	}

	@Override
	public void tick() {
		for (View v : this)
			v.tick();
	}

	@Override
	public void show(Graphics2D g2d) {
		for (View v : this)
			v.show(g2d);
	}

	public void onFocus() {
		setup();
		focus(this);
		forEach(this::focus);
	}

	public void onFocusLost() {
		forEach(this::removeFocus);
		removeFocus(this);
	}

	private void removeFocus(View v) {
		if (v instanceof MouseListener l)
			screen.removeMouseListener(l);

		if (v instanceof MouseMotionListener l)
			screen.removeMouseMotionListener(l);

		if (v instanceof MouseWheelListener l)
			screen.removeMouseWheelListener(l);

		if (v instanceof KeyListener l)
			screen.removeKeyListener(l);
	}

	private void focus(View v) {
		if (v instanceof MouseListener l)
			screen.addMouseListener(l);

		if (v instanceof MouseMotionListener l)
			screen.addMouseMotionListener(l);

		if (v instanceof MouseWheelListener l)
			screen.addMouseWheelListener(l);

		if (v instanceof KeyListener l)
			screen.addKeyListener(l);
	}

	public Screen getScreen() {
		return screen;
	}

	public void setScreen(Screen screen) {
		this.screen = screen;
	}
}
