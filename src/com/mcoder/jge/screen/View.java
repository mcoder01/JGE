package com.mcoder.jge.screen;

import com.mcoder.jge.g3d.render.Triangle;

import java.awt.*;
import java.awt.event.KeyListener;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.event.MouseWheelListener;
import java.util.LinkedList;

public abstract class View extends LinkedList<View> implements Drawable {
	protected Screen screen;
	protected GameLoop loop;

	@Override
	public boolean add(View view) {
		view.setLoop(loop);
		view.setScreen(screen);
		return super.add(view);
	}

	public void setup() {
		for (View view : this)
			view.setup();
	}

	@Override
	public void tick() {
		for (View view : this)
			view.tick();
	}

	@Override
	public void show(Graphics2D g2d) {
		for (View view : this)
			view.show(g2d);
	}

	public void render() {
		screen.addView(this);
	}

	public void onFocus() {
		focus(this);
		forEach(this::focus);
	}

	public void onFocusLost() {
		forEach(this::unfocus);
		unfocus(this);
	}

	private void unfocus(View view) {
		if (view instanceof MouseListener l)
			screen.removeMouseListener(l);

		if (view instanceof MouseMotionListener l)
			screen.removeMouseMotionListener(l);

		if (view instanceof MouseWheelListener l)
			screen.removeMouseWheelListener(l);

		if (view instanceof KeyListener l)
			screen.removeKeyListener(l);
	}

	private void focus(View view) {
		if (view instanceof MouseListener l)
			screen.addMouseListener(l);

		if (view instanceof MouseMotionListener l)
			screen.addMouseMotionListener(l);

		if (view instanceof MouseWheelListener l)
			screen.addMouseWheelListener(l);

		if (view instanceof KeyListener l)
			screen.addKeyListener(l);
	}

	public void setScreen(Screen screen) {
		this.screen = screen;
	}

	public void setLoop(GameLoop loop) {
		this.loop = loop;
	}
}
