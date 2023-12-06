package com.mcoder.jge.screen;

import java.awt.*;
import java.awt.event.KeyListener;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.event.MouseWheelListener;
import java.util.Collection;
import java.util.LinkedList;

public abstract class View extends LinkedList<Drawable> implements Drawable {
	protected Screen screen;

	@Override
	public boolean add(Drawable drawable) {
		if (drawable instanceof View v)
			v.setScreen(screen);
		return super.add(drawable);
	}

	@Override
	public boolean addAll(Collection<? extends Drawable> drawables) {
		for (Drawable drawable : drawables)
			if (!add(drawable))
				return false;
		return true;
	}

	public void setup() {
		for (Drawable drawable : this)
			if (drawable instanceof View v)
				v.setup();
	}

	@Override
	public void tick(double deltaTime) {
		for (Drawable drawable : this)
			drawable.tick(deltaTime);
	}

	@Override
	public void show(Graphics2D g2d) {
		for (Drawable drawable : this)
			drawable.show(g2d);
	}

	public void onFocus() {
		setup();
		focus(this);
		forEach(this::focus);
	}

	public void onFocusLost() {
		forEach(this::unfocus);
		unfocus(this);
	}

	private void unfocus(Drawable drawable) {
		if (drawable instanceof MouseListener l)
			screen.removeMouseListener(l);

		if (drawable instanceof MouseMotionListener l)
			screen.removeMouseMotionListener(l);

		if (drawable instanceof MouseWheelListener l)
			screen.removeMouseWheelListener(l);

		if (drawable instanceof KeyListener l)
			screen.removeKeyListener(l);
	}

	private void focus(Drawable drawable) {
		if (drawable instanceof MouseListener l)
			screen.addMouseListener(l);

		if (drawable instanceof MouseMotionListener l)
			screen.addMouseMotionListener(l);

		if (drawable instanceof MouseWheelListener l)
			screen.addMouseWheelListener(l);

		if (drawable instanceof KeyListener l)
			screen.addKeyListener(l);
	}

	public Screen getScreen() {
		return screen;
	}

	public void setScreen(Screen screen) {
		this.screen = screen;
	}
}
