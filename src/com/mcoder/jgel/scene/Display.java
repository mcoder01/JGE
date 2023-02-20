package com.mcoder.jgel.scene;

import java.awt.event.KeyListener;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.event.MouseWheelListener;
import java.util.EventListener;
import java.util.LinkedList;

public abstract class Display implements View {
	protected final LinkedList<EventListener> listeners;

	protected Display() {
		listeners = new LinkedList<>();
	}

	public void render() {
		Screen.getInstance().addDrawer(this);
	}

	public void addListener(EventListener listener) {
		listeners.add(listener);
	}

	public void removeListener(EventListener listener) {
		listeners.remove(listener);
	}

	public void onFocus() {
		if (this instanceof EventListener)
			focus((EventListener) this);
		listeners.forEach(this::focus);
	}

	public void onFocusLost() {
		listeners.forEach(this::unfocus);
		listeners.clear();
		if (this instanceof EventListener)
			unfocus((EventListener) this);
	}

	private void unfocus(EventListener l) {
		if (l instanceof MouseListener)
			Screen.getInstance().removeMouseListener((MouseListener) l);

		if (l instanceof MouseMotionListener)
			Screen.getInstance().removeMouseMotionListener((MouseMotionListener) l);

		if (l instanceof MouseWheelListener)
			Screen.getInstance().removeMouseWheelListener((MouseWheelListener) l);

		if (l instanceof KeyListener)
			Screen.getInstance().removeKeyListener((KeyListener) l);
	}

	private void focus(EventListener l) {
		if (l instanceof MouseListener)
			Screen.getInstance().addMouseListener((MouseListener) l);

		if (l instanceof MouseMotionListener)
			Screen.getInstance().addMouseMotionListener((MouseMotionListener) l);

		if (l instanceof MouseWheelListener)
			Screen.getInstance().addMouseWheelListener((MouseWheelListener) l);

		if (l instanceof KeyListener)
			Screen.getInstance().addKeyListener((KeyListener) l);
	}
}
