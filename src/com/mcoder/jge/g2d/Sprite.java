package com.mcoder.jge.g2d;

import com.mcoder.jge.math.Vector2D;
import com.mcoder.jge.util.Texture;
import com.mcoder.jge.screen.View;

import java.awt.*;
import java.awt.geom.AffineTransform;
import java.io.Serializable;

public class Sprite extends View implements Serializable {
	protected final Texture texture;
	protected final Vector2D pos, vel;
	protected final double w, h;
	protected double scale, rotation;
	protected float opacity;
	protected int zIndex;

	public Sprite(Texture texture, double x, double y, double w, double h) {
		this.texture = texture;
		this.w = w;
		this.h = h;

		pos = new Vector2D(x, y);
		vel = new Vector2D();
		scale = w / texture.nextImage().getWidth();
		opacity = 1f;
		zIndex = 1;
	}

	@Override
	public void tick() {
		pos.add(Vector2D.scale(vel, deltaTime));
	}

	@Override
	public void show(Graphics2D g2d) {
		AffineTransform transform = new AffineTransform();
		int x = (int) (pos.getX() - scaledWidth() / 2);
		int y = (int) (pos.getY() - scaledHeight() / 2);
		transform.translate(x, y);
		transform.rotate(rotation, 0, 0);
		transform.scale(scale, scale);
		Composite old = g2d.getComposite();
		g2d.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, opacity));
		g2d.drawImage(texture.getImage(), transform, null);
		g2d.setComposite(old);
	}

	private double scaledWidth() {
		return scale * texture.getImage().getWidth();
	}

	private double scaledHeight() {
		return scale * texture.getImage().getHeight();
	}

	public Vector2D getPos() {
		return pos;
	}

	public Vector2D getVel() {
		return vel;
	}

	public double getWidth() {
		return w;
	}

	public double getHeight() {
		return h;
	}

	public double getScale() {
		return scale;
	}

	public void setScale(double scale) {
		this.scale = scale;
	}

	public double getRotation() {
		return rotation;
	}

	public void setRotation(double rotation) {
		this.rotation = rotation;
	}

	public float getOpacity() {
		return opacity;
	}

	public void setOpacity(float opacity) {
		if (opacity < 0)
			opacity = 0;
		else if (opacity > 1)
			opacity = 1;
		this.opacity = opacity;
	}
}
