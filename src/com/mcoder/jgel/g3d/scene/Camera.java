package com.mcoder.jgel.g3d.scene;

import com.mcoder.jgel.g3d.geom.solid.Object3D;
import com.mcoder.jgel.scene.Screen;
import com.mcoder.jgel.scene.View;
import com.mcoder.jgel.math.Vector;

import java.awt.*;
import java.awt.event.*;

public class Camera extends Object3D implements View, KeyListener, MouseMotionListener, MouseListener {
    private final double moveSpeed;
    private int dx, dy, dz;
    private int prevMouseX, prevMouseY;

    public Camera(double x, double y, double z) {
        super(x, y, z);
        moveSpeed = 0.05;
        prevMouseX = -1;
    }

    @Override
    public void update() {
        double velX = (dx*Math.cos(rot.getY())+dz*Math.sin(rot.getY()))*moveSpeed;
        double velY = dy*moveSpeed;
        double velZ = (dz*Math.cos(rot.getY())-dx*Math.sin(rot.getY()))*moveSpeed;
        pos.add(new Vector(velX, velY, velZ));
    }

    @Override
    public void show(Graphics2D graphics2D) {}

    @Override
    public void keyTyped(KeyEvent keyEvent) {}

    @Override
    public void keyPressed(KeyEvent keyPressed) {
        switch (keyPressed.getKeyCode()) {
            case KeyEvent.VK_W -> dz = 1;
            case KeyEvent.VK_A -> dx = -1;
            case KeyEvent.VK_S -> dz = -1;
            case KeyEvent.VK_D -> dx = 1;
            case KeyEvent.VK_SPACE -> dy = 1;
            case KeyEvent.VK_SHIFT -> dy = -1;
        }
    }

    @Override
    public void keyReleased(KeyEvent e) {
        if (e.getKeyCode() == KeyEvent.VK_W || e.getKeyCode() == KeyEvent.VK_S) dz = 0;
        else if (e.getKeyCode() == KeyEvent.VK_A || e.getKeyCode() == KeyEvent.VK_D) dx = 0;
        else if (e.getKeyCode() == KeyEvent.VK_SHIFT || e.getKeyCode() == KeyEvent.VK_SPACE) dy = 0;
    }

    @Override
    public void mouseDragged(MouseEvent e) {
        if (prevMouseX != -1) {
            double rx = (double) (e.getY()-prevMouseY)/Screen.getInstance().getHeight();
            double ry = (double) (e.getX()-prevMouseX)/Screen.getInstance().getWidth();
            rot.add(new Vector(rx, ry));
        }

        prevMouseX = e.getX();
        prevMouseY = e.getY();
    }

    @Override
    public void mouseMoved(MouseEvent e) {}

    @Override
    public void mouseClicked(MouseEvent mouseEvent) {}

    @Override
    public void mousePressed(MouseEvent mouseEvent) {}

    @Override
    public void mouseReleased(MouseEvent mouseEvent) {
        prevMouseX = prevMouseY = -1;
    }

    @Override
    public void mouseEntered(MouseEvent mouseEvent) {}

    @Override
    public void mouseExited(MouseEvent mouseEvent) {}
}
