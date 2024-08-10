package com.mcoder.jge.g3d.scene;

import com.mcoder.jge.g3d.World;
import com.mcoder.jge.g3d.geom.Plane;
import com.mcoder.jge.math.Vector3D;

import java.awt.event.*;

public class Camera extends Object3D implements KeyListener, MouseMotionListener, MouseListener {
    private double moveSpeed;
    private int dx, dy, dz;
    private int prevMouseX, prevMouseY;

    public Camera(double x, double y, double z, World world) {
        super(world, x, y, z);
        moveSpeed = 5;
        prevMouseX = -1;
    }

    @Override
    public void keyTyped(KeyEvent keyEvent) {}

    @Override
    public void keyPressed(KeyEvent keyPressed) {
        switch (keyPressed.getKeyCode()) {
            case KeyEvent.VK_W -> dz = -1;
            case KeyEvent.VK_A -> dx = -1;
            case KeyEvent.VK_S -> dz = 1;
            case KeyEvent.VK_D -> dx = 1;
            case KeyEvent.VK_SPACE -> dy = 1;
            case KeyEvent.VK_SHIFT -> dy = -1;
        }

        if (keyPressed.isControlDown()) moveSpeed = 15;
    }

    @Override
    public void keyReleased(KeyEvent e) {
        if (e.getKeyCode() == KeyEvent.VK_W || e.getKeyCode() == KeyEvent.VK_S) dz = 0;
        else if (e.getKeyCode() == KeyEvent.VK_A || e.getKeyCode() == KeyEvent.VK_D) dx = 0;
        else if (e.getKeyCode() == KeyEvent.VK_SHIFT || e.getKeyCode() == KeyEvent.VK_SPACE) dy = 0;

        if (dx == 0 && dy == 0 && dz == 0)
            moveSpeed = 5;
    }

    @Override
    public void mouseDragged(MouseEvent e) {
        if (prevMouseX != -1) {
            double rx = (double) (prevMouseY-e.getY())/world.getScreen().getHeight();
            double ry = (double) (prevMouseX-e.getX())/world.getScreen().getWidth();
            rot.add(new Vector3D(rx, ry, 0));
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

    @Override
    public void tick() {
        double velX = (dx*Math.cos(rot.getY())+dz*Math.sin(rot.getY()));
        double velZ = (dz*Math.cos(rot.getY())-dx*Math.sin(rot.getY()));
        pos.add(new Vector3D(velX, dy, velZ).scale(deltaTime*moveSpeed));
    }

    public Plane[] getPlanes() {
        return new Plane[] {
                new Plane(new Vector3D(0, 0, -1), new Vector3D(0, 0, -1)),
                new Plane(new Vector3D(0, 0, -20), new Vector3D(0, 0, 1)),
                new Plane(new Vector3D(0, -20, 0), new Vector3D(0, 1, 0)),
                new Plane(new Vector3D(20, 0, 0), new Vector3D(-1, 0, 0)),
                new Plane(new Vector3D(0, 20, 0), new Vector3D(0, -1, 0)),
                new Plane(new Vector3D(-20, 0, 0), new Vector3D(1, 0, 0))
        };
    }
}
