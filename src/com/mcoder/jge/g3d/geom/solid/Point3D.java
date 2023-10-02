package com.mcoder.jge.g3d.geom.solid;

import com.mcoder.jge.math.Vector;

public class Point3D {
    private final Vector point;

    public Point3D(Vector point) {
        this.point = point;
    }

    public void move(Vector ds) {
        point.add(ds);
    }
    
    public void rotate(Vector rot) {
        double x, y, z;

        // Rotation on the Y-axis
        x = point.getX() * Math.cos(rot.getY()) + point.getZ() * Math.sin(rot.getY());
        z = point.getZ() * Math.cos(rot.getY()) - point.getX() * Math.sin(rot.getY());
        point.set(new Vector(x, point.getY(), z));

        // Rotation on the X-axis
        y = point.getY() * Math.cos(rot.getX()) + point.getZ() * Math.sin(rot.getX());
        z = point.getZ() * Math.cos(rot.getX()) - point.getY() * Math.sin(rot.getX());
        point.set(new Vector(point.getX(), y, z));
    }

    public Vector project(int fov, int width, int height) {
        double x = point.getX()/Math.abs(point.getZ())*fov+width/2.0;
        double y = -point.getY()/Math.abs(point.getZ())*fov+height/2.0;
        return new Vector(x, y, point.getZ());
    }

    public Vector invProject(int fov, int width, int height) {
        double x = (point.getX()-width/2.0)/fov*Math.abs(point.getZ());
        double y = -(point.getY()-height/2.0)/fov*Math.abs(point.getZ());
        return new Vector(x, y, point.getZ());
    }

    public Vector get() {
        return point;
    }
}
