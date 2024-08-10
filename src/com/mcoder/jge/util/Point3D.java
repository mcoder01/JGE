package com.mcoder.jge.util;

import com.mcoder.jge.math.Vector3D;

public class Point3D {
    private final Vector3D point;

    public Point3D(Vector3D point) {
        this.point = point;
    }

    public void move(Vector3D ds) {
        point.add(ds);
    }
    
    public void rotate(Vector3D rot) {
        double x, y, z;

        // Rotation on the Y-axis
        x = point.getX() * Math.cos(rot.getY()) + point.getZ() * Math.sin(rot.getY());
        z = point.getZ() * Math.cos(rot.getY()) - point.getX() * Math.sin(rot.getY());
        point.set(new Vector3D(x, point.getY(), z));

        // Rotation on the X-axis
        y = point.getY() * Math.cos(rot.getX()) - point.getZ() * Math.sin(rot.getX());
        z = point.getZ() * Math.cos(rot.getX()) + point.getY() * Math.sin(rot.getX());
        point.set(new Vector3D(point.getX(), y, z));
    }

    public Vector3D get() {
        return point;
    }
}
