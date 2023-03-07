package com.mcoder.jge.g3d.geom.solid;

import com.mcoder.jge.math.Vector;
import com.mcoder.jge.screen.Screen;

public class Point3D extends Vector {
    public Point3D(double x, double y, double z) {
        super(x, y, z);
    }

    public Point3D() {
        super();
    }

    public Point3D(Vector v) {
        this(v.getX(), v.getY(), v.getZ());
    }

    public Vector project(int fov, int width, int height) {
        double x = this.x/Math.abs(z)*fov+width/2.0;
        double y = -this.y/Math.abs(z)*fov+height/2.0;
        return new Vector(x, y, z);
    }

    @Override
    public Point3D copy() {
        return new Point3D(x, y, z);
    }
}
