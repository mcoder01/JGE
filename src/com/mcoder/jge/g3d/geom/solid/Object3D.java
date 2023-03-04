package com.mcoder.jge.g3d.geom.solid;

import com.mcoder.jge.math.Vector;

public class Object3D {
    protected final Vector pos, rot;

    public Object3D(double x, double y, double z) {
        pos = new Vector(x, y, z);
        rot = new Vector();
    }

    public void move(double x, double y, double z) {
        pos.set(new Vector(x, y, z));
    }

    public void rotate(double rx, double ry) {
        rot.set(new Vector(rx, ry));
    }

    public Vector getPos() {
        return pos;
    }

    public Vector getRot() {
        return rot;
    }
}
