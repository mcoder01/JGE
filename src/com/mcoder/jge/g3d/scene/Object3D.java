package com.mcoder.jge.g3d.scene;

import com.mcoder.jge.g3d.World;
import com.mcoder.jge.math.Vector3D;
import com.mcoder.jge.screen.View;

public abstract class Object3D extends View {
    protected final World world;
    protected Vector3D pos, rot;

    public Object3D(World world, double x, double y, double z) {
        this.world = world;
        pos = new Vector3D(x, y, z);
        rot = new Vector3D();
    }

    public void rotate(Vector3D deltaRot) {
        rot.add(deltaRot);
    }

    public Vector3D getPos() {
        return pos;
    }

    public void setPos(Vector3D pos) {
        this.pos = pos;
    }

    public Vector3D getRot() {
        return rot;
    }

    public void setRot(Vector3D rot) {
        this.rot = rot;
    }
}
