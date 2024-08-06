package com.mcoder.jge.g3d.core;

import com.mcoder.jge.g3d.scene.World;
import com.mcoder.jge.math.Vector3D;
import com.mcoder.jge.screen.View;

public abstract class Object3D extends View {
    protected final World world;
    protected Vector3D pos, worldPos, rot;

    public Object3D(World world, double x, double y, double z) {
        this.world = world;
        worldPos = new Vector3D(x, y, z);
        pos = new Vector3D();
        rot = new Vector3D();
    }

    @Override
    public void tick() {
        pos.set(worldPos);
        Point3D p3d = new Point3D(pos);
        p3d.rotate(rot);
        p3d.move(Vector3D.scale(world.getCamera().getWorldPos(), -1));
        p3d.rotate(Vector3D.scale(world.getCamera().getRot(), -1));
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

    public Vector3D getWorldPos() {
        return worldPos;
    }

    public void setWorldPos(Vector3D worldPos) {
        this.worldPos = worldPos;
    }

    public Vector3D getRot() {
        return rot;
    }

    public void setRot(Vector3D rot) {
        this.rot = rot;
    }
}
