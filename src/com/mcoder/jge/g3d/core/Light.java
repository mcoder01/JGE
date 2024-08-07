package com.mcoder.jge.g3d.core;

import com.mcoder.jge.g3d.scene.World;
import com.mcoder.jge.math.Vector3D;

public class Light extends Object3D {
    public enum LightType {
        SPOTLIGHT, AMBIENT
    }

    private final LightType type;
    private final Vector3D color, viewPos;

    public Light(LightType type, double x, double y, double z, Vector3D color, World world) {
        super(world, x, y, z);
        this.type = type;
        this.color = color;
        viewPos = new Vector3D();
    }

    @Override
    public void tick() {
        viewPos.set(pos);
        Point3D p3d = new Point3D(viewPos);
        p3d.rotate(rot);
        p3d.move(Vector3D.scale(world.getCamera().getPos(), -1));
        p3d.rotate(Vector3D.scale(world.getCamera().getRot(), -1));
    }

    public LightType getType() {
        return type;
    }

    public Vector3D getColor() {
        return color;
    }

    public Vector3D getViewPos() {
        return viewPos;
    }
}
