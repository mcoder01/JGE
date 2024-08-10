package com.mcoder.jge.g3d.scene.light;

import com.mcoder.jge.g3d.scene.Object3D;
import com.mcoder.jge.g3d.World;
import com.mcoder.jge.math.Vector3D;

public abstract class Light extends Object3D {
    protected final Vector3D color;
    protected Vector3D viewPos;

    public Light(Vector3D color, double x, double y, double z, World world) {
        super(world, x, y, z);
        this.color = color;
    }

    public Vector3D getColor() {
        return color;
    }

    public Vector3D getViewPos() {
        return viewPos;
    }
}
