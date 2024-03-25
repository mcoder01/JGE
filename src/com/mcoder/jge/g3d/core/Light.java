package com.mcoder.jge.g3d.core;

import com.mcoder.jge.g3d.scene.World;
import com.mcoder.jge.math.Vector3D;

public class Light extends Object3D {
    public enum LightType {
        SPOTLIGHT, AMBIENT
    }

    private final LightType type;
    private final Vector3D color;

    public Light(LightType type, double x, double y, double z, Vector3D color, World world) {
        super(world, x, y, z);
        this.type = type;
        this.color = color;
    }

    @Override
    public void tick() {
        pos.set(Vector3D.sub(worldPos, world.getCamera().getWorldPos()));

    }

    public LightType getType() {
        return type;
    }

    public Vector3D getColor() {
        return color;
    }
}
