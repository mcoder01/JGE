package com.mcoder.jge.g3d.scene.light;

import com.mcoder.jge.g3d.World;
import com.mcoder.jge.math.Vector3D;

public class AmbientLight extends Light {
    public AmbientLight(Vector3D color, World world) {
        super(color, 0, 0, 0, world);
    }

    @Override
    public void tick() {
        viewPos = pos.copy();
    }
}
