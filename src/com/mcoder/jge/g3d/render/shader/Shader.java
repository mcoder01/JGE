package com.mcoder.jge.g3d.render.shader;

import com.mcoder.jge.g3d.World;
import com.mcoder.jge.math.Vector3D;

public abstract class Shader {
    protected final World world;

    public Shader(World world) {
        this.world = world;
    }

    public abstract int fragment(int rgb, Vector3D point, Vector3D normal);
}
