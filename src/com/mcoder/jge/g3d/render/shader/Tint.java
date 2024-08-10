package com.mcoder.jge.g3d.render.shader;

import com.mcoder.jge.g3d.World;
import com.mcoder.jge.math.Vector3D;

public class Tint extends Shader {
    public Tint(World world) {
        super(world);
    }

    @Override
    public int fragment(int rgb, Vector3D point, Vector3D normal) {
        return rgb;
    }
}
