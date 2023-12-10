package com.mcoder.jge.g3d.render.shader;

import com.mcoder.jge.g3d.scene.World;
import com.mcoder.jge.math.Vector3D;

import java.util.List;

public class Tint extends Shader {
    public Tint(World world) {
        super(world);
    }

    @Override
    public int fragment(int rgb, Vector3D point, Vector3D normal) {
        return rgb;
    }
}
