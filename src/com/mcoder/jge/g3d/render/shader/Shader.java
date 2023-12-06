package com.mcoder.jge.g3d.render.shader;

import com.mcoder.jge.g3d.core.Light;
import com.mcoder.jge.math.Vector3D;

import java.util.List;

public abstract class Shader {
    protected final List<Light> lights;

    public Shader(List<Light> lights) {
        this.lights = lights;
    }

    public abstract int fragment(int rgb, Vector3D point, Vector3D normal);
}
