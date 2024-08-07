package com.mcoder.jge.g3d.render.shader;

import com.mcoder.jge.g3d.core.Light;
import com.mcoder.jge.g3d.scene.World;
import com.mcoder.jge.math.Vector3D;

public class Phong extends Shader {
    private double diffusionPower, specularPower, specularHardness;

    public Phong(World world) {
        super(world);
        diffusionPower = 3;
        specularPower = 1;
        specularHardness = 50;
    }

    @Override
    public int fragment(int rgb, Vector3D point, Vector3D normal) {
        Vector3D outputColor = new Vector3D();
        for (Light light : world.getLights()) {
            Vector3D color = Vector3D.rgbToVec(rgb).add(light.getColor());
            Vector3D lightDir = Vector3D.sub(point, light.getViewPos());
            double invDistance = 1/lightDir.mag();
            lightDir.scale(invDistance);

            double diffusion = normal.dot(lightDir);
            if (diffusion < 0) diffusion = 0;
            diffusion *= diffusionPower*invDistance;
            outputColor.add(Vector3D.scale(color, diffusion));

            Vector3D viewDir = point.copy().normalize();
            Vector3D halfway = Vector3D.add(lightDir, viewDir).normalize();
            double specular = normal.dot(halfway);
            if (specular < 0) specular = 0;
            specular = Math.pow(specular, specularHardness);
            specular *= specularPower * invDistance;
            outputColor.add(Vector3D.scale(color, specular));
        }

        return Vector3D.vecToRGB(outputColor);
    }

    public void setDiffusionPower(double diffusionPower) {
        this.diffusionPower = diffusionPower;
    }

    public void setSpecularPower(double specularPower) {
        this.specularPower = specularPower;
    }

    public void setSpecularHardness(double specularHardness) {
        this.specularHardness = specularHardness;
    }
}
