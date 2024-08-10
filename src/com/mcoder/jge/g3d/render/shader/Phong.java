package com.mcoder.jge.g3d.render.shader;

import com.mcoder.jge.g3d.scene.light.AmbientLight;
import com.mcoder.jge.g3d.scene.light.Light;
import com.mcoder.jge.g3d.World;
import com.mcoder.jge.g3d.scene.light.Spotlight;
import com.mcoder.jge.math.Vector3D;

public class Phong extends Shader {
    private static final double diffusionPower = 3, specularPower = 1, specularHardness = 50;

    public Phong(World world) { super(world); }

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

            if (light instanceof Spotlight) {
                Vector3D viewDir = point.copy().normalize();
                Vector3D halfway = Vector3D.add(lightDir, viewDir).normalize();
                double specular = normal.dot(halfway);
                if (specular < 0) specular = 0;
                specular = Math.pow(specular, specularHardness);
                specular *= specularPower * invDistance;
                outputColor.add(Vector3D.scale(color, specular));
            }
        }

        return Vector3D.vecToRGB(outputColor);
    }
}
