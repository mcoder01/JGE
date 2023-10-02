package com.mcoder.jge.g3d.render;

import com.mcoder.jge.math.Vector;

public class Light {
    private final Vector light;
    private final double diffusionPower, specularPower, specularHardness;

    public Light(Vector light, double diffusionPower, double specularPower, double specularHardness) {
        this.light = light;
        this.diffusionPower = diffusionPower;
        this.specularPower = specularPower;
        this.specularHardness = specularHardness;
    }

    public int lightUp(int rgb, Vector point, Vector normal, boolean specularLight) {
        Vector lightDir = Vector.sub(light, point);
        double distance = lightDir.mag();
        lightDir.div(distance);

        Vector outputColor = diffusion(normal, rgbToVec(rgb), lightDir, distance);
        if (specularLight)
            outputColor.add(specular(point, normal, rgbToVec(0xffffff), lightDir, distance));
        return vecToRGB(outputColor);
    }

    private Vector diffusion(Vector normal, Vector diffuseColor, Vector lightDir, double distance) {
        double diffusion = normal.dot(lightDir);
        if (diffusion < 0.2) diffusion = 0.5;
        else if (diffusion > 1) diffusion = 1;
        return Vector.mult(diffuseColor, diffusion*diffusionPower/distance);
    }

    private Vector specular(Vector point, Vector normal, Vector specularColor, Vector lightDir, double distance) {
        Vector halfway = Vector.sub(lightDir, point).normalize();
        double specular = normal.dot(halfway);
        if (specular < 0) specular = 0;
        else if (specular > 1) specular = 1;
        specular = Math.pow(specular, specularHardness);
        return Vector.mult(specularColor, specular*specularPower/distance);
    }

    private Vector rgbToVec(int rgb) {
        return new Vector(rgb >> 16 & 0xff, rgb >> 8 & 0xff, rgb & 0xff);
    }

    private int vecToRGB(Vector v) {
        double max = Math.max(v.getX(), v.getY());
        max = Math.max(max, v.getZ());
        if (max > 255)
            v.div(max).mult(255);
        return (int) v.getX() << 16 | (int) v.getX() << 8 | (int) v.getZ();
    }
}
