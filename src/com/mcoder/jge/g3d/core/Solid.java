package com.mcoder.jge.g3d.core;

import com.mcoder.jge.g3d.render.shader.Shader;
import com.mcoder.jge.g3d.scene.World;
import com.mcoder.jge.util.Texture;

import java.awt.*;
import java.awt.image.BufferedImage;

public class Solid extends Object3D {
    private final Model model;
    private Texture texture;
    private final Shader shader;

    public Solid(Model model, Shader shader, double x, double y, double z, World world) {
        super(world, x, y, z);
        this.model = model;
        this.shader = shader;
        setTexture(null);
    }

    public void setTexture(Texture texture) {
        if (texture == null)
            setSolidTexture(Color.WHITE);
        else this.texture = texture;
    }

    public void setSolidTexture(Color color) {
        BufferedImage image = new BufferedImage(1, 1, BufferedImage.TYPE_INT_RGB);
        image.setRGB(0, 0, color.getRGB());
        texture = new Texture(image);
    }

    public Model getModel() {
        return model;
    }

    public Texture getTexture() {
        return texture;
    }

    public Shader getShader() {
        return shader;
    }
}