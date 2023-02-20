package com.mcoder.jgel.g3d.render;

import com.mcoder.jgel.g3d.geom.plane.Plane;
import com.mcoder.jgel.g3d.geom.solid.Object3D;
import com.mcoder.jgel.g3d.scene.Camera;
import com.mcoder.jgel.g3d.scene.World;
import com.mcoder.jgel.scene.View;
import com.mcoder.jgel.math.Vector;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class Solid extends Object3D implements View {
    private final Model model;
    private BufferedImage texture;
    private int[] pixels;

    public Solid(Model model, double x, double y, double z) {
        super(x, y, z);
        this.model = model;
    }

    @Override
    public void update() {}

    @Override
    public void show(Graphics2D g2d) {
        setTexture(texture);
        Triangle[] triangles = model.obtainTriangles(texture.getWidth(), texture.getHeight());
        for (Triangle triangle : triangles) {
            triangle.rotate(rot);
            Camera camera = World.getInstance().getCamera();
            triangle.move(Vector.sub(pos, camera.getPos()));
            triangle.rotate(Vector.mult(camera.getRot(), -1));
        }

        Plane[] planes = {
                new Plane(new Vector(0, 0, 1), new Vector(0, 0, 1)),
                new Plane(new Vector(0, 0, 16), new Vector(0, 0, -1))};
        ArrayList<Triangle> clipped = clipTriangles(triangles, planes);
        for (Triangle triangle : clipped)
            triangle.show(pixels, texture.getWidth(), texture.getHeight());
    }

    private ArrayList<Triangle> clipTriangles(Triangle[] triangles, Plane[] planes) {
        ArrayList<Triangle> clipped = new ArrayList<>();
        LinkedList<Triangle> toClipQueue = new LinkedList<>(List.of(triangles));

        int length = toClipQueue.size();
        int i = 0;
        while(length > 0) {
            Triangle triangle = toClipQueue.poll();
            if (triangle != null && (i > 0 || triangle.isVisible())) {
                ArrayList<Triangle> clippedTriangle = triangle.clip(planes[i]);
                if (i == planes.length-1)
                    clipped.addAll(clippedTriangle);
                else toClipQueue.addAll(clippedTriangle);
            }

            length--;
            if (length == 0) {
                length = toClipQueue.size();
                i++;
            }
        }

        return clipped;
    }

    public void setTexture(BufferedImage texture) {
        if (texture == null) {
            this.texture = new BufferedImage(1, 1, BufferedImage.TYPE_INT_RGB);
            this.texture.setRGB(0, 0, Color.WHITE.getRGB());
        } else this.texture = texture;
        pixels = this.texture.getRGB(0, 0, this.texture.getWidth(),
                this.texture.getHeight(), null, 0, this.texture.getWidth());
    }
}