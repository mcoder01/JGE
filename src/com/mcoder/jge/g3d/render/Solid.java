package com.mcoder.jge.g3d.render;

import com.mcoder.jge.g3d.geom.plane.Plane;
import com.mcoder.jge.g3d.geom.solid.Object3D;
import com.mcoder.jge.g3d.scene.Camera;
import com.mcoder.jge.g3d.scene.World;
import com.mcoder.jge.math.Vector;
import com.mcoder.jge.util.Texture;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.LinkedList;

public class Solid extends Object3D {
    private final World world;
    private final Model model;
    private Texture texture;

    public Solid(Model model, double x, double y, double z, World world) {
        super(x, y, z);
        this.model = model;
        this.world = world;
    }

    @Override
    public void tick() {
        setTexture(texture);
        int texW = texture.getImage().getWidth();
        int texH = texture.getImage().getHeight();
        Triangle[] triangles = model.obtainTriangles(texW, texH);
        LinkedList<Triangle> toClip = new LinkedList<>();
        Camera camera = world.getCamera();
        for (Triangle triangle : triangles) {
            triangle.rotate(rot);
            triangle.move(pos);
            if (triangle.isVisible(camera)) {
                triangle.move(Vector.mult(camera.getPos(), -1));
                triangle.rotate(Vector.mult(camera.getRot(), -1));
                toClip.add(triangle);
            }
        }

        Plane[] planes = {
                new Plane(new Vector(0, 0, 1), new Vector(0, 0, 1)),
                new Plane(new Vector(0, 0, 150), new Vector(0, 0, -1))};
        ArrayList<Triangle> clipped = clipTriangles(toClip, planes);
        for (Triangle triangle : clipped) {
            triangle.setTexture(texture);
            triangle.setLight(world.getLight());
            add(triangle);
        }

        super.tick();
    }

    @Override
    public void show(Graphics2D g2d) {
        super.show(g2d);
        clear();
    }

    private ArrayList<Triangle> clipTriangles(LinkedList<Triangle> toClip, Plane[] planes) {
        ArrayList<Triangle> clipped = new ArrayList<>();
        int length = toClip.size();
        int i = 0;
        while(length > 0) {
            Triangle triangle = toClip.poll();
            if (triangle != null) {
                ArrayList<Triangle> clippedTriangle = triangle.clip(planes[i]);
                if (i == planes.length-1)
                    clipped.addAll(clippedTriangle);
                else toClip.addAll(clippedTriangle);
            }

            length--;
            if (length == 0) {
                length = toClip.size();
                i++;
            }
        }

        return clipped;
    }

    public void setTexture(Texture texture) {
        if (texture == null) {
            BufferedImage image = new BufferedImage(1, 1, BufferedImage.TYPE_INT_RGB);
            image.setRGB(0, 0, Color.WHITE.getRGB());
            texture = new Texture(image);
        }
        this.texture = texture;
    }
}