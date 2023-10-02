package com.mcoder.jge.g3d.render;

import com.mcoder.jge.g3d.geom.Triangle;
import com.mcoder.jge.g3d.geom.plane.Plane;
import com.mcoder.jge.g3d.geom.solid.Object3D;
import com.mcoder.jge.g3d.geom.solid.Point3D;
import com.mcoder.jge.g3d.scene.Camera;
import com.mcoder.jge.math.Vector;
import com.mcoder.jge.util.Texture;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.LinkedList;

public class Solid extends Object3D {
    private final Model model;
    private Texture texture;

    public Solid(Model model, double x, double y, double z) {
        super(x, y, z);
        this.model = model;
        setTexture(null);
    }

    @Override
    public void show(Graphics2D g2d) {
        Vector[] points = new Vector[model.getPoints().size()];
        Camera camera = screen.getWorld().getCamera();
        for (int i = 0; i < points.length; i++) {
            points[i] = model.getPoints().get(i).copy();
            Point3D p3d = new Point3D(points[i]);
            p3d.rotate(rot);
            p3d.move(pos);
        }

        Vector[] texPoints = new Vector[model.getTexPoints().size()];
        for (int i = 0; i < texPoints.length; i++) {
            Vector tp = model.getTexPoints().get(i).copy();
            texPoints[i] = new Vector(tp.getX()*texture.getWidth(), tp.getY()*texture.getHeight());
        }

        LinkedList<Triangle> triangles = buildTriangles(points, texPoints);
        for (Vector v : points) {
            Point3D p3d = new Point3D(v);
            p3d.move(Vector.mult(camera.getPos(), -1));
            p3d.rotate(Vector.mult(camera.getRot(), -1));
        }

        for (Triangle t : triangles)
            t.calculateNormal();

        ArrayList<Triangle> clipped = clipTriangles(triangles, new Plane[] {
                new Plane(new Vector(0, 0, 1), new Vector(0, 0, 1)),
                new Plane(new Vector(0, 0, 150), new Vector(0, 0, -1))
        });

        for (Triangle triangle : clipped) {
            Vector[] projected = project(triangle.getPoints());
            LinkedList<Triangle> toClip = new LinkedList<>();
            toClip.add(new Triangle(projected, triangle.getTexPoints(), triangle.getNormal()));
            ArrayList<Triangle> clipProj = clipTriangles(toClip, new Plane[] {
                    new Plane(new Vector(0, 0, 0), new Vector(0, 1, 0)),
                    new Plane(new Vector(screen.getWidth()-1, 0, 0), new Vector(-1, 0, 0)),
                    new Plane(new Vector(0, screen.getHeight()-1, 0), new Vector(0, -1, 0)),
                    new Plane(new Vector(0, 0, 0), new Vector(1, 0, 0))
            });

            for (Triangle ready : clipProj) {
                TriangleRasterizer tr = new TriangleRasterizer(ready.getPoints(),
                        ready.getTexPoints(), ready.getNormal());
                tr.setTexture(texture);
                add(tr);
            }
        }

        super.show(g2d);
        clear();
    }

    private LinkedList<Triangle> buildTriangles(Vector[] points, Vector[] texPoints) {
        LinkedList<Triangle> triangles = new LinkedList<>();
        for (int i = 0; i < model.getFaces(); i++) {
            int[] map = model.getMap().get(i);
            Vector[] vertexes = new Vector[map.length];
            for (int j = 0; j < map.length; j++)
                vertexes[j] = points[map[j]];

            int[] texMap = model.getTexMap().get(i);
            Vector[] texVertexes = new Vector[texMap.length];
            for (int j = 0; j < texMap.length; j++)
                texVertexes[j] = texPoints[texMap[j]];

            Triangle triangle = new Triangle(vertexes, texVertexes);
            if (triangle.isVisible(screen.getWorld().getCamera()))
                triangles.add(triangle);
        }

        return triangles;
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

    private Vector[] project(Vector[] points) {
        Vector[] projected = new Vector[points.length];
        for (int i = 0; i < points.length; i++) {
            Point3D p3d = new Point3D(points[i]);
            projected[i] = p3d.project(screen.getFOV(), screen.getWidth(), screen.getHeight());
        }

        return projected;
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