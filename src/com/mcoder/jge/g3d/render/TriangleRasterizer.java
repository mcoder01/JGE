package com.mcoder.jge.g3d.render;

import com.mcoder.jge.g3d.core.Point3D;
import com.mcoder.jge.g3d.core.Solid;
import com.mcoder.jge.g3d.core.Vertex;
import com.mcoder.jge.g3d.geom.Triangle;
import com.mcoder.jge.g3d.render.shader.Shader;
import com.mcoder.jge.g3d.render.slope.Slope;
import com.mcoder.jge.g3d.render.slope.SlopeDouble;
import com.mcoder.jge.g3d.render.slope.SlopeVertex;
import com.mcoder.jge.g3d.scene.World;
import com.mcoder.jge.math.Vector2D;
import com.mcoder.jge.math.Vector3D;
import com.mcoder.jge.util.Texture;

public class TriangleRasterizer {
    private final World world;

    public TriangleRasterizer(World world) {
        this.world = world;
    }

    public void drawTriangle(Triangle triangle, Solid solid) {
        Vertex[] vertices = triangle.vertices();

        compswap(vertices, 0, 1);
        compswap(vertices, 0, 2);
        compswap(vertices, 1, 2);

        Slope[] side1 = calculateSlopes(vertices, 0, 1);
        Slope[] side2 = calculateSlopes(vertices, 1, 2);
        Slope[] side3 = calculateSlopes(vertices, 0, 2);

        int y1 = (int) vertices[0].getScreenPosition().getY();
        int y2 = (int) vertices[1].getScreenPosition().getY();
        int y3 = (int) vertices[2].getScreenPosition().getY();

        fitHalf(solid, side1, side3, y1, y2);
        fitHalf(solid, side2, side3, y2, y3);
    }

    private void compswap(Vertex[] vertices, int i, int j) {
        if (vertices[i].getScreenPosition().getY() > vertices[j].getScreenPosition().getY()) {
            Vertex temp = vertices[i];
            vertices[i] = vertices[j];
            vertices[j] = temp;
        }
    }

    private Slope[] calculateSlopes(Vertex[] vertices, int i, int j) {
        double zbegin = 1/vertices[i].getPosition().getZ(), zend = 1/vertices[j].getPosition().getZ();
        Vector2D leftPoint = vertices[i].getScreenPosition(), rightPoint = vertices[j].getScreenPosition();
        Vertex left = new Vertex(
                Vector3D.scale(vertices[i].getPosition(), zbegin),
                Vector2D.scale(vertices[i].getTexCoords(), zbegin),
                Vector3D.scale(vertices[i].getNormal(), zbegin),
                Vector2D.scale(vertices[i].getScreenPosition(), zbegin)
        );

        Vertex right = new Vertex(
                Vector3D.scale(vertices[j].getPosition(), zend),
                Vector2D.scale(vertices[j].getTexCoords(), zend),
                Vector3D.scale(vertices[j].getNormal(), zend),
                Vector2D.scale(vertices[j].getScreenPosition(), zend)
        );

        int numSteps = (int) (rightPoint.getY()-leftPoint.getY());
        Slope[] slopes = {
                new SlopeDouble(leftPoint.getX(), rightPoint.getX(), numSteps),
                new SlopeDouble(zbegin, zend, numSteps),
                new SlopeVertex(left, right, numSteps)
        };

        if (leftPoint.getY() < 0)
            for (Slope slope : slopes)
                slope.advance((int) Math.abs(leftPoint.getY()));
        return slopes;
    }

    private void fitHalf(Solid solid, Slope[] left, Slope[] right, int startY, int endY) {
        if (startY < 0) startY = 0;
        if (endY >= world.getScreen().getHeight())
            endY = world.getScreen().getHeight()-1;

        for (int y = startY; y < endY; y++) {
            if ((double) left[0].getValue() > (double) right[0].getValue()) {
                Slope[] temp = left;
                left = right;
                right = temp;
            }

            int startX = (int) (double) left[0].getValue();
            int endX = (int) (double) right[0].getValue();

            int numSteps = endX-startX;
            int offset = 0;
            if (startX < 0) {
                offset = Math.abs(startX);
                startX = 0;
            }
            if (endX >= world.getScreen().getWidth())
                endX = world.getScreen().getWidth()-1;

            Slope[] props = new Slope[2];
            for (int i = 0; i < props.length; i++) {
                if (left[i+1] instanceof SlopeDouble l && right[i+1] instanceof SlopeDouble r)
                    props[i] = new SlopeDouble(l.getValue(), r.getValue(), numSteps);
                else if (left[i+1] instanceof SlopeVertex l && right[i+1] instanceof SlopeVertex r)
                    props[i] = new SlopeVertex(l.getValue(), r.getValue(), numSteps);
                props[i].advance(offset);
            }

            for (int x = startX; x < endX; x++) {
                double z = 1/(double) props[0].getValue();
                int index = x+y*world.getScreen().getWidth();
                if (world.getScreen().zBuffer[index] == 0 || z > world.getScreen().zBuffer[index]) {
                    Vertex vertex = (Vertex) props[1].getValue();

                    Vector2D texCoords = Vector2D.scale(vertex.getTexCoords(), z);
                    int u = (int) texCoords.getX(), v = (int) texCoords.getY();
                    if (u < 0) u = 0;
                    else if (u >= solid.getTexture().getWidth())
                        u = solid.getTexture().getWidth() - 1;
                    if (v < 0) v = 0;
                    else if (v >= solid.getTexture().getHeight())
                        v = solid.getTexture().getHeight() - 1;

                    int rgb = solid.getTexture().getRGB(u, v);
                    world.getScreen().pixels[index] = solid.getShader().fragment(rgb,
                            Vector3D.scale(vertex.getPosition(), z), vertex.getNormal());
                    world.getScreen().zBuffer[index] = z;
                }

                for (Slope slope : props) slope.advance();
            }

            for (Slope slope : left) slope.advance();
            for (Slope slope : right) slope.advance();
        }
    }
}
