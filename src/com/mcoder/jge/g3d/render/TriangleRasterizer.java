package com.mcoder.jge.g3d.render;

import com.mcoder.jge.g3d.core.Vertex;
import com.mcoder.jge.g3d.geom.Triangle;
import com.mcoder.jge.g3d.core.Point3D;
import com.mcoder.jge.g3d.render.shader.Shader;
import com.mcoder.jge.g3d.render.slope.Slope;
import com.mcoder.jge.g3d.render.slope.SlopeDouble;
import com.mcoder.jge.g3d.render.slope.SlopeVector;
import com.mcoder.jge.math.Vector2D;
import com.mcoder.jge.math.Vector3D;
import com.mcoder.jge.screen.Screen;
import com.mcoder.jge.util.Texture;

public class TriangleRasterizer {
    private final Screen screen;

    public TriangleRasterizer(Screen screen) {
        this.screen = screen;
    }

    public void drawTriangle(Triangle triangle, Texture texture, Shader shader) {
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

        fitHalf(texture, shader, side1, side3, y1, y2);
        fitHalf(texture, shader, side2, side3, y2, y3);
    }

    private void compswap(Vertex[] vertices, int i, int j) {
        if (vertices[i].getScreenPosition().getY() > vertices[j].getScreenPosition().getY()) {
            Vertex temp = vertices[i];
            vertices[i] = vertices[j];
            vertices[j] = temp;
        }
    }

    private Slope[] calculateSlopes(Vertex[] vertices, int i, int j) {
        double zbegin = 1/vertices[i].getPosition().getZ(), zend = 1/vertices[i].getPosition().getZ();
        Vector2D leftPoint = vertices[i].getScreenPosition(), rightPoint = vertices[j].getScreenPosition();
        Vector2D leftTexPoint = vertices[i].getTexCoords(), rightTexPoint = vertices[j].getTexCoords();
        Vector3D leftNormal = vertices[i].getNormal(), rightNormal = vertices[j].getNormal();

        int numSteps = (int) (rightPoint.getY()-leftPoint.getY());
        Slope[] slopes = {
                new SlopeDouble(leftPoint.getX(), rightPoint.getX(), numSteps),
                new SlopeDouble(zbegin, zend, numSteps),
                new SlopeVector(Vector2D.mult(leftTexPoint, zbegin), Vector2D.mult(rightTexPoint, zend), numSteps),
                new SlopeVector(Vector3D.mult(leftNormal, zbegin), Vector3D.mult(rightNormal, zend), numSteps)
        };

        if (leftPoint.getY() < 0)
            for (Slope slope : slopes)
                slope.advance((int) Math.abs(leftPoint.getY()));
        return slopes;
    }

    private void fitHalf(Texture texture, Shader shader, Slope[] left, Slope[] right, int startY, int endY) {
        if (startY < 0) startY = 0;
        if (endY >= screen.getHeight())
            endY = screen.getHeight()-1;

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
            if (endX >= screen.getWidth())
                endX = screen.getWidth()-1;

            Slope[] props = new Slope[3];
            for (int i = 0; i < props.length; i++) {
                if (left[i+1] instanceof SlopeDouble l && right[i+1] instanceof SlopeDouble r)
                    props[i] = new SlopeDouble(l.getValue(), r.getValue(), numSteps);
                else if (left[i+1] instanceof SlopeVector l && right[i+1] instanceof SlopeVector r)
                    props[i] = new SlopeVector(l.getValue(), r.getValue(), numSteps);
                props[i].advance(offset);
            }

            for (int x = startX; x < endX; x++) {
                double z = 1/(double) props[0].getValue();
                int index = x+y*screen.getWidth();
                if (screen.zbuffer[index] == 0 || z < screen.zbuffer[index]) {
                    Vector2D texCoords = (Vector2D) props[1].getValue();
                    texCoords = Vector2D.mult(texCoords, z);

                    int u = (int) texCoords.getX(), v = (int) texCoords.getY();
                    if (u < 0) u = 0;
                    else if (u >= texture.getWidth())
                        u = texture.getWidth() - 1;
                    if (v < 0) v = 0;
                    else if (v >= texture.getHeight())
                        v = texture.getHeight() - 1;

                    int rgb = texture.getRGB(u, v);
                    Vector3D point = new Point3D(new Vector3D(x, y, z))
                            .invProject(screen.getFOV(), screen.getWidth(), screen.getHeight());

                    Vector3D normal = ((Vector3D) props[2].getValue());
                    normal = Vector3D.mult(normal, z).normalize();
                    screen.pixels[index] = shader.fragment(rgb, point, normal);
                    screen.zbuffer[index] = z;
                }

                for (Slope slope : props) slope.advance();
            }

            for (Slope slope : left) slope.advance();
            for (Slope slope : right) slope.advance();
        }
    }
}
