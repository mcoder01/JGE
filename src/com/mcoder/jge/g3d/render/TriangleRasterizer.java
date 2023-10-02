package com.mcoder.jge.g3d.render;

import com.mcoder.jge.g3d.geom.solid.Point3D;
import com.mcoder.jge.math.Vector;
import com.mcoder.jge.screen.View;
import com.mcoder.jge.util.Texture;

import java.awt.*;

public class TriangleRasterizer extends View {
    private final Vector[] points, texPoints;
    private Texture texture;
    private final Vector normal;

    public TriangleRasterizer(Vector[] points, Vector[] texPoints, Vector normal) {
        this.points = points;
        this.texPoints = texPoints;
        this.normal = normal;
    }

    @Override
    public void show(Graphics2D g2d) {
        if (texture == null || texPoints == null)
            return;
        
        compswap(0, 1);
        compswap(0, 2);
        compswap(1, 2);

        Slope[] side1 = calculateSlopes(0, 1);
        Slope[] side2 = calculateSlopes(1, 2);
        Slope[] side3 = calculateSlopes(0, 2);

        fitHalf(texture, side1, side3, (int) points[0].getY(), (int) points[1].getY());
        fitHalf(texture, side2, side3, (int) points[1].getY(), (int) points[2].getY());
    }

    private void compswap(int i, int j) {
        if (points[i].getY() > points[j].getY()) {
            swap(points, i, j);
            swap(texPoints, i, j);
        }
    }

    private void swap(Vector[] v, int i, int j) {
        Vector temp = v[i];
        v[i] = v[j];
        v[j] = temp;
    }

    private Slope[] calculateSlopes(int i, int j) {
        double zbegin = 1/points[i].getZ(), zend = 1/points[j].getZ();
        double[] firstData = {points[i].getX(), zbegin, texPoints[i].getX()*zbegin, texPoints[i].getY()*zbegin};
        double[] lastData = {points[j].getX(), zend, texPoints[j].getX()*zend, texPoints[j].getY()*zend};

        Slope[] slopes = new Slope[firstData.length];
        int numSteps = (int) (points[j].getY()-points[i].getY());
        for (int k = 0; k < slopes.length; k++) {
            slopes[k] = new Slope(firstData[k], lastData[k], numSteps);
            if (points[i].getY() < 0)
                slopes[k].advance((int) Math.abs(points[i].getY()));
        }

        return slopes;
    }

    private void fitHalf(Texture texture, Slope[] left, Slope[] right, int startY, int endY) {
        if (startY < 0) startY = 0;
        if (endY >= screen.getHeight())
            endY = screen.getHeight()-1;

        for (int y = startY; y < endY; y++) {
            if (left[0].getValue() > right[0].getValue()) {
                Slope[] temp = left;
                left = right;
                right = temp;
            }

            int startX = (int) left[0].getValue();
            int endX = (int) right[0].getValue();

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
                props[i] = new Slope(left[i+1].getValue(), right[i+1].getValue(), numSteps);
                props[i].advance(offset);
            }

            for (int x = startX; x < endX; x++) {
                double z = 1/props[0].getValue();

                int index = x+y*screen.getWidth();
                if (screen.zbuffer[index] == 0 || z < screen.zbuffer[index]) {
                    int u = (int) (props[1].getValue() * z);
                    int v = (int) (props[2].getValue() * z);

                    if (u < 0) u = 0;
                    else if (u >= texture.getWidth())
                        u = texture.getWidth() - 1;
                    if (v < 0) v = 0;
                    else if (v >= texture.getHeight())
                        v = texture.getHeight() - 1;

                    int rgb = texture.getRGB(u, v);
                    Vector point = new Point3D(new Vector(x, y, z))
                            .invProject(screen.getFOV(), screen.getWidth(), screen.getHeight());
                    screen.pixels[index] = screen.getWorld().getLight().lightUp(rgb, point, normal, true);
                    screen.zbuffer[index] = z;
                }

                for (Slope slope : props) slope.advance();
            }

            for (Slope slope : left) slope.advance();
            for (Slope slope : right) slope.advance();
        }
    }

    public void setTexture(Texture texture) {
        this.texture = texture;
    }
}
