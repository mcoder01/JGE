package com.mcoder.jge.g3d.render;

import com.mcoder.jge.g3d.geom.plane.Plane;
import com.mcoder.jge.g3d.geom.plane.PlaneLineIntersection;
import com.mcoder.jge.g3d.geom.solid.Point3D;
import com.mcoder.jge.g3d.scene.Camera;
import com.mcoder.jge.math.Vector;
import com.mcoder.jge.screen.Screen;
import com.mcoder.jge.screen.View;
import com.mcoder.jge.util.Texture;

import java.awt.*;
import java.util.ArrayList;

public class Triangle extends View {
    private final Point3D[] points;
    private final Vector[] texPoints;
    private Texture texture;
    private Vector normal, light;

    public Triangle(Point3D[] points, Vector[] texPoints) {
        this.points = points;
        this.texPoints = texPoints;
    }

    private void calculateNormal() {
        Vector v1 = Vector.sub(points[1], points[0]);
        Vector v2 = Vector.sub(points[2], points[0]);
        normal = v1.cross(v2).normalize();
    }

    public boolean isVisible(Camera camera) {
        calculateNormal();
        Vector cameraRay = Vector.sub(points[0], camera.getPos());
        return normal.dot(cameraRay) < 0;
    }

    private Vector[] project() {
        Vector[] projected = new Vector[points.length];
        for (int i = 0; i < points.length; i++)
            projected[i] = points[i].project(screen.getFOV(), screen.getWidth(), screen.getHeight());
        return projected;
    }

    public void move(Vector pos) {
        for (Point3D point : points)
            point.add(pos);
    }

    public void rotate(Vector rot) {
        double rx = rot.getX(), ry = rot.getY();
        for (Point3D v : points) {
            double x, y, z;

            // Rotation on the Y-axis
            x = v.getX() * Math.cos(ry) + v.getZ() * Math.sin(ry);
            z = v.getZ() * Math.cos(ry) - v.getX() * Math.sin(ry);
            v.set(new Vector(x, v.getY(), z));

            // Rotation on the X-axis
            y = v.getY() * Math.cos(rx) + v.getZ() * Math.sin(rx);
            z = v.getZ() * Math.cos(rx) - v.getY() * Math.sin(rx);
            v.set(new Vector(v.getX(), y, z));
        }
    }

    @Override
    public void tick() {}

    @Override
    public void show(Graphics2D g2d) {
        if (texPoints == null)
            return;

        calculateNormal();
        Vector[] points = project();
        compswap(points, 0, 1);
        compswap(points, 0, 2);
        compswap(points, 1, 2);

        Slope[] side1 = calculateSlopes(points, 0, 1);
        Slope[] side2 = calculateSlopes(points, 1, 2);
        Slope[] side3 = calculateSlopes(points, 0, 2);

        fitHalf(texture, side1, side3, (int) points[0].getY(), (int) points[1].getY());
        fitHalf(texture, side2, side3, (int) points[1].getY(), (int) points[2].getY());
    }

    private void compswap(Vector[] points, int i, int j) {
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

    private Slope[] calculateSlopes(Vector[] points, int i, int j) {
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

                    int rgb = applyBrightness(texture.getRGB(u, v));
                    screen.pixels[index] = rgb;
                    screen.zbuffer[index] = z;
                }

                for (Slope slope : props) slope.advance();
            }

            for (Slope slope : left) slope.advance();
            for (Slope slope : right) slope.advance();
        }
    }

    private int applyBrightness(int rgb) {
        int r = rgb >> 16 & 0xff;
        int g = rgb >> 8 & 0xff;
        int b = rgb & 0xff;

        int brightness = (int) (Math.abs(normal.dot(light))*200+55);
        r = (int) (r/255.0*brightness);
        g = (int) (g/255.0*brightness);
        b = (int) (b/255.0*brightness);
        return r << 16 | g << 8 | b;
    }

    public ArrayList<Triangle> clip(Plane plane) {
        ArrayList<Triangle> triangles = new ArrayList<>(2);

        ArrayList<Point3D> inside = new ArrayList<>(3);
        ArrayList<Point3D> outside = new ArrayList<>(3);
        ArrayList<Vector> insideTex = new ArrayList<>(3);
        ArrayList<Vector> outsideTex = new ArrayList<>(3);

        for (int i = 0; i < points.length; i++) {
            double dist = plane.distanceToPoint(points[i]);
            if (dist >= 0) {
                inside.add(points[i]);
                insideTex.add(texPoints[i]);
            } else {
                outside.add(points[i]);
                outsideTex.add(texPoints[i]);
            }
        }

        if (inside.size() == 3)
            triangles.add(this);
        else if (inside.size() == 1 && outside.size() == 2) {
            PlaneLineIntersection intersection1 = PlaneLineIntersection.compute(plane, inside.get(0), outside.get(0));
            Point3D p1 = new Point3D(intersection1.getPoint());
            Vector tp1 = intersection1.applyToLine(insideTex.get(0), outsideTex.get(0));
            PlaneLineIntersection intersection2 = PlaneLineIntersection.compute(plane, inside.get(0), outside.get(1));

            Point3D p2 = new Point3D(intersection2.getPoint());
            Vector tp2 = intersection2.applyToLine(insideTex.get(0), outsideTex.get(1));
            Point3D[] points = {inside.get(0), p1, p2};
            Vector[] texPoints = {insideTex.get(0), tp1, tp2};
            triangles.add(new Triangle(points, texPoints));
        } else if (inside.size() == 2 && outside.size() == 1) {
            PlaneLineIntersection intersection1 = PlaneLineIntersection.compute(plane, inside.get(0), outside.get(0));
            Point3D p1 = new Point3D(intersection1.getPoint());
            Vector tp1 = intersection1.applyToLine(insideTex.get(0), outsideTex.get(0));

            Point3D[] points1 = {inside.get(0), inside.get(1), p1};
            Vector[] texPoints1 = {insideTex.get(0), insideTex.get(1), tp1};
            triangles.add((new Triangle(points1, texPoints1)));

            PlaneLineIntersection intersection2 = PlaneLineIntersection.compute(plane, inside.get(1), outside.get(0));
            Point3D p2 = new Point3D(intersection2.getPoint());
            Vector tp2 = intersection2.applyToLine(insideTex.get(1), outside.get(0));

            Point3D[] points2 = {inside.get(1), p1, p2};
            Vector[] texPoints2 = {insideTex.get(1), tp1, tp2};
            triangles.add(new Triangle(points2, texPoints2));
        }

        return triangles;
    }

    public void setTexture(Texture texture) {
        this.texture = texture;
    }

    public void setLight(Vector light) {
        this.light = light;
    }
}
