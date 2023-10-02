package com.mcoder.jge.g3d.geom;

import com.mcoder.jge.g3d.geom.plane.Plane;
import com.mcoder.jge.g3d.geom.plane.PlaneLineIntersection;
import com.mcoder.jge.g3d.scene.Camera;
import com.mcoder.jge.math.Vector;

import java.util.ArrayList;

public class Triangle {
    private final Vector[] points, texPoints;
    private Vector normal;

    public Triangle(Vector[] points, Vector[] texPoints) {
        this.points = points;
        this.texPoints = texPoints;
        calculateNormal();
    }

    public Triangle(Vector[] points, Vector[] texPoints, Vector normal) {
        this(points, texPoints);
        this.normal = normal;
    }

    public void calculateNormal() {
        Vector v1 = Vector.sub(points[1], points[0]);
        Vector v2 = Vector.sub(points[2], points[0]);
        normal = v1.cross(v2).normalize();
    }

    public ArrayList<Triangle> clip(Plane plane) {
        ArrayList<Triangle> triangles = new ArrayList<>(2);

        ArrayList<Vector> inside = new ArrayList<>(3);
        ArrayList<Vector> outside = new ArrayList<>(3);
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
            Vector tp1 = intersection1.applyToLine(insideTex.get(0), outsideTex.get(0));
            PlaneLineIntersection intersection2 = PlaneLineIntersection.compute(plane, inside.get(0), outside.get(1));
            Vector tp2 = intersection2.applyToLine(insideTex.get(0), outsideTex.get(1));

            Vector[] points = {inside.get(0), intersection1.getPoint(), intersection2.getPoint()};
            Vector[] texPoints = {insideTex.get(0), tp1, tp2};
            triangles.add(new Triangle(points, texPoints, normal));
        } else if (inside.size() == 2 && outside.size() == 1) {
            PlaneLineIntersection intersection1 = PlaneLineIntersection.compute(plane, inside.get(0), outside.get(0));
            Vector tp1 = intersection1.applyToLine(insideTex.get(0), outsideTex.get(0));

            Vector[] points1 = {inside.get(0), inside.get(1), intersection1.getPoint()};
            Vector[] texPoints1 = {insideTex.get(0), insideTex.get(1), tp1};
            triangles.add(new Triangle(points1, texPoints1, normal));

            PlaneLineIntersection intersection2 = PlaneLineIntersection.compute(plane, inside.get(1), outside.get(0));
            Vector tp2 = intersection2.applyToLine(insideTex.get(1), outsideTex.get(0));

            Vector[] points2 = {inside.get(1), intersection1.getPoint(), intersection2.getPoint()};
            Vector[] texPoints2 = {insideTex.get(1), tp1, tp2};
            triangles.add(new Triangle(points2, texPoints2, normal));
        }

        return triangles;
    }

    public boolean isVisible(Camera camera) {
        Vector cameraRay = Vector.sub(points[0], camera.getPos());
        return cameraRay.dot(normal) < 0;
    }

    public Vector[] getPoints() {
        return points;
    }

    public Vector[] getTexPoints() {
        return texPoints;
    }

    public Vector getNormal() {
        return normal;
    }
}
