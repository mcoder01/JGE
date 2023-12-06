package com.mcoder.jge.g3d.geom;

import com.mcoder.jge.g3d.core.Vertex;
import com.mcoder.jge.g3d.scene.Camera;
import com.mcoder.jge.math.Vector2D;
import com.mcoder.jge.math.Vector3D;

import java.util.ArrayList;

public record Triangle(Vertex... vertices) {
    public ArrayList<Triangle> clip(Plane plane) {
        ArrayList<Triangle> triangles = new ArrayList<>(2);

        ArrayList<Vertex> inside = new ArrayList<>(3);
        ArrayList<Vertex> outside = new ArrayList<>(3);

        boolean clipScreenPos = true;
        for (Vertex vertex : vertices) {
            if (vertex.getScreenPosition() == null)
                clipScreenPos = false;

            double dist = plane.distanceToPoint(vertex.getPosition());
            if (dist >= 0)
                inside.add(vertex);
            else outside.add(vertex);
        }

        if (inside.size() == 3)
            triangles.add(this);
        else if (inside.size() == 1 && outside.size() == 2) {
            double t1 = planeLineIntersection(plane, inside.get(0).getPosition(), outside.get(0).getPosition());
            Vector3D point1 = Vector3D.lerp(inside.get(0).getPosition(), outside.get(0).getPosition(), t1);
            Vector2D tp1 = Vector2D.lerp(inside.get(0).getTexCoords(), outside.get(0).getTexCoords(), t1);

            double t2 = planeLineIntersection(plane, inside.get(0).getPosition(), outside.get(1).getPosition());
            Vector3D point2 = Vector3D.lerp(inside.get(0).getPosition(), outside.get(1).getPosition(), t2);
            Vector2D tp2 = Vector2D.lerp(inside.get(0).getTexCoords(), outside.get(1).getTexCoords(), t2);

            Vector2D sp1 = null, sp2 = null;
            if (clipScreenPos) {
                sp1 = Vector2D.lerp(inside.get(0).getScreenPosition(), outside.get(0).getScreenPosition(), t1);
                sp2 = Vector2D.lerp(inside.get(0).getScreenPosition(), outside.get(1).getScreenPosition(), t2);
            }

            triangles.add(new Triangle(inside.get(0), new Vertex(point1, tp1, outside.get(0).getNormal(), sp1),
                    new Vertex(point2, tp2, outside.get(1).getNormal(), sp2)));
        } else if (inside.size() == 2 && outside.size() == 1) {
            double t1 = planeLineIntersection(plane, inside.get(0).getPosition(), outside.get(0).getPosition());
            Vector3D point1 = Vector3D.lerp(inside.get(0).getPosition(), outside.get(0).getPosition(), t1);
            Vector2D tp1 = Vector2D.lerp(inside.get(0).getTexCoords(), outside.get(0).getTexCoords(), t1);

            double t2 = planeLineIntersection(plane, inside.get(1).getPosition(), outside.get(0).getPosition());
            Vector3D point2 = Vector3D.lerp(inside.get(1).getPosition(), outside.get(0).getPosition(), t2);
            Vector2D tp2 = Vector2D.lerp(inside.get(1).getTexCoords(), outside.get(0).getTexCoords(), t2);

            Vector2D sp1 = null, sp2 = null;
            if (clipScreenPos) {
                sp1 = Vector2D.lerp(inside.get(0).getScreenPosition(), outside.get(0).getScreenPosition(), t1);
                sp2 = Vector2D.lerp(inside.get(1).getScreenPosition(), outside.get(0).getScreenPosition(), t2);
            }

            Vector2D sp = outside.get(0).getScreenPosition();
            Vertex common = new Vertex(point1, tp1, outside.get(0).getNormal(), sp1);
            triangles.add(new Triangle(inside.get(0), inside.get(1), common));
            triangles.add(new Triangle(inside.get(1), common.copy(), new Vertex(point2, tp2,
                    outside.get(0).getNormal().copy(), sp2)));
        }

        return triangles;
    }

    private double planeLineIntersection(Plane plane, Vector3D a, Vector3D b) {
        double pd = plane.pos().dot(plane.normal());
        double ad = a.dot(plane.normal());
        double bd = b.dot(plane.normal());
        return (pd-ad)/(bd-ad);
    }

    public boolean isVisible() {
        for (Vertex v : vertices)
            if (v.getPosition().dot(v.getNormal()) < 0)
                return true;
        return false;
    }
}