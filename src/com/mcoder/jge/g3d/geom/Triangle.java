package com.mcoder.jge.g3d.geom;

import com.mcoder.jge.math.Vector3D;

import java.util.ArrayList;

public record Triangle(Vertex... vertices) {
    public ArrayList<Triangle> clip(Plane plane) {
        ArrayList<Triangle> triangles = new ArrayList<>(2);

        Vertex[] inside = new Vertex[3];
        Vertex[] outside = new Vertex[3];
        int insideSize = 0, outsideSize = 0;

        for (Vertex vertex : vertices) {
            double dist = plane.distanceToPoint(vertex.getPosition());
            if (dist >= 0) inside[insideSize++] = vertex;
            else outside[outsideSize++] = vertex;
        }

        if (insideSize == 3)
            triangles.add(this);
        else if (insideSize == 1 && outsideSize == 2) {
            double t1 = planeLineIntersection(plane, inside[0].getPosition(), outside[0].getPosition());
            double t2 = planeLineIntersection(plane, inside[0].getPosition(), outside[1].getPosition());
            triangles.add(new Triangle(inside[0], Vertex.lerp(inside[0], outside[0], t1),
                    Vertex.lerp(inside[0], outside[1], t2)));
        } else if (insideSize == 2 && outsideSize == 1) {
            double t1 = planeLineIntersection(plane, inside[0].getPosition(), outside[0].getPosition());
            double t2 = planeLineIntersection(plane, inside[1].getPosition(), outside[0].getPosition());
            Vertex common = Vertex.lerp(inside[0], outside[0], t1);
            triangles.add(new Triangle(inside[0], inside[1], common));
            triangles.add(new Triangle(inside[1], common.copy(), Vertex.lerp(inside[1], outside[0], t2)));
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