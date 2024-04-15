package com.mcoder.jge.g3d.render;

import com.mcoder.jge.g3d.core.*;
import com.mcoder.jge.g3d.geom.Triangle;
import com.mcoder.jge.g3d.geom.Plane;
import com.mcoder.jge.g3d.scene.Camera;
import com.mcoder.jge.g3d.scene.World;
import com.mcoder.jge.math.Vector2D;
import com.mcoder.jge.math.Vector3D;

import java.util.ArrayList;
import java.util.Collections;

public class Pipeline {
    static {
        System.loadLibrary("shader");
    }

    private final World world;
    private final TriangleRasterizer rasterizer;

    public Pipeline(World world) {
        this.world = world;
        rasterizer = new TriangleRasterizer(world.getScreen());
    }

    public void drawSolid(Solid solid) {
        Vector3D[] points = new Vector3D[solid.getModel().getPoints().size()];
        Camera camera = world.getCamera();
        for (int i = 0; i < points.length; i++) {
            points[i] = solid.getModel().getPoints().get(i).copy();
            Point3D p3d = new Point3D(points[i]);
            p3d.rotate(solid.getRot());
            p3d.move(solid.getWorldPos());
        }

        Vector2D[] texPoints = new Vector2D[solid.getModel().getTexCoords().size()];
        for (int i = 0; i < texPoints.length; i++) {
            Vector2D tp = solid.getModel().getTexCoords().get(i);
            texPoints[i] = new Vector2D(tp.getX()*solid.getTexture().getWidth(),
                    tp.getY()*solid.getTexture().getHeight());
        }

        for (Vector3D point : points) {
            Point3D p3d = new Point3D(point);
            p3d.move(Vector3D.mult(camera.getWorldPos(), -1));
            p3d.rotate(Vector3D.mult(camera.getRot(), -1));
        }

        Vector3D[] normals = calculateNormals(solid.getModel(), points);
        ArrayList<OBJIndex[]> faces = solid.getModel().getTriangles();
        ArrayList<Triangle> triangles = new ArrayList<>();
        for (OBJIndex[] face : faces) {
            Vertex[] triangleVertices = new Vertex[face.length];
            for (int j = 0; j < face.length; j++) {
                triangleVertices[j] = new Vertex();
                int pointIndex = face[j].getPointIndex();
                triangleVertices[j].setPosition(points[pointIndex]);
                triangleVertices[j].setTexCoords(texPoints[face[j].getTexCoordsIndex()]);
                triangleVertices[j].setNormal(normals[pointIndex]);
            }

            Triangle triangle = new Triangle(triangleVertices);
            if (triangle.isVisible())
                triangles.add(triangle);
        }

        ArrayList<Triangle> clipped = new ArrayList<>();
        for (Triangle triangle : triangles)
            Collections.addAll(clipped, clipTriangle(triangle, camera.getDepthPlanes(), false));

        ArrayList<Triangle> finalTriangles = new ArrayList<>();
        for (Triangle triangle : clipped) {
            for (Vertex v : triangle.vertices()) {
                Vector2D proj = new Point3D(v.getPosition()).project(world.getScreen().getFOV(),
                        world.getScreen().getWidth(), world.getScreen().getHeight());
                v.setScreenPosition(proj);
            }

            Collections.addAll(finalTriangles, clipTriangle(triangle, camera.getSidePlanes(), true));
        }

        finalTriangles.forEach(triangle ->
            rasterizer.drawTriangle(triangle, solid.getTexture(), solid.getShader()));
    }

    private Vector3D[] calculateNormals(Model model, Vector3D[] points) {
        Vector3D[] normals = new Vector3D[points.length];
        for (OBJIndex[] face : model.getFaces()) {
            Vector3D v1 = Vector3D.sub(points[face[1].getPointIndex()], points[face[0].getPointIndex()]);
            Vector3D v2 = Vector3D.sub(points[face[2].getPointIndex()], points[face[0].getPointIndex()]);
            Vector3D normal = v1.cross(v2).normalize();

            for (OBJIndex index : face)
                if (normals[index.getPointIndex()] == null)
                    normals[index.getPointIndex()] = normal.copy();
                else normals[index.getPointIndex()].add(normal);
        }

        for (Vector3D normal : normals)
            normal.normalize();
        return normals;
    }

    private native Triangle[] clipTriangle(Triangle toClip, Plane[] planes, boolean clipProjection);
}
