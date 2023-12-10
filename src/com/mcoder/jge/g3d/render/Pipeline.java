package com.mcoder.jge.g3d.render;

import com.mcoder.jge.g3d.core.*;
import com.mcoder.jge.g3d.geom.Triangle;
import com.mcoder.jge.g3d.geom.Plane;
import com.mcoder.jge.g3d.scene.Camera;
import com.mcoder.jge.g3d.scene.World;
import com.mcoder.jge.math.Vector2D;
import com.mcoder.jge.math.Vector3D;
import com.mcoder.jge.util.ThreadPool;

import java.util.ArrayList;
import java.util.LinkedList;

public class Pipeline {
    private final World world;
    private final TriangleRasterizer rasterizer;

    public Pipeline(World world) {
        this.world = world;
        rasterizer = new TriangleRasterizer(world.getScreen());
    }

    public void drawSolid(Solid solid) {
        Vector3D[] points = new Vector3D[solid.getModel().getPoints().size()];
        Camera camera = world.getCamera();
        ThreadPool.executeInParallel(points.length, i ->  {
            points[i] = solid.getModel().getPoints().get(i).copy();
            Point3D p3d = new Point3D(points[i]);
            p3d.rotate(solid.getRot());
            p3d.move(solid.getWorldPos());
        });

        Vector2D[] texPoints = new Vector2D[solid.getModel().getTexCoords().size()];
        ThreadPool.executeInParallel(texPoints.length, i -> {
            Vector2D tp = solid.getModel().getTexCoords().get(i);
            texPoints[i] = new Vector2D(tp.getX()*solid.getTexture().getWidth(),
                    tp.getY()*solid.getTexture().getHeight());
        });

        ThreadPool.executeInParallel(points.length, i -> {
            Point3D p3d = new Point3D(points[i]);
            p3d.move(Vector3D.mult(camera.getWorldPos(), -1));
            p3d.rotate(Vector3D.mult(camera.getRot(), -1));
        });

        Vector3D[] normals = calculateNormals(solid.getModel(), points);
        ArrayList<OBJIndex[]> faces = solid.getModel().getTriangles();
        ArrayList<Triangle> triangles = new ArrayList<>();
        ThreadPool.executeInParallel(faces.size(), i -> {
            OBJIndex[] face = faces.get(i);
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
                synchronized (triangles) {
                    triangles.add(triangle);
                }
        });

        ArrayList<Triangle> clipped = new ArrayList<>();
        ThreadPool.executeInParallel(triangles.size(), i -> {
            LinkedList<Triangle> clips = clipTriangles(triangles.get(i), camera.getDepthPlanes());
            synchronized (clipped) {
                clipped.addAll(clips);
            }
        });

        ArrayList<Triangle> finalTriangles = new ArrayList<>();
        ThreadPool.executeInParallel(clipped.size(), i -> {
            Triangle triangle = clipped.get(i);
            for (Vertex v : triangle.vertices()) {
                Vector3D proj = new Point3D(v.getPosition()).project(world.getScreen().getFOV(),
                        world.getScreen().getWidth(), world.getScreen().getHeight());
                v.setScreenPosition(proj);
            }

            LinkedList<Triangle> clips = clipTriangles(triangle, camera.getSidePlanes());
            synchronized (finalTriangles) {
                finalTriangles.addAll(clips);
            }
        });

        for (Triangle triangle : finalTriangles)
            rasterizer.drawTriangle(triangle, solid.getTexture(), solid.getShader());
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

        ThreadPool.executeInParallel(normals.length, i -> normals[i].normalize());
        return normals;
    }

    private LinkedList<Triangle> clipTriangles(Triangle toClip, Plane[] planes) {
        LinkedList<Triangle> currQueue = new LinkedList<>();
        currQueue.add(toClip);
        for (Plane plane : planes) {
            LinkedList<Triangle> nextQueue = new LinkedList<>();
            while (!currQueue.isEmpty()) {
                Triangle triangle = currQueue.poll();
                nextQueue.addAll(triangle.clip(plane));
            }

            currQueue = nextQueue;
        }

        return currQueue;
    }
}
