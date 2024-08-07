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
        rasterizer = new TriangleRasterizer(world);
    }

    public void drawSolid(Solid solid) {
        Vector3D[] points = new Vector3D[solid.getModel().getPoints().size()];
        Camera camera = world.getCamera();
        ThreadPool.executeInParallel(points.length, i ->  {
            points[i] = solid.getModel().getPoints().get(i).copy();
            Point3D p3d = new Point3D(points[i]);
            p3d.rotate(solid.getRot());
            p3d.move(Vector3D.sub(solid.getPos(), camera.getPos()));
            p3d.rotate(Vector3D.scale(camera.getRot(), -1));
        });

        Vector2D[] texPoints = new Vector2D[solid.getModel().getTexCoords().size()];
        ThreadPool.executeInParallel(texPoints.length, i -> {
            Vector2D tp = solid.getModel().getTexCoords().get(i);
            texPoints[i] = Vector2D.scale(tp, solid.getTexture().getSize());
        });

        ArrayList<OBJIndex[]> faces = solid.getModel().getTriangles();
        ArrayList<Triangle> triangles = new ArrayList<>();
        Vector3D[] normals = calculateNormals(faces, points);
        for (OBJIndex[] face : faces) {
            Vertex[] triangleVertices = new Vertex[face.length];
            for (int i = 0; i < face.length; i++)
                triangleVertices[i] = new Vertex(
                        points[face[i].getPointIndex()],
                        texPoints[face[i].getTexCoordsIndex()],
                        normals[face[i].getPointIndex()].normalize()
                );

            Triangle triangle = new Triangle(triangleVertices);
            if (triangle.isVisible()) triangles.add(triangle);
        }

        ArrayList<Triangle> clipped = new ArrayList<>();
        for (Triangle triangle : triangles)
            clipped.addAll(clipTriangles(triangle, camera.getDepthPlanes()));

        ArrayList<Triangle> finalTriangles = new ArrayList<>();
        for (Triangle triangle : clipped) {
            for (Vertex v : triangle.vertices()) {
                Vector3D proj = new Point3D(v.getPosition()).project(world.getScreen().getFOV(),
                        world.getScreen().getWidth(), world.getScreen().getHeight());
                v.setScreenPosition(proj);
            }

            finalTriangles.addAll(clipTriangles(triangle, camera.getSidePlanes()));
        }

        for (Triangle triangle : finalTriangles)
            rasterizer.drawTriangle(triangle, solid);
    }

    private Vector3D[] calculateNormals(ArrayList<OBJIndex[]> faces, Vector3D[] points) {
        Vector3D[] normals = new Vector3D[points.length];
        for (OBJIndex[] face : faces) {
            Vector3D v1 = Vector3D.sub(points[face[1].getPointIndex()], points[face[0].getPointIndex()]);
            Vector3D v2 = Vector3D.sub(points[face[2].getPointIndex()], points[face[0].getPointIndex()]);
            Vector3D normal = v1.cross(v2).normalize();

            for (OBJIndex index : face)
                if (normals[index.getPointIndex()] == null)
                    normals[index.getPointIndex()] = normal.copy();
                else normals[index.getPointIndex()].add(normal);
        }

        return normals;
    }

    private LinkedList<Triangle> clipTriangles(Triangle toClip, Plane[] planes) {
        LinkedList<Triangle> queue = new LinkedList<>();
        queue.add(toClip);
        for (Plane plane : planes) {
            int queueSize = queue.size();
            for (int i = 0; i < queueSize; i++) {
                Triangle triangle = queue.poll();
                assert triangle != null;
                queue.addAll(triangle.clip(plane));
            }
        }

        return queue;
    }
}
