package com.mcoder.jge.g3d.render;

import com.mcoder.jge.g3d.model.*;
import com.mcoder.jge.g3d.scene.Solid;
import com.mcoder.jge.util.Point3D;
import com.mcoder.jge.g3d.geom.Triangle;
import com.mcoder.jge.g3d.geom.Plane;
import com.mcoder.jge.g3d.geom.Vertex;
import com.mcoder.jge.g3d.scene.Camera;
import com.mcoder.jge.g3d.World;
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
        ThreadPool.getInstance().execute(() -> {
            for (int i = 0; i < points.length; i++) {
                points[i] = solid.getModel().getPoints().get(i).copy();
                Point3D p3d = new Point3D(points[i]);
                p3d.rotate(solid.getRot());
                p3d.move(Vector3D.sub(solid.getPos(), camera.getPos()));
                p3d.rotate(Vector3D.scale(camera.getRot(), -1));
            }
        });

        Vector2D[] texPoints = new Vector2D[solid.getModel().getTexCoords().size()];
        ThreadPool.getInstance().execute(() -> {
            for (int i = 0; i < texPoints.length; i++) {
                Vector2D tp = solid.getModel().getTexCoords().get(i);
                texPoints[i] = Vector2D.scale(tp, solid.getTexture().getSize());
            }
        });

        ThreadPool.getInstance().waitForAllTasks();

        LinkedList<Triangle> triangles = new LinkedList<>();
        ArrayList<OBJIndex[]> faces = solid.getModel().getTriangles();
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

        clipTriangles(triangles, camera.getPlanes());
        for (Triangle triangle : triangles) {
            for (Vertex v : triangle.vertices())
                v.project(world.getScreen());
            rasterizer.drawTriangle(triangle, solid);
        }
    }

    private Vector3D[] calculateNormals(ArrayList<OBJIndex[]> faces, Vector3D[] points) {
        Vector3D[] normals = new Vector3D[points.length];
        ThreadPool.getInstance().executeInParallel(faces.size(), i -> {
            OBJIndex[] face = faces.get(i);
            Vector3D v1 = Vector3D.sub(points[face[1].getPointIndex()], points[face[0].getPointIndex()]);
            Vector3D v2 = Vector3D.sub(points[face[2].getPointIndex()], points[face[0].getPointIndex()]);
            Vector3D normal = v1.cross(v2).normalize();

            for (OBJIndex index : face)
                synchronized (normals) {
                    if (normals[index.getPointIndex()] == null)
                        normals[index.getPointIndex()] = normal.copy();
                    else normals[index.getPointIndex()].add(normal);
                }
        });

        return normals;
    }

    private void clipTriangles(LinkedList<Triangle> queue, Plane[] planes) {
        for (Plane plane : planes) {
            int queueSize = queue.size();
            for (int i = 0; i < queueSize; i++) {
                Triangle triangle = queue.poll();
                assert triangle != null;
                triangle.clip(plane, queue);
            }
        }
    }
}
