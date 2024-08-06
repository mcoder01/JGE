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
    private ArrayList<Vector3D> positions;
    private ArrayList<Vector2D> texCoords;

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
            p3d.move(Vector3D.sub(solid.getWorldPos(), camera.getWorldPos()));
            p3d.rotate(Vector3D.scale(camera.getRot(), -1));
        });

        Vector2D[] texPoints = new Vector2D[solid.getModel().getTexCoords().size()];
        ThreadPool.executeInParallel(texPoints.length, i -> {
            Vector2D tp = solid.getModel().getTexCoords().get(i);
            texPoints[i] = new Vector2D(tp.getX()*solid.getTexture().getWidth(),
                    tp.getY()*solid.getTexture().getHeight());
        });

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

    private Vector3D[] calculateNormals(Model model, Vector3D[] points) {
        Vector3D[] normals = new Vector3D[points.length];
        for (OBJIndex[] face : model.getFaces()) {
            Vector3D v1 = Vector3D.sub(points[face[1].getPointIndex()], points[face[0].getPointIndex()]);
            Vector3D v2 = Vector3D.sub(points[face[2].getPointIndex()], points[face[0].getPointIndex()]);
            Vector3D normal = v1.cross(v2).normalize();

            for (OBJIndex index : face)
                //if (normals[index.getPointIndex()] == null)
                    normals[index.getPointIndex()] = normal.copy();
                //else normals[index.getPointIndex()].add(normal);
        }

        //ThreadPool.executeInParallel(normals.length, i -> normals[i].normalize());
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

    public void addSolid(Solid solid) {
        positions.addAll(solid.getModel().getPoints());
        for (Vector2D coord : solid.getModel().getTexCoords())
            texCoords.add(coord.scale(solid.getTexture().getSize()));
    }

    public void execute() {
        Camera camera = world.getCamera();
        double[][] points = new double[positions.size()][3];
        for (int i = 0; i < points.length; i++)
            points[i] = positions.get(i).getValues();

        points = transformToCamera(camera.getPos().getValues(),
                camera.getRot().getValues(), points);


    }

    private native double[][] transformToCamera(double[] cameraPos, double[] cameraRot, double[][] points);
}
