package com.mcoder.jgel.g3d.render;

import com.mcoder.jgel.g3d.geom.solid.Point3D;
import com.mcoder.jgel.g3d.geom.plane.Surface;
import com.mcoder.jgel.math.Vector;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class Model {
    private final ArrayList<Surface> surfaces;

    private Model() {
        surfaces = new ArrayList<>();
    }

    public static Model loadFromFile(String modelFile) {
        Model model = new Model();
        ArrayList<Point3D> points = new ArrayList<>();
        ArrayList<Vector> texPoints = new ArrayList<>();

        try(BufferedReader reader = new BufferedReader(new FileReader(modelFile))) {
            while (reader.ready()) {
                String[] info = reader.readLine().split(" ");
                switch (info[0]) {
                    case "v" -> points.add(parsePoint(info));
                    case "vt" -> texPoints.add(parseVector(info));
                    case "f" -> {
                        Point3D[] vertexes = new Point3D[info.length-1];
                        Vector[] texVertexes = new Vector[vertexes.length];
                        for (int i = 0; i < vertexes.length; i++) {
                            String[] indexes = info[i + 1].split("/");
                            vertexes[i] = points.get(Integer.parseInt(indexes[0])-1).copy();
                            if (indexes.length == 2)
                                texVertexes[i] = texPoints.get(Integer.parseInt(indexes[1])-1).copy();
                            else texVertexes[i] = new Vector();
                        }

                        model.surfaces.add(new Surface(vertexes, texVertexes));
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return model;
    }

    public Triangle[] obtainTriangles(int texW, int texH) {
        Triangle[] triangles = new Triangle[surfaces.size()];
        for (int i = 0; i < triangles.length; i++) {
            Surface surface = surfaces.get(i);
            triangles[i] = new Triangle(surface.obtainPoints(), surface.obtainTexPoints(texW, texH));
        }

        return triangles;
    }

    private static Vector parseVector(String[] info) {
        double[] values = parseDoubles(info);
        return new Vector(values[0], values[1], values[2]);
    }

    private static Point3D parsePoint(String[] info) {
        double[] values = parseDoubles(info);
        return new Point3D(values[0], values[1], values[2]);
    }

    private static double[] parseDoubles(String[] info) {
        double[] values = new double[3];
        for (int i = 0; i < info.length-1; i++)
            values[i] = Double.parseDouble(info[i+1]);
        return values;
    }
}
