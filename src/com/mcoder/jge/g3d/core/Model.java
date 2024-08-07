package com.mcoder.jge.g3d.core;

import com.mcoder.jge.math.Vector;
import com.mcoder.jge.math.Vector2D;
import com.mcoder.jge.math.Vector3D;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

public class Model {
    private final ArrayList<Vector3D> points;
    private final ArrayList<Vector2D> texCoords;
    private final ArrayList<OBJIndex[]> faces;

    private Model() {
        points = new ArrayList<>();
        texCoords = new ArrayList<>();
        faces = new ArrayList<>();
    }

    public ArrayList<Vector3D> getPoints() { return points; }

    public ArrayList<Vector2D> getTexCoords() {
        return texCoords;
    }

    public ArrayList<OBJIndex[]> getTriangles() {
        ArrayList<OBJIndex[]> triangles = new ArrayList<>();
        for (OBJIndex[] face : faces)
            if (face.length == 4)
                for (int i = 0; i < face.length-2; i++)
                    triangles.add(new OBJIndex[]{face[0], face[1 + i], face[2 + i]});
            else triangles.add(face);
        return triangles;
    }

    public static Model loadFromFile(String modelFile) {
        Model model = new Model();
        int[] defaultTexIndices = {3, 0, 1, 2};
        try (BufferedReader reader = new BufferedReader(new FileReader(modelFile))) {
            while (reader.ready()) {
                String[] info = reader.readLine().split(" ");
                switch (info[0]) {
                    case "v" -> model.points.add((Vector3D) parseVector(info));
                    case "vt" -> model.texCoords.add((Vector2D) parseVector(info));
                    case "f" -> {
                        OBJIndex[] face = new OBJIndex[info.length-1];
                        for (int i = 0; i < face.length; i++) {
                            String[] indices = info[i + 1].split("/");
                            face[i] = new OBJIndex();
                            face[i].setPointIndex(Integer.parseInt(indices[0])-1);
                            if (indices.length > 1)
                                face[i].setTexCoordsIndex(Integer.parseInt(indices[1]) - 1);
                            else face[i].setTexCoordsIndex(defaultTexIndices[i]);
                        }
                        model.faces.add(face);
                    }
                }
            }
        } catch (IOException e) {
            System.out.println("Unable to load model from " + modelFile);
        }

        if (model.texCoords.isEmpty())
            model.texCoords.addAll(Arrays.asList(
                    new Vector2D(0, 0),
                    new Vector2D(1, 0),
                    new Vector2D(1, 1),
                    new Vector2D(0, 1)
            ));

        System.out.println("Loaded " + model.faces.size() + " faces from " + modelFile);
        return model;
    }

    private static Vector parseVector(String[] info) {
        double[] values = parseDoubles(info);
        if (values.length == 2)
            return new Vector2D(values[0], values[1]);
        return new Vector3D(values[0], values[1], values[2]);
    }

    private static double[] parseDoubles(String[] info) {
        double[] values = new double[info.length-1];
        for (int i = 0; i < values.length; i++)
            values[i] = Double.parseDouble(info[i+1]);
        return values;
    }
}
