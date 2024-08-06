package com.mcoder.jge.g3d.core;

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

    public int numberOfFaces() {
        return faces.size();
    }

    public ArrayList<Vector3D> getPoints() { return points; }

    public ArrayList<Vector2D> getTexCoords() {
        return texCoords;
    }

    public ArrayList<OBJIndex[]> getTriangles() {
        ArrayList<OBJIndex[]> triangles = new ArrayList<>();
        for (OBJIndex[] face : faces)
            for (int i = 0; i < face.length-2; i++)
                triangles.add(new OBJIndex[]{face[0], face[1 + i], face[2 + i]});
        return triangles;
    }

    public ArrayList<OBJIndex[]> getFaces() {
        return faces;
    }

    public static Model loadFromFile(String modelFile) {
        Model model = new Model();
        int[] defaultTexIndices = {3, 0, 1, 2};
        try (BufferedReader reader = new BufferedReader(new FileReader(modelFile))) {
            while (reader.ready()) {
                String[] info = reader.readLine().split(" ");
                switch (info[0]) {
                    case "v" -> model.points.add(parseVector(info));
                    case "vt" -> model.texCoords.add(parseVector(info));
                    case "f" -> {
                        OBJIndex[] face = new OBJIndex[info.length-1];
                        for (int i = 0; i < face.length; i++) {
                            String[] indexes = info[i + 1].split("/");
                            face[i] = new OBJIndex();
                            face[i].setPointIndex(Integer.parseInt(indexes[0])-1);
                            if (indexes.length > 1)
                                face[i].setTexCoordsIndex(Integer.parseInt(indexes[1]) - 1);
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

    private static Vector3D parseVector(String[] info) {
        double[] values = parseDoubles(info);
        return new Vector3D(values[0], values[1], values[2]);
    }

    private static double[] parseDoubles(String[] info) {
        double[] values = new double[3];
        for (int i = 0; i < info.length-1; i++)
            values[i] = Double.parseDouble(info[i+1]);
        return values;
    }
}
