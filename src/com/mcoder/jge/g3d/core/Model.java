package com.mcoder.jge.g3d.core;

import com.mcoder.jge.math.Matrix;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class Model {
    private final Matrix points, texCoords;
    private final ArrayList<OBJIndex[]> faces;

    private Model(Matrix points, Matrix texCoords, ArrayList<OBJIndex[]> faces) {
        this.points = points;
        this.texCoords = texCoords;
        this.faces = faces;
    }

    public int numberOfFaces() {
        return faces.size();
    }

    public Matrix getPoints() { return points; }

    public Matrix getTexCoords() {
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
        final int[] defaultTexIndices = {3, 0, 1, 2};

        ArrayList<Double> points = new ArrayList<>();
        ArrayList<Double> texCoords = new ArrayList<>();
        ArrayList<OBJIndex[]> faces = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(modelFile))) {
            while (reader.ready()) {
                String[] info = reader.readLine().split(" ");
                switch (info[0]) {
                    case "v" -> parseVertex(info, points);
                    case "vt" -> parseVertex(info, texCoords);
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
                        faces.add(face);
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println("Loaded " + faces.size() + " faces from " + modelFile);
        return new Model(
                new Matrix(points.size()/3, 3, points),
                new Matrix(texCoords.size()/2, 2, texCoords),
                faces
        );
    }

    private static void parseVertex(String[] info, ArrayList<Double> array) {
        for (int i = 1; i < info.length; i++)
            array.add(Double.parseDouble(info[i]));
    }
}
