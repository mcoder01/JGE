package com.mcoder.jge.g3d.render;

import com.mcoder.jge.math.Vector;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

public class Model {
    private final ArrayList<Vector> points, texPoints;
    private final ArrayList<int[]> map, texMap;

    private Model() {
        points = new ArrayList<>();
        map = new ArrayList<>();
        texPoints = new ArrayList<>();
        texMap = new ArrayList<>();
    }

    public int getFaces() {
        return map.size();
    }

    public ArrayList<Vector> getPoints() {
        return points;
    }

    public ArrayList<Vector> getTexPoints() {
        return texPoints;
    }

    public ArrayList<int[]> getMap() {
        return map;
    }

    public ArrayList<int[]> getTexMap() {
        return texMap;
    }

    public static Model loadFromFile(String modelFile) {
        Model model = new Model();
        try(BufferedReader reader = new BufferedReader(new FileReader(modelFile))) {
            while (reader.ready()) {
                String[] info = reader.readLine().split(" ");
                switch (info[0]) {
                    case "v" -> model.points.add(parseVector(info));
                    case "vt" -> model.texPoints.add(parseVector(info));
                    case "f" -> {
                        int[] map = new int[info.length-1];
                        int[] texMap = new int[map.length];
                        for (int i = 0; i < map.length; i++) {
                            String[] indexes = info[i + 1].split("/");
                            map[i] = Integer.parseInt(indexes[0])-1;
                            if (indexes.length == 2)
                                texMap[i] = Integer.parseInt(indexes[1])-1;
                        }

                        model.map.add(map);
                        model.texMap.add(texMap);
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        if (model.texPoints.size() == 0)
            model.texPoints.addAll(Arrays.asList(
                    new Vector(0, 1),
                    new Vector(0, 0),
                    new Vector(1, 0),
                    new Vector(1, 1)
            ));

        if (model.texMap.size() == 0)
            for (int i = 0; i < model.getFaces(); i++)
                if (i%2 == 0)
                    model.texMap.add(new int[] {0, 1, 2});
                else model.texMap.add(new int[] {0, 2, 3});

        return model;
    }

    private static Vector parseVector(String[] info) {
        double[] values = parseDoubles(info);
        return new Vector(values[0], values[1], values[2]);
    }

    private static double[] parseDoubles(String[] info) {
        double[] values = new double[3];
        for (int i = 0; i < info.length-1; i++)
            values[i] = Double.parseDouble(info[i+1]);
        return values;
    }
}
