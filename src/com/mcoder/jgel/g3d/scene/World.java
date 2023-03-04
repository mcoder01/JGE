package com.mcoder.jgel.g3d.scene;

import com.mcoder.jgel.g3d.render.Model;
import com.mcoder.jgel.g3d.render.Solid;
import com.mcoder.jgel.scene.Display;
import com.mcoder.jgel.math.Vector;
import com.mcoder.jgel.scene.Screen;
import com.mcoder.jgel.util.Texture;

import java.awt.*;
import java.util.LinkedList;

public class World extends Display {
    private static World instance;

    private final Camera camera;
    private final Vector light;
    private final LinkedList<Solid> stuffs;

    private World() {
        super();
        light = new Vector(0, 0, 1);
        camera = new Camera(0, 0, -4);
        addListener(camera);
        stuffs = new LinkedList<>();

        // Testing
        /*Model cubeModel = Model.loadFromFile("res/model/cube.obj");
        Texture texture = new Texture("cobblestone.png");
        for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++)
                for (int k = -1; k <= 1; k++) {
                    Solid cube = new Solid(cubeModel, i, 0, k);
                    cube.setTexture(texture);
                    stuffs.add(cube);
                }

        Solid cube = new Solid(cubeModel, 0, 0, 0);
        cube.setTexture(texture);
        stuffs.add(cube);*/

        Model mountainsModel = Model.loadFromFile("res/model/mountains.obj");
        stuffs.add(new Solid(mountainsModel, 0, 0, 0));

        /*Model shipModel = Model.loadFromFile("res/model/ship.obj");
        stuffs.add(new Solid(shipModel, 0, 0, 0));*/
    }

    @Override
    public void update() {
        camera.update();
        for (Solid stuff : stuffs)
            stuff.update();
    }

    @Override
    public void show(Graphics2D g2d) {
        for (Solid stuff : stuffs)
            stuff.show(g2d);
    }

    public Camera getCamera() {
        return camera;
    }

    public Vector getLight() {
        return light;
    }

    public static World getInstance() {
        if (instance == null)
            instance = new World();
        return instance;
    }
}
