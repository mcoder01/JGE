package com.mcoder.jge.g3d.scene;

import com.mcoder.jge.g3d.render.Solid;
import com.mcoder.jge.screen.Display;
import com.mcoder.jge.math.Vector;

import java.awt.*;
import java.util.LinkedList;

public class World extends Display {
    private static World instance;

    private final Camera camera;
    private final Vector light;
    private final LinkedList<Solid> solids;

    private World() {
        super();
        solids = new LinkedList<>();
        light = new Vector(0, 0, 1);
        camera = new Camera(0, 0, 0);
        addListener(camera);

        // Testing
        /*Model cubeModel = Model.loadFromFile("res/model/cube.obj");
        Texture texture = new Texture("cobblestone.png");
        for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++)
                for (int k = -1; k <= 1; k++) {
                    Solid cube = new Solid(cubeModel, i, 0, k);
                    cube.setTexture(texture);
                    solids.add(cube);
                }*/

        /*Solid cube = new Solid(cubeModel, 0, 0, 0);
        cube.setTexture(texture);
        solids.add(cube);*/

        /*Model mountainsModel = Model.loadFromFile("res/model/mountains.obj");
        solids.add(new Solid(mountainsModel, 0, 0, 0));*/

        /*Model shipModel = Model.loadFromFile("res/model/ship.obj");
        solids.add(new Solid(shipModel, 0, 0, 0));*/
    }

    @Override
    public void update() {
        camera.update();
        for (Solid stuff : solids)
            stuff.update();
    }

    @Override
    public void show(Graphics2D g2d) {
        for (Solid stuff : solids)
            stuff.show(g2d);
    }

    public void addSolid(Solid solid) {
        solids.add(solid);
    }

    public void removeSolid(Solid solid) {
        solids.remove(solid);
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
