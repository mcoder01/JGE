package com.mcoder.jge.g3d.scene;

import com.mcoder.jge.g3d.render.Model;
import com.mcoder.jge.g3d.render.Solid;
import com.mcoder.jge.screen.Display;
import com.mcoder.jge.math.Vector;
import com.mcoder.jge.util.Texture;

import java.awt.*;
import java.util.LinkedList;

public class World extends Display {
    private final Camera camera;
    private final Vector light;
    private final LinkedList<Solid> solids;

    public World() {
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
                    Solid cube = new Solid(cubeModel, i, 0, k, this);
                    cube.setTexture(texture);
                    solids.add(cube);
                }*/

        /*Solid cube = new Solid(cubeModel, 0, 0, 0);
        cube.setTexture(texture);
        solids.add(cube);*/

        Model mountainsModel = Model.loadFromFile("res/model/mountains.obj");
        solids.add(new Solid(mountainsModel, 0, 0, 0, this));

        /*Model shipModel = Model.loadFromFile("res/model/ship.obj");
        solids.add(new Solid(shipModel, 0, 0, 0, this));*/
    }

    @Override
    public void tick() {
        camera.update();
        for (Solid stuff : solids)
            stuff.tick();
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
}
