package com.mcoder.jge.g3d.scene;

import com.mcoder.jge.g3d.render.Model;
import com.mcoder.jge.g3d.render.Solid;
import com.mcoder.jge.math.Vector;
import com.mcoder.jge.screen.View;

public class World extends View {
    private final Camera camera;
    private final Vector light;

    public World() {
        super();
        light = new Vector(0, 0, 1);
        camera = new Camera(0, 0, 0);
    }

    @Override
    public void setup() {
        add(camera);

        // Testing
        /*Model cubeModel = Model.loadFromFile("res/model/cube.obj");
        Texture texture = new Texture("cobblestone.png");
        for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++)
                for (int k = -1; k <= 1; k++) {
                    Solid cube = new Solid(cubeModel, i, 0, k, this);
                    cube.setTexture(texture);
                    add(cube);
                }*/

        /*Solid cube = new Solid(cubeModel, 0, 0, 0);
        cube.setTexture(texture);
        add(cube);*/

        Model mountainsModel = Model.loadFromFile("res/model/mountains.obj");
        add(new Solid(mountainsModel, 0, 0, 0, this));

        /*Model shipModel = Model.loadFromFile("res/model/ship.obj");
        add(new Solid(shipModel, 0, 0, 0, this));*/
        super.setup();
    }

    @Override
    public void tick() {
        camera.tick();
        super.tick();
    }

    public Camera getCamera() {
        return camera;
    }

    public Vector getLight() {
        return light;
    }
}
