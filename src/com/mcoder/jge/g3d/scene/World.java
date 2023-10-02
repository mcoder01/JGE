package com.mcoder.jge.g3d.scene;

import com.mcoder.jge.g3d.render.Light;
import com.mcoder.jge.g3d.render.Model;
import com.mcoder.jge.g3d.render.Solid;
import com.mcoder.jge.math.Vector;
import com.mcoder.jge.screen.View;
import com.mcoder.jge.util.Texture;

public class World extends View {
    private final Camera camera;
    private final Light light;

    public World() {
        super();
        light = new Light(new Vector(0, 10, 0), 8, 1, 50);
        camera = new Camera(0, 10, 0);
    }

    @Override
    public void setup() {
        add(camera);

        // Testing
        Model cubeModel = Model.loadFromFile("res/model/cube.obj");
        Texture texture = new Texture("cobblestone.png");
        Solid[][][] platform = new Solid[3][1][3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) {
                platform[i][0][j] = new Solid(cubeModel, i-1, 0, j-1);
                platform[i][0][j].setTexture(texture);
                //add(platform[i][0][j]);
            }

        Solid cube = new Solid(cubeModel, 0, 0, 0);
        cube.setTexture(texture);
        //add(cube);

        Model mountainsModel = Model.loadFromFile("res/model/mountains.obj");
        Solid mountains = new Solid(mountainsModel, 0, 0, 0);
        add(mountains);

        /*Model shipModel = Model.loadFromFile("res/model/ship.obj");
        Solid ship = new Solid(shipModel, 0, 0, 0);
        //add(ship);*/
        super.setup();
    }

    public Camera getCamera() {
        return camera;
    }

    public Light getLight() {
        return light;
    }
}
