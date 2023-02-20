package com.mcoder.jgel.g3d.scene;

import com.mcoder.jgel.g3d.render.Model;
import com.mcoder.jgel.g3d.render.Solid;
import com.mcoder.jgel.scene.Display;
import com.mcoder.jgel.scene.Screen;
import com.mcoder.jgel.math.Vector;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;
import java.util.LinkedList;

public class World extends Display {
    private static World instance;

    private final Camera camera;
    private final Vector light;
    private final LinkedList<Solid> stuffs;

    private World() {
        super();
        light = new Vector(0, 0, 1);
        camera = new Camera(0, 0, -10);
        addListener(camera);
        stuffs = new LinkedList<>();

        // Testing
        BufferedImage texture = null;
        try {
            InputStream imageStream = getClass().getClassLoader().getResourceAsStream("cobblestone.png");
            if (imageStream != null) {
                texture = ImageIO.read(imageStream);
                imageStream.close();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        Model cubeModel = Model.loadFromFile("res/model/cube.obj");
        for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++)
                for (int k = -1; k <= 1; k++) {
                    Solid cube = new Solid(cubeModel, i, 0, k);
                    cube.setTexture(texture);
                    stuffs.add(cube);
                }
    }

    @Override
    public void update() {
        camera.update();
        for (Solid stuff : stuffs)
            stuff.update();
    }

    @Override
    public void show(Graphics2D g2d) {
        for (int i = 0; i < Screen.getInstance().getWidth()*Screen.getInstance().getHeight(); i++)
            Screen.getInstance().setPixel(i, 0);

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
