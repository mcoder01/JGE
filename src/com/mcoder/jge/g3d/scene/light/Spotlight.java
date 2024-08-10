package com.mcoder.jge.g3d.scene.light;

import com.mcoder.jge.g3d.World;
import com.mcoder.jge.math.Vector3D;
import com.mcoder.jge.util.Point3D;

public class Spotlight extends Light {
    public Spotlight(Vector3D color, double x, double y, double z, World world) {
        super(color, x, y, z, world);
    }

    @Override
    public void tick() {
        Point3D p3d = new Point3D(Vector3D.sub(pos, world.getCamera().getPos()));
        p3d.rotate(Vector3D.scale(world.getCamera().getRot(), -1));
        viewPos = p3d.get();
    }
}
