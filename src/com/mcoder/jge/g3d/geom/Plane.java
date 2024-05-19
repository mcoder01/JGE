package com.mcoder.jge.g3d.geom;

import com.mcoder.jge.math.Vector3D;

public record Plane(Vector3D pos, Vector3D normal) {
    public double distanceToPoint(Vector3D point) {
        return normal.times(point).sub(normal.times(pos)).get(0);
    }
}
