package com.mcoder.jgel.g3d.geom.plane;

import com.mcoder.jgel.g3d.geom.solid.Point3D;
import com.mcoder.jgel.math.Vector;

public class Surface {
    private final Point3D[] points;
    private final Vector[] texPoints;

    public Surface(Point3D[] points, Vector[] texPoints) {
        this.points = points;
        this.texPoints = texPoints;
    }

    public Point3D[] obtainPoints() {
        Point3D[] newPoints = new Point3D[points.length];
        for (int i = 0; i < points.length; i++)
            newPoints[i] = points[i].copy();
        return newPoints;
    }

    public Vector[] obtainTexPoints(int texW, int texH) {
        if (texPoints != null) {
            Vector[] newTexPoints = new Vector[texPoints.length];
            for (int i = 0; i < texPoints.length; i++) {
                newTexPoints[i] = texPoints[i].copy();
                newTexPoints[i].setX(newTexPoints[i].getX() * texW);
                newTexPoints[i].setY(newTexPoints[i].getY() * texH);
            }

            return newTexPoints;
        }

        return null;
    }
}
