package com.mcoder.jge.g3d.render.slope;

import com.mcoder.jge.g3d.core.Vertex;
import com.mcoder.jge.math.Vector2D;
import com.mcoder.jge.math.Vector3D;

public class SlopeVertex implements Slope {
    private final SlopeVector[] slopes;

    public SlopeVertex(Vertex start, Vertex end, int steps) {
        slopes = new SlopeVector[] {
                new SlopeVector(start.getPosition(), end.getPosition(), steps),
                new SlopeVector(start.getTexCoords(), end.getTexCoords(), steps),
                new SlopeVector(start.getNormal(), end.getNormal(), steps),
                new SlopeVector(start.getScreenPosition(), end.getScreenPosition(), steps)
        };
    }

    @Override
    public void advance(int steps) {
        for (SlopeVector s : slopes)
            s.advance(steps);
    }

    @Override
    public void advance() {
        for (SlopeVector s : slopes)
            s.advance();
    }

    @Override
    public Vertex getValue() {
        return new Vertex(
                (Vector3D) slopes[0].getValue(),
                (Vector2D) slopes[1].getValue(),
                (Vector3D) slopes[2].getValue(),
                (Vector2D) slopes[3].getValue()
        );
    }
}
