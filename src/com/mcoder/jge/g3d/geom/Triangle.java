package com.mcoder.jge.g3d.geom;

import com.mcoder.jge.g3d.core.Vertex;

public record Triangle(Vertex... vertices) {
    public boolean isVisible() {
        for (Vertex v : vertices)
            if (v.getPosition().transpose().times(v.getNormal()).get(0) < 0)
                return true;
        return false;
    }
}