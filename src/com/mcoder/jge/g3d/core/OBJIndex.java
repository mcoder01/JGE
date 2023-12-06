package com.mcoder.jge.g3d.core;

public class OBJIndex {
    private int pointIndex, texCoordsIndex;

    public OBJIndex(int pointIndex, int texCoordsIndex) {
        this.pointIndex = pointIndex;
        this.texCoordsIndex = texCoordsIndex;
    }

    public OBJIndex() {
        this(-1, -1);
    }

    public int getPointIndex() {
        return pointIndex;
    }

    public void setPointIndex(int pointIndex) {
        this.pointIndex = pointIndex;
    }

    public int getTexCoordsIndex() {
        return texCoordsIndex;
    }

    public void setTexCoordsIndex(int texCoordsIndex) {
        this.texCoordsIndex = texCoordsIndex;
    }
}
