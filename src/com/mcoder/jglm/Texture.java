package com.mcoder.jglm;

import java.awt.image.BufferedImage;

public class Texture {
    private final BufferedImage[] images;
    private int index;

    /**
     * Creates a texture basing on one or multiple images, for animated sprites
     * @param images Array of images which constitutes the texture
     */
    public Texture(BufferedImage... images) {
        this.images = images;
    }

    /**
     * Creates a texture basing on a unique image which contains all the components of the texture
     * @param global Global image containing all the textures
     * @param subW Width of each single image
     * @param subH Height of each single image
     */
    public Texture(BufferedImage global, int subW, int subH) {
        int rows = global.getHeight()/subH;
        int cols = global.getWidth()/subW;
        images = new BufferedImage[rows*cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                images[i*cols+j] = global.getSubimage(j*subW, i*subH, subW, subH);
    }

    public BufferedImage getImage() {
        if (index == images.length)
            index = 0;
        return images[index++];
    }
}
