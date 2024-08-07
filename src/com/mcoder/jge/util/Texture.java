package com.mcoder.jge.util;

import com.mcoder.jge.math.Vector2D;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;

public class Texture {
    private BufferedImage[] images;
    private int[] pixels;
    private int index;

    public Texture(BufferedImage... images) {
        this.images = images;
    }

    /**
     * Creates a texture basing on one or multiple images, for animated sprites
     * @param imagePaths Array of paths to the images which constitute the texture
     */
    public Texture(String... imagePaths) {
        images = new BufferedImage[imagePaths.length];
        for (int i = 0; i < images.length; i++)
            images[i] = loadImage(imagePaths[i]);
    }

    /**
     * Creates a texture basing on a unique image which contains all the components of the texture
     * @param globalImagePath Path to the image containing all the textures
     * @param subW Width of each single image
     * @param subH Height of each single image
     */
    public Texture(String globalImagePath, int subW, int subH) {
        BufferedImage global = loadImage(globalImagePath);
        if (global != null) {
            int rows = global.getHeight() / subH;
            int cols = global.getWidth() / subW;
            images = new BufferedImage[rows * cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    images[i * cols + j] = global.getSubimage(j * subW, i * subH, subW, subH);
        }
    }

    private BufferedImage loadImage(String fileName) {
        try {
            InputStream is = getClass().getClassLoader().getResourceAsStream(fileName);
            if (is != null) return ImageIO.read(is);
        } catch (IOException e) {
            System.err.println("Unable to load " + fileName);
        }

        return null;
    }

    public BufferedImage nextImage() {
        if (images == null)
            return null;

        index++;
        if (index == images.length)
            index = 0;
        pixels = null;
        return images[index];
    }

    public int[] getPixels() {
        if (pixels == null) {
            BufferedImage image = images[index];
            return pixels = image.getRGB(0, 0, image.getWidth(),
                    image.getHeight(), null, 0, image.getWidth());
        }

        return pixels;
    }

    public int getRGB(int x, int y) {
        return getPixels()[x+y*getWidth()];
    }

    public BufferedImage getImage() {
        return images[index];
    }

    public int getWidth() {
        return getImage().getWidth();
    }

    public int getHeight() {
        return getImage().getHeight();
    }

    public Vector2D getSize() {
        return new Vector2D(getWidth(), getHeight());
    }
}
