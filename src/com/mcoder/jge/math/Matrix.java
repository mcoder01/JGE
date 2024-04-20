package com.mcoder.jge.math;

import java.util.function.Consumer;

public class Matrix {
    protected final int rows, cols;
    protected double[] data;
    private Matrix transposed;

    protected Matrix(int rows, int cols, double... data) {
        this.rows = rows;
        this.cols = cols;
        this.data = data;
    }

    public Matrix(int rows, int cols) {
        this(rows, cols, new double[rows*cols]);
    }

    protected void forEach(Consumer<Integer> consumer) {
        for (int i = 0; i < data.length; i++) consumer.accept(i);
    }

    public Matrix add(Matrix m) {
        if (rows != m.rows || cols != m.cols) {
            System.err.println("Sum between matrices of different sizes!");
            return this;
        }

        Matrix result = new Matrix(rows, cols);
        result.forEach(i -> data[i] += m.data[i]);
        return result;
    }

    public Matrix sub(Matrix m) {
        if (rows != m.rows || cols != m.cols) {
            System.err.println("Subtraction between matrices of different sizes!");
            return this;
        }

        return add(m.scale(-1));
    }

    public Matrix scale(double factor) {
        Matrix result = new Matrix(rows, cols);
        result.forEach(i -> data[i] *= factor);
        return result;
    }

    public Matrix times(Matrix m) {
        if (cols != m.rows) {
            System.err.println("Multiplication between matrices of incompatible sizes!");
            return this;
        }

        Matrix result = new Matrix(rows, m.cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < m.cols; j++)
                for (int k = 0; k < cols; k++)
                    result.data[i*m.cols+j] += data[i*cols+k]*m.data[k*m.cols+j];
        return result;
    }

    public Matrix transpose() {
        if (transposed == null) {
            transposed = new Matrix(cols, rows);
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    transposed.data[j*rows+i] = data[i*cols+j];
        }

        return transposed;
    }
}
