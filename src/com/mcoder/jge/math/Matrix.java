package com.mcoder.jge.math;

import com.mcoder.jge.g3d.geom.Axis;

import java.util.ArrayList;
import java.util.function.Consumer;

public class Matrix {
    protected final int rows, cols;
    protected double[] data;
    private Matrix transposed;

    public Matrix(int rows, int cols, double... data) {
        this.rows = rows;
        this.cols = cols;
        this.data = data;
    }

    public Matrix(int rows, int cols) {
        this(rows, cols, new double[rows*cols]);
    }

    public Matrix(int rows, int cols, ArrayList<Double> dataList) {
        this(rows, cols);
        forEach(i -> data[i] = dataList.get(i));
    }

    protected void forEach(Consumer<Integer> consumer) {
        for (int i = 0; i < data.length; i++) consumer.accept(i);
    }

    private Matrix elementWiseOperation(Matrix m, Operation operation) {
        if (data.length < m.data.length)
            return m.elementWiseOperation(this, operation);

        if ((rows != m.rows && cols != m.cols && (m.rows > 1 || m.cols > 1))
                || (rows == m.rows && cols != m.cols && m.cols > 1)
                || (cols == m.cols && rows != m.rows && m.rows > 1))
            throw new RuntimeException("Invalid matrices sizes for this operation!");

        Matrix result = new Matrix(rows, cols);
        if (rows == m.rows && cols == m.cols)
            result.forEach(i -> result.data[i] = operation.compute(data[i], m.data[i]));
        else if (m.rows == 1 && m.cols == 1)
            result.forEach(i -> result.data[i] = operation.compute(data[i], m.data[0]));
        else result.forEach(i -> result.data[i] = operation.compute(data[i], m.data[i%m.data.length]));
        return result;
    }

    public Matrix add(Matrix m) {
        return elementWiseOperation(m, Double::sum);
    }

    public Matrix sub(Matrix m) {
        if (rows != m.rows && cols != m.cols)
            throw new RuntimeException("Subtraction between matrices of different sizes!");
        return add(m.scale(-1));
    }

    public Matrix scale(double factor) {
        return scale(new Matrix(1, 1, factor));
    }

    public Matrix scale(Matrix m) {
        return elementWiseOperation(m, (a, b) -> a*b);
    }

    public Matrix times(Matrix m) {
        if (cols != m.rows)
            throw new RuntimeException("Multiplication between matrices of incompatible sizes!");

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

    public int getRows() {
        return rows;
    }

    public int getCols() {
        return cols;
    }

    public double get(int index) {
        return data[index];
    }

    public static Matrix eye(int size) {
        Matrix eye = new Matrix(size, size);
        for (int i = 0; i < size; i++)
            eye.data[i*size+i] = 1;
        return eye;
    }

    public static Matrix euler(Matrix angles) {
        return euler(angles.data[0], Axis.X)
                .times(euler(angles.data[1], Axis.Y))
                .times(euler(angles.data[2], Axis.Z));
    }

    private static Matrix euler(double angle, Axis axis) {
        final double s = Math.sin(angle), c = Math.cos(angle);
        switch (axis) {
            case Axis.X -> {
                return new Matrix(3, 3,
                        1, 0, 0,
                        0, c, -s,
                        0, s, c);
            }
            case Axis.Y -> {
                return new Matrix(3, 3,
                        c, 0, s,
                        0, 1, 0,
                        -s, 0, c);
            }
            case Axis.Z -> {
                return new Matrix(3, 3,
                        c, -s, 0,
                        s, c, 0,
                        0, 0, 1);
            }
            default -> throw new RuntimeException("Input axis is not valid!");
        }
    }

    private interface Operation {
        double compute(double a, double b);
    }
}
