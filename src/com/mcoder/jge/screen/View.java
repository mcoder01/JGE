package com.mcoder.jge.screen;

import java.awt.*;

public interface View {
	Screen screen = Screen.getInstance();

	void tick();
	void show(Graphics2D g2d);
}
