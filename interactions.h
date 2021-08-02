#ifndef INTERACTIONS_H
#define INTERACTIONS_H
#define W 1600
#define H 800
#define DELTA 5 // pixel increment for arrow keys
#define TITLE_STRING "Mandelbrot"
int2 loc = {0, 0};
int scrollDepth = 0;
bool itermode = 0;
void keyboard(unsigned char key, int x, int y) {
    if (key == 'a') itermode = !itermode; // toggle tracking mode
    if (key == 27) exit(0);
    glutPostRedisplay();
}

void mouseMove(int x, int y) {
    if (true) return;
}

void mouseDrag(int x, int y) {
    if (true) return;
}


/**
 * gets called once for each mouse button press.
 * @param button the button that was pressed
 * @param state If pressed or released: GLUT_DOWN or GLUT_UP
 * @param x,y The x,y position of the mouse at the time of this event
 */
void mouse(int button, int state, int x, int y) {
    // Save the left button state
    if (button == GLUT_LEFT_BUTTON) {
        // leftMouseButtonDown = (state == GLUT_DOWN);
    } else if (button == GLUT_RIGHT_BUTTON) {
        // right MouseButton
        // rightMouseButtonDown = (state == GLUT_DOWN);
    } else if (button == GLUT_MIDDLE_BUTTON) {
        // middle MouseButton
        // middleMouseButtonDown = (state == GLUT_DOWN);
    }
    // Save the mouse position
    // mousePos.x = x;
    // mousePos.y = y;
}

/**
 * Gets only called on mouse wheel movement
 * @param button the buttons on the mouse that are currently pressed. In bitmask format. 1<<0 = left mouse button, 1<<1 right mouse button, 1<<4 middle mouse button
 * @param dir the direction of the wheel roll. >0 is up / zoom in,
 * @param x,y The x,y position of the mouse at the time of this event
 */
void mouseWheel(int button, int dir, int x, int y) {
//    if(button&1<<0) printf("Left Mouse Button\n");
//    if(button&1<<1) printf("right Mouse Button\n");
//    if(button&1<<4) printf("Middle Mouse Button\n");
    scrollDepth+=dir;
    glutPostRedisplay();
}

void handleSpecialKeypress(int key, int x, int y) {
    if (key == GLUT_KEY_LEFT) loc.x -= DELTA;
    if (key == GLUT_KEY_RIGHT) loc.x += DELTA;
    if (key == GLUT_KEY_UP) loc.y -= DELTA;
    if (key == GLUT_KEY_DOWN) loc.y += DELTA;
    glutPostRedisplay();
}

void printInstructions() {
    printf("Mandelbrot interactions\n");
    printf("a: toggle logarithmic iteration coloring\n");
    printf("arrow keys: move ref location\n");
    printf("mouse scroll: zoom\n");
    printf("esc: close graphics window\n");
}

#endif