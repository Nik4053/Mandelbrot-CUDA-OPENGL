#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
    // save diagnostic state
    #pragma warning( push, 3 )
    #define WINDOWS_LEAN_AND_MEAN
    #define NOMINMAX
    #define WIN32_LEAN_AND_MEAN
    #include <Windows.h>
#endif
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#ifdef _WIN32
    // turn the warnings back on
    #pragma warning( pop )
#endif

#include "interactions.h"

// texture and pixel objects
GLuint pbo = 0;     // OpenGL pixel buffer object
GLuint tex = 0;     // OpenGL texture object
struct cudaGraphicsResource *cuda_pbo_resource;

void render() {
    uchar4 *d_out = 0;
    cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void **) &d_out, NULL,
                                         cuda_pbo_resource);
    kernelLauncher(d_out, W, H, loc,scrollDepth,itermode);
    cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);

}

void drawTexture() {
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, NULL);
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(0, 0);
    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(0, H);
    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(W, H);
    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(W, 0);
    glEnd();
    glDisable(GL_TEXTURE_2D);
}

void display() {
    render();
    drawTexture();
    glutSwapBuffers();
}

void initGLUT(int *argc, char **argv) {
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    // glutInitDisplayString("hidpi core rgba double")
    glutInitWindowSize(W, H);
    glutCreateWindow(TITLE_STRING);
#ifndef __APPLE__
    glewInit();
#endif
}

void initPixelBuffer() {
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, 4 * W * H * sizeof(GLubyte), 0,
                 GL_STREAM_DRAW);
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo,
                                 cudaGraphicsMapFlagsWriteDiscard);
}

void exitfunc() {
    if (pbo) {
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &tex);
    }
}

int main(int argc, char **argv) {
    printInstructions();
    init(W,H);
    initGLUT(&argc, argv);
    gluOrtho2D(0, W, H, 0);
    glutKeyboardFunc(keyboard);
    glutSpecialFunc(handleSpecialKeypress);
    glutPassiveMotionFunc(mouseMove);
    glutMotionFunc(mouseDrag);
    glutMouseWheelFunc(mouseWheel);
    glutMouseFunc(mouse);
    glutDisplayFunc(display);
    initPixelBuffer();
    //glutFullScreen();
    glutMainLoop();
    atexit(exitfunc);
    destroy();
    return 0;
}