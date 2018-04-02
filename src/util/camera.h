#include <cuda/gpu_kernels.h>

#include "vector.h"

#ifndef CAMERA_H_
#define CAMERA_H_

class Camera
{
public:
    Camera(int width, int height, float angle = 45,
           Vector pos = Vector(0, 0, 0), Vector dir = Vector(0, 0, -1));
    ~Camera();

    void strafe(float velocity, float dt);
    void move(float velocity, float dt);
    void lookAt(float x, float y);

    const Vector pos() const;
    const Vector dir() const;
    const Vector up() const;
    const Vector right() const;
    const int width() const;
    const int height() const;
    const int size() const;
    const camera_t gpu_type() const;

private:
    int _width, _height;
    Vector _pos;
    Vector _dir;
    Vector _right;
};

#endif /* CAMERA_H_ */
