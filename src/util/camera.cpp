#include "camera.h"

Camera::Camera(int width, int height, float angle, Vector pos, Vector dir)
: _width(width),
  _height(height),
  _pos(pos),
  _dir(dir),
  _right { tanf(angle / 360 * 2 * M_PI), 0, 0 }
{
}

Camera::~Camera()
{
}

void Camera::strafe(float velocity, float dt)
{
    _pos += velocity * dt * _right;
}

void Camera::move(float velocity, float dt)
{
    _pos += velocity * dt * _dir;
}

void Camera::lookAt(float x, float y)
{
    float invwidth = 1.f / _width;
    float invheight = 1.f / _height;
    x = (x + .5) * invwidth - 0.5;
    y = (y + .5) * invheight - 0.5;
    Vector up = this->up();
    Vector newDir = (x * _right + y * up + _dir).normalize();
    Vector newRight = _right - (_right * newDir) * newDir;
    _dir = newDir, _right = newRight;
}

const Vector Camera::pos() const
{
    return _pos;
}
const Vector Camera::dir() const
{
    return _dir;
}
const Vector Camera::up() const
{
    return _right % _dir * _height / float(_width);
}
const Vector Camera::right() const
{
    return _right;
}
const int Camera::width() const
{
    return _width;
}
const int Camera::height() const
{
    return _height;
}
const int Camera::size() const
{
    return _height * _width;
}
const camera_t Camera::gpu_type() const
{
    camera_t camera = { _width, _height, pos().gpu_type(), dir().gpu_type(),
                        up().gpu_type(), right().gpu_type() };
    return camera;
}
