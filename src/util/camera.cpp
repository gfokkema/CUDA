#include <cmath>
#include <iostream>
#include <util/camera.h>

template<typename T>
const T pi() {
    return std::acos(-T(1));
}

Camera::Camera(int _width, int _height, float _angle, Vector _pos, Vector _dir)
: _width(_width),
  _height(_height),
  _pos(_pos),
  _dir(_dir),
  _right { std::tan(_angle / 360 * 2 * pi<float>()), 0, 0 }
{
    std::cout << "Right: " << right().length() << std::endl;
    std::cout << "Up: " << up().length() << std::endl;
    std::cout << "Dir: " << dir().length() << std::endl;
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
    return _right % _dir;
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

const float Camera::ratio() const
{
    return _width / float(_height);
}

const int Camera::size() const
{
    return _height * _width;
}

const camera_t Camera::gpu_type() const
{
    camera_t camera = { width(), height(), ratio(), pos().gpu_type(),
                        dir().gpu_type(), up().gpu_type(), right().gpu_type() };
    return camera;
}
