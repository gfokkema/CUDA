#include "vector.h"

#ifndef RAY_CPP_
#define RAY_CPP_

class Ray {
public:
    Ray(Vector pos, Vector dir) : _pos(pos), _dir(dir) {};
    ~Ray() {};

    const Vector pos() const      { return _pos; };
    const Vector dir() const      { return _dir; };
    const ray_t  gpu_type() const { return { _pos.gpu_type(),
                                             _dir.gpu_type() }; };
private:
    Vector _pos;
    Vector _dir;
};

#endif /* RAY_CPP_ */
