#ifndef RAY_CPP_
#define RAY_CPP_

#include "vector.h"

class Ray {
public:
    Ray(Vector pos, Vector dir) : _pos(pos), _dir(dir) {};
    ~Ray() {};

    const Vector pos() const { return _pos; };
    const Vector dir() const { return _dir; };
private:
    Vector _pos;
    Vector _dir;
};

#endif /* RAY_CPP_ */
