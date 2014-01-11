#pragma once
#include"point.h"

class staple
{
public:
	point p1_img1;
	point p2_img1;
	point p1_img2;
	point p2_img2;
	int NumOfMatch;
	staple(void);
	staple(point,point, point, point);
	~staple(void);
};

