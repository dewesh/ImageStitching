#pragma once
#include"point.h"
#include <utility>
#include <vector>

using namespace std;
class staple
{
public:
	point p1_img1;
	point p2_img1;
	point p1_img2;
	point p2_img2;
	vector<pair<point,point>> match;
	int NumOfMatch;
	staple(void);
	staple(point,point, point, point);
	~staple(void);
};

