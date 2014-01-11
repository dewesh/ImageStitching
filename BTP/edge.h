#pragma once
#include <vector> 
#include"point.h"

class edge
{
public:
	int x1;
	int x2,y1,y2;
	std::vector<point> points;
	int Type; //0 -> circualr, 1-> normal

	edge(void);
	~edge(void);
};

