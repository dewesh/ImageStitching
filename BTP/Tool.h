#pragma once
#include <cv.h>
#include"point.h"
#include"edge.h"
#include"staple.h"

using namespace cv;


class Tool
{
private:
	static Tool *tool;
	Tool();
public:
	static Tool* getInstance();
	int getChainCode(int , int);
	Mat thinEdges_std(Mat);
	Mat removePoints(Mat ,vector<point>,int );
	vector<point> getJunctionPoints(Mat ,int );

	Mat CannyThreshold(Mat ,int ,int ,int ,int );
	Mat joinEdges(Mat ,int );
	bool checkValidJoin(int ,int ,Mat,int);
	int getTotalNeighbours(int , int , Mat, int);
	vector<point> GetLocalMaxima(vector<edge>,float,int);
	vector<edge> calculateCurvature(int ,vector<edge> );
	vector<point> removeLeft(vector<point> points,int p);
	vector<point> removeRight(vector<point> points,int p);
	Mat PlotImage(char* imgPath, char* outputPath, vector<point> points,int thickness);
	Mat removeCurve(edge tempEdge, Mat temp);
	template <class T, class U>
	int approxComp (T, T, U );
	edge getCircularCurve(Mat, int, int, int);
	~Tool();
};

