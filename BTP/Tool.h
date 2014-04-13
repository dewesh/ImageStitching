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
	Mat PlotImage(char* imgPath, char* outputPath, vector<point> points,int thickness,int padding, int PixVal);
	Mat GetEdgeScheleton(char* imgPath,int lowThreshold,int ratio,int kernel_size,int BWThreshold);
	Mat removeCurve(edge tempEdge, Mat temp);
	template <class T, class U>
	int approxComp (T, T, U );
	edge getCircularCurve(Mat, int, int, int);
	Mat blendImages1(Mat left, Mat right, double alpha, staple st);
	Mat blendImages2(Mat left, Mat right, double alpha, staple st);
	Mat blendImages3(Mat left, Mat right, staple st);
	Mat blendImages4(Mat left, Mat right, staple st);
	Mat blendImages5(Mat left, Mat right, staple st);
	staple getFeatureMatch(staple tempStaple,vector<point> points1,vector<point> points2,double slope1,double slope2 ,double dist_tollerence,double slope_tollerence);
	double StructureDistance(point p1,point p2);
	double StructureSlope(point p1,point p2);
	vector<staple> getStaples(vector<point> points1, vector<point> points2,double dist_tollerence,double slope_tollerence);
	void rotateImage(const Mat &input, Mat &output, double alpha, double beta, double gamma, double dx, double dy, double dz, double f);
	void rotate(cv::Mat& src, double angle, double scaleFactor, cv::Mat& dst);
	vector<edge> removeSmallCurves(int th, Mat img,vector<edge> edges,vector<point> junction_pts,int curveLenThreshold,int BWThreshold);
	vector<edge> GetCurvature(Mat temp,int k,int curveLenThreshold,int BWThreshold);
	~Tool();
};

