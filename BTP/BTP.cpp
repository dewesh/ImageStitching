// BTP.cpp : Defines the entry point for the console application.
#include "stdafx.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <algorithm> //sort
#include <vector> 
#include <string.h>
#include <math.h>
#include"Tool.h"

using namespace cv;
using namespace std;

int lowThreshold=25;
int const max_lowThreshold = 100;
int ratio = 3;

int kernel_size = 3;
int BWThreshold = 50;

string _path = "E:/personal/acads/BTP/images/Set1/scaled2/";
string leftImageName="set1.jpg",rightImageName="set2.png";

char* imgPath1 =  "E:/personal/acads/BTP/images/Set1/scaled2/set1.jpg";
char* outputPath1 = "E:/personal/acads/BTP/images/Set1/scaled2/set1_out2.png";
char* outputPath1_canny = "E:/personal/acads/BTP/images/Set1/scaled2/set12_l1_canny.png";
char* outputPath1_thin = "E:/personal/acads/BTP/images/Set1/scaled2/set12_l1_thin.png";
char* outputPath1_short = "E:/personal/acads/BTP/images/Set1/scaled2/set12_l1_short.png";
char* outputPath1_fps = "E:/personal/acads/BTP/images/Set1/scaled2/set12_l1.png";
char* outputPath1_fps1 = "E:/personal/acads/BTP/images/Set1/scaled2/set11.jpg";
char* outputPath1_fps2 = "E:/personal/acads/BTP/images/Set1/scaled2/set22.jpg";

char* outputPath2_canny = "E:/personal/acads/BTP/images/Set1/scaled2/set12_l2_canny.png";
char* outputPath2_thin = "E:/personal/acads/BTP/images/Set1/scaled2/set12_l2_thin.png";
char* outputPath2_short = "E:/personal/acads/BTP/images/Set1/scaled2/set12_l2_short.png";
char* outputPath2_fps = "E:/personal/acads/BTP/images/Set1/scaled2/set12_l2.png";

char* imgPath2 =  "E:/personal/acads/BTP/images/Set1/scaled2/set2.jpg";
char* outputPath2 = "E:/personal/acads/BTP/images/Set1/scaled2/set2_out2.png";

char* finalOutput1 = "E:/personal/acads/BTP/images/Set1/scaled2/Result1_avg.png";
char* finalOutput2 = "E:/personal/acads/BTP/images/Set1/scaled2/Result2_small_avg.png";
char* finalOutput3 = "E:/personal/acads/BTP/images/Set1/scaled2/Result3_dynamic_alpha_const_variation.png";
char* finalOutput4 = "E:/personal/acads/BTP/images/Set1/scaled2/Result4_dynamic_alpha_gaussian_variation.png";
char* finalOutput5 = "E:/personal/acads/BTP/images/Set1/scaled2/Result4_dynamic_alpha_exp_variation.png";
char* finalOutput6 = "E:/personal/acads/BTP/images/Set1/scaled2/Result4_dynamic_alpha_exp_variation.png";

//****************************************************************************************************
Mat src_col1,src_col2,out1,out2;
int scale1,scale2;
//****************************************************************************************************
bool comp(point p1, point p2){
	return (p1.curvature > p2.curvature);
}
//****************************************************************************************************
bool compStaples(staple p1, staple p2){
	return (p1.NumOfMatch > p2.NumOfMatch);
}
//****************************************************************************************************
vector<point> StructureSort(vector<point> points){
	sort(points.begin(),points.end(),comp);
	return points;
}
//***************************************************************************************************
/** @function main */
//#define _CALIBRATE
#define _STITCH
//#define _MINIMIZE1
//#define _MINIMIZE2
//#define _UNDISTORT
// the parameters to tune are in main and 3 in getStaple and the 
int main()
{
	int lol;
	vector<edge> edges1,edges2;
	vector<point> points1,points2;
	vector<staple> staples;
	vector<point> junction_pts1;
	//***************************************************** set variable ******************************************
	double overlap = 0.6; 
	double distanceTollerence= 0.02,slopeTollerence = 0.02;
	int k=10; //curvature estimate
	int curveLenThreshold = 21; // 2*k + 1
	//***************************************************** set variable ******************************************


#ifdef _MINIMIZE1	
	src_col1 = imread(imgPath1);
	src_col2 = imread(imgPath2);
	pyrDown( src_col1, src_col1, Size( src_col1.cols/2, src_col1.rows/2 ) );
	pyrDown( src_col2, src_col2, Size( src_col2.cols/2, src_col2.rows/2 ) );
	imwrite(imgPath1,src_col1);
	imwrite(imgPath2,src_col2);
#endif
	
#ifdef _UNDISTORT
	src_col1 = imread(imgPath1);
	src_col2 = imread(imgPath2);
	Mat camMat(3,3,CV_64F),distCoeffs(1,5,CV_64F);
	camMat = 0;
	distCoeffs = 0;
	camMat.at<double>(0,0) = 1.1883304229609116e+003;
	camMat.at<double>(0,2) = 6.4750000000000000e+002;
	camMat.at<double>(1,1) = 1.1883304229609116e+003;
	camMat.at<double>(1,2) = 4.8350000000000000e+002;
	camMat.at<double>(2,2) = 1;
	distCoeffs.at<double>(0,0) = 2.8674786343994491e-001;
	distCoeffs.at<double>(0,1) = -9.5074245338710095e-001;
	distCoeffs.at<double>(0,4) = 8.0169128676582824e-001;
	undistort(src_col1,out1,camMat,distCoeffs,noArray());
	imwrite(imgPath1,out1);
	undistort(src_col2,out2,camMat,distCoeffs,noArray());
	imwrite(imgPath2,out2);
#endif

#ifdef _MINIMIZE2
	src_col1 = imread(imgPath1);
	src_col2 = imread(imgPath2);
	pyrDown( src_col1, src_col1, Size( src_col1.cols/2, src_col1.rows/2 ) );
	pyrDown( src_col2, src_col2, Size( src_col2.cols/2, src_col2.rows/2 ) );
	imwrite(imgPath1,src_col1);
	imwrite(imgPath2,src_col2);
#endif
	int lol1;

	src_col1 = imread(imgPath1);
	src_col2 = imread(imgPath2);
	//scanf("%d",&lol1);
#ifdef _CALIBRATE
	 
	 cameraMat = getDefaultNewCameraMatrix(camMat,src_col1.size(),true);
	 
	distMat = 0;
	distMat.at<double>(0,0) = -0.00000004;
	distMat.at<double>(0,1) = 0;

	undistort(src_col1,out1,cameraMat,distMat,cameraMat);
	undistort(src_col2,out2,cameraMat,distMat,cameraMat);
	//imwrite(imgPath1,out1);
	//imwrite(imgPath2,out2);
	
	imshow("left image",out1);
	createTrackbar("scale1", "left image", &scale1, 10, CallbackForTrackBar);
	createTrackbar("scale2", "left image", &scale2, 10, CallbackForTrackBar);
	imshow("right image",out2);
#endif
#ifdef _STITCH

	int pivot_left=(src_col1.cols+6)*(1-overlap);
	int pivot_right=(src_col2.cols+6)*overlap;
	
	
	//left image 
	Mat tempImage;
	//***********************************************************************************************************************************************
	tempImage = Tool::getInstance()->GetEdgeScheleton(imgPath1,outputPath1_canny,outputPath1_thin,lowThreshold,ratio,kernel_size,BWThreshold);
	//junction_pts1 = Tool::getInstance()->getJunctionPoints(tempImage,BWThreshold); // set 1  of new feature points
	edges1 = Tool::getInstance()->GetCurvature(tempImage,k,curveLenThreshold,BWThreshold,outputPath1_short);	// k =30 :: the last edge returned contains the junction points
	//junction points calculation
	junction_pts1 = (edges1.back()).points;	// last one because the points are pushed in the last one so need to pop them out before using
	edges1.pop_back();
	//high curvature points calculation
	points1 = Tool::getInstance()->GetLocalMaxima(edges1,5,k);	// calculate local maxima of the edge list
	printf("removing left part of left image\n");
	
	Tool::getInstance()->PlotImage(imgPath1, outputPath1_fps, points1,5,3,0);

	points1 =  Tool::getInstance()->removeLeft(points1,pivot_left);
	printf("num of feature pts: %d\n",points1.size());
	//junction points calculations
	junction_pts1 =  Tool::getInstance()->removeLeft(junction_pts1,pivot_left);
	points1.insert(points1.end(),junction_pts1.begin(),junction_pts1.end());
	//plot the images using the feature points detected
	points1 = StructureSort(points1);
	Tool::getInstance()->PlotImage(imgPath1, outputPath1, points1,5,3,0);
	Tool::getInstance()->PlotImage(outputPath1, outputPath1, junction_pts1,3,0,100);

	printf("Left image feature detection complete\n---------------------------------------------\nstarting with right image\n");
	//***********************************************************************************************************************************************
	// right image
	tempImage = Tool::getInstance()->GetEdgeScheleton(imgPath2,outputPath2_canny,outputPath2_thin,lowThreshold,ratio,kernel_size,BWThreshold); 
	vector<point> junction_pts2;
	
	edges2 = Tool::getInstance()->GetCurvature(tempImage,k,curveLenThreshold,BWThreshold,outputPath2_short);	// k =30 :: the last edge returned contains the junction points
	junction_pts2 = (edges2.back()).points;
	edges2.pop_back();

	points2 = Tool::getInstance()->GetLocalMaxima(edges2,5,k);
	printf("removing right part of right image\n");
	
	Tool::getInstance()->PlotImage(imgPath2, outputPath2_fps, points2,5,3,0);

	points2 =  Tool::getInstance()->removeRight(points2,pivot_right);
	printf("num of feature pts: %d\n",points2.size());
	junction_pts2 =  Tool::getInstance()->removeRight(junction_pts2,pivot_right);
	points2.insert(points2.end(),junction_pts2.begin(),junction_pts2.end());

	points2 = StructureSort(points2);
	Tool::getInstance()->PlotImage(imgPath2, outputPath2, points2,5,3,0);
	Tool::getInstance()->PlotImage(outputPath2, outputPath2, junction_pts2,3,0,100);

	printf("size of 1st pointset :%d\nsize of 2nd point set :%d\n ",points1.size()+junction_pts1.size(),points2.size()+junction_pts1.size());
	printf("Staple collection Started\n");
	staples = Tool::getInstance()->getStaples(points1,points2,distanceTollerence,slopeTollerence); //***************** some studapa here
	printf("Done with collecting staples\n");
	//************************************************ print sample considered agfter maximum match************************************************
	staple tempStaple;
	int max = 0,maxIndex=0;
	for(int t=0 ; t < staples.size() ; t++){ //********** get the maximum match
		tempStaple = staples[t];
		if(tempStaple.NumOfMatch > max){
			maxIndex = t;
			max = tempStaple.NumOfMatch;
		}
		cout << tempStaple.NumOfMatch << endl;
		//printf("Staple info::\n-----------------\nimage-1 : x1=%d, y1=%d, X2= %d , Y2= %d :: Distance = %f\n",tempStaple.p1_img1.x,tempStaple.p1_img1.y,tempStaple.p2_img1.x,tempStaple.p2_img1.y,StructureDistance(tempStaple.p1_img1,tempStaple.p2_img1));
		//printf("image-2 : x1=%d, y1=%d, X2= %d , Y2= %d :: Distance = %f\n",tempStaple.p1_img2.x,tempStaple.p1_img2.y,tempStaple.p2_img2.x,tempStaple.p2_img2.y,StructureDistance(tempStaple.p1_img2,tempStaple.p2_img2));				
	}

	int t= maxIndex;
	printf("done with %d number of match in staple number:: %d\n",staples[t].NumOfMatch,t);
	//*****************************************************display the images generated*****************************************
	

	//printf("Staple info::\nimage-1 : \n----------\nx1=%d, y1=%d, X2= %d , Y2= %d :: Distance = %f\n",staples[t].p1_img1.x,staples[t].p1_img1.y,staples[t].p2_img1.x,staples[t].p2_img1.y,StructureDistance(staples[t].p1_img1,staples[t].p2_img1));
	//printf("Staple info::\nimage-2 : \n----------\nx1=%d, y1=%d, X2= %d , Y2= %d :: Distance = %f\n",staples[t].p1_img2.x,staples[t].p1_img2.y,staples[t].p2_img2.x,staples[t].p2_img2.y,StructureDistance(staples[t].p1_img2,staples[t].p2_img2));
	
	Mat src_l = imread(outputPath1);
	Mat src_r = imread(outputPath2);

	printf("\nslope:%lf,dist: %lf\n",Tool::getInstance()->StructureSlope(staples[t].p1_img1,staples[t].p2_img1),Tool::getInstance()->StructureDistance(staples[t].p1_img1,staples[t].p2_img1));
	printf("slope:%lf,dist: %lf\n",Tool::getInstance()->StructureSlope(staples[t].p1_img2,staples[t].p2_img2),Tool::getInstance()->StructureDistance(staples[t].p1_img2,staples[t].p2_img2));
	
	for(vector<staple>::iterator ia = staples.begin();ia != staples.end() ; ++ia){
		if(ia->NumOfMatch >= staples[t].NumOfMatch/2){
			cv::line(src_l, Point(ia->p1_img1.y,ia->p1_img1.x), Point(ia->p2_img1.y,ia->p2_img1.x), Scalar( 0, 0, 255 ), 1, 8,0);
			cv::line(src_r, Point(ia->p1_img2.y,ia->p1_img2.x), Point(ia->p2_img2.y,ia->p2_img2.x), Scalar( 0, 0, 255 ), 1, 8,0);
		}
	}
	
	cv::line(src_l, Point(staples[t].p1_img1.y,staples[t].p1_img1.x), Point(staples[t].p2_img1.y,staples[t].p2_img1.x), Scalar( 255, 0, 0 ), 3, 8,0);
	cv::line(src_r, Point(staples[t].p1_img2.y,staples[t].p1_img2.x), Point(staples[t].p2_img2.y,staples[t].p2_img2.x), Scalar( 255, 0, 0 ), 3, 8,0);
	
	/*for(vector<pair<point,point>>::iterator ia = staples[t].match.begin();ia != staples[t].match.end() ; ++ia){
		cv::line(src_l, Point(ia->first.y,ia->first.x),Point(ia->first.y,ia->first.x), Scalar( 255, 0, 0), 3, 8,0);
		cv::line(src_r, Point(ia->second.y,ia->second.x),Point(ia->second.y,ia->second.x), Scalar( 255, 0, 0), 3, 8,0);
	}*/

	cout << "p1_image1.x " << staples[t].p1_img1.x ;
	cout << " p1_image1.y " << staples[t].p1_img1.y << endl;
	cout << "p1_image2.x " << staples[t].p1_img2.x ;
	cout << " p1_image2.y " << staples[t].p1_img2.y << endl;
	
	imwrite(outputPath2,src_r);
	imwrite(outputPath1,src_l);
	src_l = imread(outputPath1_fps1);
	src_r = imread(outputPath1_fps2);
	staples[t].p1_img1.x *= 2;
	staples[t].p2_img1.x *= 2;
	staples[t].p1_img2.x *= 2;
	staples[t].p2_img2.x *= 2;
	staples[t].p1_img1.y *= 2;
	staples[t].p2_img1.y *= 2;
	staples[t].p1_img2.y *= 2;
	staples[t].p2_img2.y *= 2;
	double deltaThetaF,scalingF,theta1,theta2;
	theta1 = Tool::getInstance()->StructureSlope(staples[t].p1_img1,staples[t].p2_img1);
	theta2 = Tool::getInstance()->StructureSlope(staples[t].p1_img2,staples[t].p2_img2);
	scalingF = Tool::getInstance()->StructureDistance(staples[t].p1_img1,staples[t].p2_img1) / Tool::getInstance()->StructureDistance(staples[t].p1_img2,staples[t].p2_img2);
	deltaThetaF = (theta1-theta2)*180/3.1415;
	
	Mat M_l(2,3,CV_64F),M_r_scale,M_r,M_r_mov(2,3,CV_64F); // the transformation matrix
	Mat final,final1,final2,final3,final4,final5; // final image

	M_l = 0;
	M_r_mov = 0;
	M_l.at<double>(0,0) = 1;
	M_l.at<double>(1,1) = 1;
	M_l.at<double>(1,2) = (src_l.rows + src_r.rows)/4;
	M_l.at<double>(0,2) = 0;

	M_r_mov.at<double>(0,0) = 1;
	M_r_mov.at<double>(1,1) = 1;
	M_r_mov.at<double>(0,2) = (staples[t].p1_img1.y - staples[t].p1_img2.y) ;
	M_r_mov.at<double>(1,2) = (((src_l.rows + src_r.rows)/4 )+ (staples[t].p1_img1.x - staples[t].p1_img2.x));
	
	//M_r = getRotationMatrix2D((Point2f)(src_l.cols/2.0,src_l.rows/2.0),20,1);//3*(staples[t].p1_img1.x - staples[t].p1_img2.x),((src_l.rows + src_r.rows)/4 )+ staples[t].p1_img1.y
	//M_r = getRotationMatrix2D(Point(0, 0),deltaThetaF,1);//((src_l.rows + src_r.rows)/4 )+ (staples[t].p1_img1.x), staples[t].p1_img1.y
	M_r = getRotationMatrix2D(Point(staples[t].p1_img2.y, staples[t].p1_img2.x),deltaThetaF,1);
	M_r_scale = getRotationMatrix2D((Point2f)( staples[t].p1_img2.y, staples[t].p1_img2.x),0,scalingF);

	
	
Size s = src_r.size() + src_l.size();
	warpAffine(src_r,final,M_r,src_r.size(),1,0,1);		// ROTATE right image rotation
	warpAffine(final,src_r,M_r_scale,final.size(),1,0,1); // SCALE right image
	warpAffine(final,src_r,M_r_mov,s,1,0,1);	// TRANSLATE right image translation

	warpAffine(src_l,src_l,M_l,s,1,0,1);		//TRANSLATE left image
	
	//scanf("%d",&lol);	

	final5 = Tool::getInstance()->blendImages5(src_l,src_r,staples[t]);
	imwrite(finalOutput5,final5);
	/*
	final1 = Tool::getInstance()->blendImages1(src_l,src_r,0.5,staples[t]);
	imwrite(finalOutput1,final1);
		
	final2 = Tool::getInstance()->blendImages2(src_l,src_r,0.5,staples[t]);
	imwrite(finalOutput2,final2);

	final3 = Tool::getInstance()->blendImages3(src_l,src_r,staples[t]);
	imwrite(finalOutput3,final3);
final4 = Tool::getInstance()->blendImages4(src_l,src_r,staples[t]);
	imwrite(finalOutput4,final4);
		
	*/

#endif
	scanf("%d",&lol);
	waitKey(0);
	return 0;
}