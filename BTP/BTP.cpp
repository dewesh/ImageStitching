// BTP.cpp : Defines the entry point for the console application.
//

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

int k=10; //curvature estimate
int curveLenThreshold = 41; // 2*k + 1

char* window_name = "Edge Map";
char* imgPath1 =  "E:/personal/acads/BTP/images/Set1/scaled2/set1.jpg";
char* outputPath1 = "E:/personal/acads/BTP/images/Set1/scaled2/set1_out2.png";
char* finalOutput = "E:/personal/acads/BTP/images/Set1/scaled2/Result.png";
char* outputPath1_canny = "E:/personal/acads/BTP/images/Set1/scaled2/set12_l1_out2.png";
char* outputPath1_thin = "E:/personal/acads/BTP/images/Set1/scaled2/set12_l2_out2.png";
//char* outputPath1_max = "E:/personal/acads/BTP/images/Set1/scaled2/set12_l3_out2.png";
//char* outputPath1_short = "E:/personal/acads/BTP/images/Set1/scaled2/set12_l4_out2.png";

char* imgPath2 =  "E:/personal/acads/BTP/images/Set1/scaled2/set2.jpg";
char* outputPath2 = "E:/personal/acads/BTP/images/Set1/scaled2/set2_out2.png";

/*
char* imgPath1 =  "E:/personal/acads/BTP/images/Set1/set16.JPG";
char* outputPath1 = "E:/personal/acads/BTP/images/Set1/set16_out.png";

char* imgPath2 =  "E:/personal/acads/BTP/images/Set1/set17.JPG";
char* outputPath2 = "E:/personal/acads/BTP/images/Set1/set17_out.png";
*/
//****************************************************************************************************
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
//*****************************squared distance is returned***********************************************************************
double StructureDistance(point p1,point p2){
	double dist;
	double x= p1.x-p2.x;
	double y= p1.y-p2.y;
	dist = (x*x) + (y*y);
	dist = sqrt(dist);
	return dist;
}
//********************************* radian slope **************************************************************
double StructureSlope(point p1,point p2) // angle slope
{
	double slope;
	double x= p1.x-p2.x;
	double y= p1.y-p2.y;
	slope =atan2(y,x) + 3.15;
	return slope;
}
//****************************************************************************************************
void rotateImage(const Mat &input, Mat &output, double alpha, double beta, double gamma, double dx, double dy, double dz, double f)
  {
	  int len = std::max(input.cols, input.rows);
	  int len_c = input.cols;
	  int len_r = input.rows;
    alpha = (alpha - 90.)*CV_PI/180.;
    beta = (beta - 90.)*CV_PI/180.;
    gamma = (gamma - 90.)*CV_PI/180.;
    // get width and height for ease of use in matrices
    double w = (double)input.cols;
    double h = (double)input.rows;
    // Projection 2D -> 3D matrix
    Mat A1 = (Mat_<double>(4,3) <<
              1, 0, -w/2,
              0, 1, -h/2,
              0, 0,    0,
              0, 0,    1);
    // Rotation matrices around the X, Y, and Z axis
    Mat RX = (Mat_<double>(4, 4) <<
              1,          0,           0, 0,
              0, cos(alpha), -sin(alpha), 0,
              0, sin(alpha),  cos(alpha), 0,
              0,          0,           0, 1);
    Mat RY = (Mat_<double>(4, 4) <<
              cos(beta), 0, -sin(beta), 0,
              0, 1,          0, 0,
              sin(beta), 0,  cos(beta), 0,
              0, 0,          0, 1);
    Mat RZ = (Mat_<double>(4, 4) <<
              cos(gamma), -sin(gamma), 0, 0,
              sin(gamma),  cos(gamma), 0, 0,
              0,          0,           1, 0,
              0,          0,           0, 1);
    // Composed rotation matrix with (RX, RY, RZ)
    Mat R = RX * RY * RZ;
    // Translation matrix
    Mat T = (Mat_<double>(4, 4) <<
             1, 0, 0, dx,
             0, 1, 0, dy,
             0, 0, 1, dz,
             0, 0, 0, 1);
    // 3D -> 2D matrix
    Mat A2 = (Mat_<double>(3,4) <<
              f, 0, w/2, 0,
              0, f, h/2, 0,
              0, 0,   1, 0);
    // Final transformation matrix
    Mat trans = A2 * (T * (R * A1));
    // Apply matrix transformation
    warpPerspective(input, output, trans,cv::Size(len, len), INTER_LANCZOS4);
  }
//****************************************************************************************************
void rotate(cv::Mat& src, double angle, double scaleFactor, cv::Mat& dst)
{
    int len = std::max(src.cols, src.rows);
    cv::Point2f pt(len/2., len/2.);
	cv::Mat r = cv::getRotationMatrix2D(pt, angle, scaleFactor);

    cv::warpAffine(src, dst, r, cv::Size(len, len));
}
//****************************************************************************************************
vector<edge> removeSmallCurves(int th, Mat img,vector<edge> edges,vector<point> junction_pts){
	//printf("started removing small curves\n");
	Mat temp,temp1;

	edges.clear();
	edge tempEdge;
	point tempPoint;

	int totalEdges=0;
	int i,j,n;
	int i1,j1,k,l,i2,j2;
	temp.create(img.size(),img.type());
	temp = Scalar::all(0);

	temp1.create(img.size(),img.type());
	temp1 = Scalar::all(0);
	temp1 = img - temp1;

	for(i=2;i<img.rows-2;i++)
	{
		for(j=2;j<img.cols-2;j++)
		{
			if(img.at<uchar>(i,j) > BWThreshold && temp.at<uchar>(i,j) < BWThreshold )
			{
				n=Tool::getInstance()->getTotalNeighbours(i,j,img,BWThreshold);
				if(n==1)
				{
					//start traversing path
					tempEdge.x1=i;
					tempEdge.y1=j;

					tempPoint.x = i;
					tempPoint.y = j;
					tempPoint.chainCode = 0;

					tempEdge.points.clear();
					tempEdge.Type = 1;
					tempEdge.points.push_back(tempPoint);

					temp.at<uchar>(i,j)=255;
					i1=i;
					j1=j;
					// imwrite("E:/personal/acads/BTP/images/Set1/smallEdgesRemoved_Output.png",temp);
					do{
						j2=j1;
						i2=i1;
						for(k=-1;k<=1;k++)
						{
							for(l=-1;l<=1 ;l++)
							{
								if(k!=0 || l!=0)
								{
									i1=i2+k;
									j1=j2+l;
									if(i1>2 && j1>2 && i1 < (img.rows-3) && j1 < (img.cols-3))
									{
										if(img.at<uchar>(i1,j1) > BWThreshold && temp.at<uchar>(i1,j1) < BWThreshold)
										{
											temp.at<uchar>(i1,j1)=255;
											tempPoint.chainCode=Tool::getInstance()->getChainCode(k,l);
											tempPoint.x=i1;
											tempPoint.y=j1;
											tempPoint.curvature=0;
											goto endofloop;
										}
									}
									else{
										goto endofloop1;
									}
								}
							}
						}
endofloop:
						tempEdge.points.push_back(tempPoint);
endofloop1:;
						//printf("%d::%d-->in->",i1,j1);
					}while(Tool::getInstance()->getTotalNeighbours(i1,j1,img,BWThreshold)==2 && img.at<uchar>(i1,j1) > BWThreshold && i1>2 && j1>2 && i1 < (img.rows-3) && j1 < (img.cols-3));
					//  printf("out\n");
					tempEdge.x2=i1;
					tempEdge.y2=j1; // all the edges are stored in here in tempEdge as temp
					totalEdges++;

					if(tempEdge.points.size() > curveLenThreshold)
					{
						edges.push_back(tempEdge);
					}
					else
					{
						img = Tool::getInstance()->removeCurve(tempEdge,img);
					}
				}

				if(n==0)
				{
					img.at<uchar>(i,j)=0;
				}

			}
			//printf("flag2");
		}

	}
	//printf("lol");
	// loop for circular path detection can be optimised and merged with the main loop and can be determined in one go
	for(i=2;i<img.rows-2;i++) 
	{
		for(j=2;j<img.cols-2;j++)
		{
			if(img.at<uchar>(i,j) > BWThreshold && temp.at<uchar>(i,j) < BWThreshold)
			{
				n=Tool::getInstance()->getTotalNeighbours(i,j,img-temp,BWThreshold);
				if(n==2)
				{	
					tempEdge.points.clear();
					//todo: get circular path and store in tempedge
					tempEdge = Tool::getInstance()->getCircularCurve(img,BWThreshold,i,j);
					totalEdges++;
					if(tempEdge.points.size() > curveLenThreshold)
					{
						edges.push_back(tempEdge);
						temp = Tool::getInstance()->removePoints(temp,tempEdge.points,255);
					}
					else
					{
						img = Tool::getInstance()->removeCurve(tempEdge,img);

					}
				}
			}
		}
	}

	//imwrite("E:/personal/acads/BTP/images/Set1/scaled2/set18_smallEdgesRemoved_temp1.png",temp);
	imwrite("E:/personal/acads/BTP/images/Set1/scaled2/set2_smallEdgesRemoved_img.png",img);
	tempEdge.points.clear();
	for each(point tempPoint in junction_pts){
		n=Tool::getInstance()->getTotalNeighbours(tempPoint.x,tempPoint.y,img,BWThreshold);
		if(n==3){
			tempPoint.curvature = 0;
			tempEdge.points.push_back(tempPoint);
		}
	}
	edges.push_back(tempEdge);
	//temp1 = temp1 - img;
	//printf("total %d edges detected and %d are taken into consideration ::\n",totalEdges,edges.size());
	//imwrite("E:/personal/acads/BTP/images/Set1/scaled2/set18_smallEdgesRemoved.png",temp1);

	//imwrite(outputPath1,img);

	return edges;
}
//****************************************************************************************************
Mat GetEdgeScheleton(char* imgPath){
	Mat src = imread(imgPath,0);
	if( !src.data ){ 
		printf("error-noimage found\n");
	}

	Mat temp;
	temp.create(src.size(), src.type());

	temp=Tool::getInstance()->CannyThreshold(src,lowThreshold,ratio,kernel_size,BWThreshold);
	//imwrite(outputPath1_canny,temp);
	copyMakeBorder( temp, temp, 3, 3, 3, 3, BORDER_CONSTANT, 0);
	//copyMakeBorder( src_col, src_col, 3, 3, 3, 3, BORDER_CONSTANT, 0);

	temp = Tool::getInstance()->thinEdges_std(temp);
	//imwrite(outputPath1_thin,temp);
	//temp = Tool::getInstance()->joinEdges(temp,BWThreshold);
	//imwrite(outputPath1_thin,temp);
	return temp;
}
//****************************************************************************************************
vector<edge> GetCurvature(Mat temp,int k){
	vector<edge> edges;  
	vector<point> junction_pts;

	junction_pts = Tool::getInstance()->getJunctionPoints(temp,BWThreshold); //TODO : return the junction point obtained here
	temp = Tool::getInstance()->removePoints(temp,junction_pts,0);

	//edges = extractCurves(temp);
	temp.col(3).setTo(cv::Scalar(0));
	temp.col(4).setTo(cv::Scalar(0));
	temp.col(temp.cols-4).setTo(cv::Scalar(0));
	temp.col(temp.cols-5).setTo(cv::Scalar(0));
	temp.row(3).setTo(cv::Scalar(0));
	temp.row(4).setTo(cv::Scalar(0));
	temp.row(temp.rows-4).setTo(cv::Scalar(0));
	temp.row(temp.rows-5).setTo(cv::Scalar(0));

	edges = removeSmallCurves(curveLenThreshold,temp,edges,junction_pts);
	//imwrite(outputPath1_short,temp);
	edge junction = edges.back();
	edges.pop_back();

	edges = Tool::getInstance()->calculateCurvature(k,edges);
	edges.push_back(junction);
	//imwrite("E:/personal/acads/BTP/images/Set1/scaled/set18_smallEdgesRemoved.png",temp);

	//int x,y;
	//for(int i=0;i<edges.size();i++){
	//	for(int j=k;j<edges[i].points.size()-k;j++){
	//		x=edges[i].points[j].x;
	//		y=edges[i].points[j].y;
	//		src_col.at<Vec3b>(x,y)[1]=edges[i].points[j].curvature*255/30;
	//		src_col.at<Vec3b>(x,y)[0]=255;
	//		src_col.at<Vec3b>(x,y)[2]=255;
	//	}
	//}
	//imwrite(outputPath,src_col);
	//return temp;
	return edges;
}
//***************************************************************************************************
int getFeatureMatch(staple tempStaple,vector<point> points1,vector<point> points2,double slope1,double slope2 ,double dist_tollerence,double slope_tollerence){
	int count=0,p,q;
	double slope11,slope22,dist11,dist22;
	point p1,p2;
	p1 = tempStaple.p1_img1;
	p2 = tempStaple.p1_img2;
	for( p=0; p < points1.size(); p++)
	{
		dist11 = StructureDistance(p1 , points1[p]); // distance from staples high x edge
		slope11 = StructureSlope(p1 , points1[p]) - slope1;		//slope with repect to the line
		for( q=0; q < points2.size(); q++)
		{
			dist22 = StructureDistance(p2 , points2[q]);	// distance from staples high x edge
			slope22 = StructureSlope(p2 , points2[q]) - slope2;	//slope with repect to the line

			if(Tool::getInstance()->approxComp(dist11,dist22,dist11*dist_tollerence) == 0 && Tool::getInstance()->approxComp(slope11,slope22,slope11*slope_tollerence)==0)
			{
				count++;
				break;
			}
		}
	}
	return count;
}
//***************************************************************************************************
vector<staple> getStaples(vector<point> points1, vector<point> points2)
{
	int num=0;
	double dist1,dist2; //*************************************** change vals below **********************
	double dist_tollerence = 0.02,slope_tollerence = 0.02; //assuming shots are horizontal
	vector<staple> staples;
	double slope1, slope2;

	//Mat src_l = imread(outputPath1);
	//Mat src_r = imread(outputPath2);
	int size1 = points1.size()>20?20:points1.size();
	int size2 = points2.size()>30?30:points2.size();
	for(int j=1; j < size1 ; j++)
	{
		for(int i=0; i < j ; i++)  // curvature of i is greater than j
		{
			dist1 = StructureDistance(points1[i],points1[j]);
			slope1 = StructureSlope(points1[i],points1[j]);
			//if(Tool::getInstance()->approxComp(points1[i].y ,points1[j].y, (double)30 )!=0){ //considering only vertical staples
			//	continue;
			//}
			for(int n=1; n < size2; n++)
			{
				for(int m=0; m < n ; m++)	// curvature of m is greater than n
				{
					//if(Tool::getInstance()->approxComp(points1[j].curvature, points2[n].curvature, points1[j].curvature*curvature_tollerence) != -1)
					//{
					dist2 = StructureDistance(points2[m],points2[n]); 
					slope2 = StructureSlope(points2[m],points2[n]);

					if(Tool::getInstance()->approxComp(dist1,dist2,dist1*dist_tollerence) == 0 && Tool::getInstance()->approxComp(slope1,slope2,slope1*slope_tollerence)==0 )// && Tool::getInstance()->approxComp(points1[i].curvature + points2[j].curvature, points2[m].curvature + points2[n].curvature, (points2[i].curvature + points2[j].curvature)*curvature_tollerence) == 0)
					{
						staple tempStaple(points1[i],points1[j],points2[m],points2[n]);//p1's x is greater than p2's x in each staple
						//printf("dist:: %f %f :::: slopes:: %f %f\n",dist1,dist2,slope1,slope2);
						//printf("Staple info::\n-----------------\nimage-1 : x1=%d, y1=%d, X2= %d , Y2= %d :: Distance = %f\n",tempStaple.p1_img1.x,tempStaple.p1_img1.y,tempStaple.p2_img1.x,tempStaple.p2_img1.y,StructureDistance(tempStaple.p1_img1,tempStaple.p2_img1));
						//printf("image-2 : x1=%d, y1=%d, X2= %d , Y2= %d :: Distance = %f\n",tempStaple.p1_img2.x,tempStaple.p1_img2.y,tempStaple.p2_img2.x,tempStaple.p2_img2.y,StructureDistance(tempStaple.p1_img2,tempStaple.p2_img2));
						tempStaple.NumOfMatch = getFeatureMatch(tempStaple,points1,points2,slope1,slope2,dist_tollerence,slope_tollerence);

						staples.push_back(tempStaple);
						//printf("matches:: %d\n", tempStaple.NumOfMatch);
						if(staples.size() >= 100)
						{
							//int t=12;
							//printf("Staple info::\nimage-1 : \n----------\nx1=%d, y1=%d, X2= %d , Y2= %d :: Distance = %f\n",staples[t].p1_img1.x,staples[t].p1_img1.y,staples[t].p2_img1.x,staples[t].p2_img1.y,StructureDistance(staples[t].p1_img1,staples[t].p2_img1));
							//printf("Staple info::\nimage-2 : \n----------\nx1=%d, y1=%d, X2= %d , Y2= %d :: Distance = %f\n",staples[t].p1_img2.x,staples[t].p1_img2.y,staples[t].p2_img2.x,staples[t].p2_img2.y,StructureDistance(staples[t].p1_img2,staples[t].p2_img2));
							goto hehe;
						}
					}
					//}
					//else
					//{
					//	break;
					//}
				}
			}
		}
	}


hehe:

	return staples;
}
//****************************************************************************************************
Mat blendImages(Mat left, Mat right, double alpha, staple st){
	for(int i =0 ; i < left.cols ; i++){
		for(int j =0 ; j < left.rows ; j++){
			if((left.at<Vec3b>(j,i)[0] <=2 && left.at<Vec3b>(j,i)[1] <=2 && left.at<Vec3b>(j,i)[2] <=2 )|| (right.at<Vec3b>(j,i)[0] <=2 && right.at<Vec3b>(j,i)[1] <=2 && right.at<Vec3b>(j,i)[2] <=2 )){
				left.at<uchar>(j,3*i)  = left.at<uchar>(j,3*i)  + right.at<uchar>(j,3*i) ;
				left.at<uchar>(j,3*i+1)  = left.at<uchar>(j,3*i+1)  + right.at<uchar>(j,3*i+1) ;
				left.at<uchar>(j,3*i+2)  = left.at<uchar>(j,3*i+2)  + right.at<uchar>(j,3*i+2) ;
			}
			else{
				left.at<uchar>(j,3*i) = (left.at<uchar>(j,3*i)  + right.at<uchar>(j,3*i)) / 2; 
				left.at<uchar>(j,3*i+1) = (left.at<uchar>(j,3*i+1) + right.at<uchar>(j,3*i+1) ) / 2; 
				left.at<uchar>(j,3*i+2) = (left.at<uchar>(j,3*i+2)  + right.at<uchar>(j,3*i+2)  ) / 2; 
			}
		}
	}
	return left;
}
//***************************************************************************************************
/** @function main */
int main()
{
	int lol;
	vector<edge> edges1,edges2;
	vector<point> points1,points2;
	vector<staple> staples;

	double overlap = 0.75; //*************************************** overlap **********************


	Mat src_col1 = imread(imgPath1);
	Mat src_col2 = imread(imgPath2);

	int pivot_left=(src_col1.cols+6)*(1-overlap);
	int pivot_right=(src_col2.cols+6)*overlap;

	//left image 
	Mat tempImage;
	tempImage = GetEdgeScheleton(imgPath1);

	vector<point> junction_pts1;
	//junction_pts1 = Tool::getInstance()->getJunctionPoints(tempImage,BWThreshold); // set 1  of new feature points

	edges1 = GetCurvature(tempImage,k);	// k =30 :: the last edge returned contains the junction points
	junction_pts1 = (edges1.back()).points;
	edges1.pop_back();

	points1 = Tool::getInstance()->GetLocalMaxima(edges1,5,k); 
	printf("removing left part of left image\n");
	points1 =  Tool::getInstance()->removeLeft(points1,pivot_left);
	printf("num of feature pts: %d\n",points1.size());
	junction_pts1 =  Tool::getInstance()->removeLeft(junction_pts1,pivot_left);
	points1.insert(points1.end(),junction_pts1.begin(),junction_pts1.end());

	points1 = StructureSort(points1);
	Tool::getInstance()->PlotImage(imgPath1, outputPath1, points1,5,3,0);
	Tool::getInstance()->PlotImage(outputPath1, outputPath1, junction_pts1,3,0,100);

	printf("Left image feature detection complete\n---------------------------------------------\nstarting with right image\n");
	// right image
	tempImage = GetEdgeScheleton(imgPath2); 
	vector<point> junction_pts2;
	
	edges2 = GetCurvature(tempImage,k);	// k =30 :: the last edge returned contains the junction points
	junction_pts2 = (edges2.back()).points;
	edges2.pop_back();

	points2 = Tool::getInstance()->GetLocalMaxima(edges2,5,k);
	printf("removing right part of right image\n");
	points2 =  Tool::getInstance()->removeRight(points2,pivot_right);
	printf("num of feature pts: %d\n",points2.size());
	junction_pts2 =  Tool::getInstance()->removeRight(junction_pts2,pivot_right);
	points2.insert(points2.end(),junction_pts2.begin(),junction_pts2.end());

	points2 = StructureSort(points2);
	Tool::getInstance()->PlotImage(imgPath2, outputPath2, points2,5,3,0);
	Tool::getInstance()->PlotImage(outputPath2, outputPath2, junction_pts2,3,0,100);


	printf("size of 1st pointset :%d\nsize of 2nd point set :%d\n ",points1.size()+junction_pts1.size(),points2.size()+junction_pts1.size());
	printf("Staple collection Started\n");
	staples = getStaples(points1,points2); //***************** some studapa here
	printf("Done with collecting staples\n");

	staple tempStaple;
	int max = 0,maxIndex=0;
	for(int t=0 ; t < staples.size() ; t++){ //********** get the maximum match
		tempStaple = staples[t];
		if(tempStaple.NumOfMatch > max){
			maxIndex = t;
			max = tempStaple.NumOfMatch;
		}
		//printf("Staple info::\n-----------------\nimage-1 : x1=%d, y1=%d, X2= %d , Y2= %d :: Distance = %f\n",tempStaple.p1_img1.x,tempStaple.p1_img1.y,tempStaple.p2_img1.x,tempStaple.p2_img1.y,StructureDistance(tempStaple.p1_img1,tempStaple.p2_img1));
		//printf("image-2 : x1=%d, y1=%d, X2= %d , Y2= %d :: Distance = %f\n",tempStaple.p1_img2.x,tempStaple.p1_img2.y,tempStaple.p2_img2.x,tempStaple.p2_img2.y,StructureDistance(tempStaple.p1_img2,tempStaple.p2_img2));				
		
	}
	int t= maxIndex;
	printf("done with %d number of match in staple number:: %d\n",staples[t].NumOfMatch,t);
	//printf("Staple info::\nimage-1 : \n----------\nx1=%d, y1=%d, X2= %d , Y2= %d :: Distance = %f\n",staples[t].p1_img1.x,staples[t].p1_img1.y,staples[t].p2_img1.x,staples[t].p2_img1.y,StructureDistance(staples[t].p1_img1,staples[t].p2_img1));
	//printf("Staple info::\nimage-2 : \n----------\nx1=%d, y1=%d, X2= %d , Y2= %d :: Distance = %f\n",staples[t].p1_img2.x,staples[t].p1_img2.y,staples[t].p2_img2.x,staples[t].p2_img2.y,StructureDistance(staples[t].p1_img2,staples[t].p2_img2));
	
	Mat src_l = imread(outputPath1);
	Mat src_r = imread(outputPath2);

	printf("\nslope:%lf,dist: %lf\n",StructureSlope(staples[t].p1_img1,staples[t].p2_img1),StructureDistance(staples[t].p1_img1,staples[t].p2_img1));
	printf("slope:%lf,dist: %lf\n",StructureSlope(staples[t].p1_img2,staples[t].p2_img2),StructureDistance(staples[t].p1_img2,staples[t].p2_img2));
	
	cv::line(src_l, Point(staples[t].p1_img1.y,staples[t].p1_img1.x), Point(staples[t].p2_img1.y,staples[t].p2_img1.x), Scalar( 255, 0, 0 ), 3, 8,0);
	cv::line(src_r, Point(staples[t].p1_img2.y,staples[t].p1_img2.x), Point(staples[t].p2_img2.y,staples[t].p2_img2.x), Scalar( 255, 0, 0 ), 3, 8,0);

	cout << "p1_image1.x" << staples[t].p1_img1.x << endl;
	cout << "p1_image2.x" << staples[t].p1_img2.x << endl;
	cout << "p1_image1.y" << staples[t].p1_img1.y << endl;
	cout << "p1_image2.y" << staples[t].p1_img2.y << endl;

	double deltaThetaF,scalingF,theta1,theta2;
	theta1 = StructureSlope(staples[t].p1_img1,staples[t].p2_img1);
	theta2 = StructureSlope(staples[t].p1_img2,staples[t].p2_img2);
	scalingF = StructureDistance(staples[t].p1_img1,staples[t].p2_img1) / StructureDistance(staples[t].p1_img2,staples[t].p2_img2);
	deltaThetaF = (theta1-theta2)*180/3.1415;
	/*int delta_x,delta_y;
	delta_x = staples[t].p1_img1.x - staples[t].p1_img2.x;
	delta_x = delta_x > 0 ? delta_x : -1*delta_x ;
	delta_y = delta_y > 0 ? delta_y : -1*delta_y ;
	*/

	//*remove
	//src_l = imread(outputPath1);
	//src_r = imread(outputPath2);
	//---/remove

	
	//rotate(src_l, -30, src_l);
	//rotateImage(src_l,src_l,90,100,90,0,0,2000,2000);
	Mat M_l(2,3,CV_64F),M_r_scale,M_r,M_r_mov(2,3,CV_64F); // the transformation matrix
	Mat final; // final image

	M_l = 0;
	M_r_mov = 0;
	M_l.at<double>(0,0) = 1;
	M_l.at<double>(1,1) = 1;
	M_l.at<double>(1,2) = (src_l.rows + src_r.rows)/4;
	M_l.at<double>(0,2) = 0;

	M_r_mov.at<double>(0,0) = 1;
	M_r_mov.at<double>(1,1) = 1;
	M_r_mov.at<double>(0,2) = (staples[t].p1_img1.y - staples[t].p1_img2.y) ;
	M_r_mov.at<double>(1,2) = ((src_l.rows + src_r.rows)/4 )+ (staples[t].p1_img1.x - staples[t].p1_img2.x);
	

	//M_r = getRotationMatrix2D((Point2f)(src_l.cols/2.0,src_l.rows/2.0),20,1);//3*(staples[t].p1_img1.x - staples[t].p1_img2.x),((src_l.rows + src_r.rows)/4 )+ staples[t].p1_img1.y
	M_r = getRotationMatrix2D((Point2f)(staples[t].p1_img1.y ,((src_l.rows + src_r.rows)/4 )+ (staples[t].p1_img1.x)),deltaThetaF,1);
	M_r_scale = getRotationMatrix2D((Point2f)(0,0),0,scalingF);

	Size s = src_r.size() + src_l.size();
	
	//scale
	warpAffine(src_r,final,M_r_scale,src_r.size(),1,0,1); // right image
	//translate
	warpAffine(src_l,src_l,M_l,s,1,0,1);		// left image
	warpAffine(final,src_r,M_r_mov,s,1,0,1);	// right image translation
	//rotate
	warpAffine(src_r,final,M_r,s,1,0,1);		// right image rotation
	
	
	imwrite(outputPath2,final);
	imwrite(outputPath1,src_l);
	

	final = blendImages(src_l,final,0.5,staples[t]);
	imwrite(finalOutput,final);
	//joinImages
	/// Wait until user exit program by pressing a key
	scanf("%d",&lol);
	waitKey(0);
	return 0;
}