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

int k=30; //curvature estimate
int curveLenThreshold = 61; // 2*k + 1

char* window_name = "Edge Map";
char* imgPath1 =  "E:/personal/acads/BTP/images/Set1/scaled2/set1.jpg";
char* outputPath1 = "E:/personal/acads/BTP/images/Set1/scaled2/set1_out2.png";
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
bool compStaples(staple p1, staple p2){
	return (p1.NumOfMatch > p2.NumOfMatch);
}
vector<point> StructureSort(vector<point> points){
	sort(points.begin(),points.end(),comp);
	return points;
}
//*****************************squared distance is returned***********************************************************************
double StructureDistance(point p1,point p2){
	double dist;
	int x= p1.x-p2.x;
	int y= p1.y-p2.y;
	dist = (x*x) + (y*y);
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
vector<edge> removeSmallCurves(int th, Mat img,vector<edge> edges){
	printf("started removing small curves\n");
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
					tempEdge.points.clear();
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

	for(i=2;i<img.rows-2;i++)
	{
		for(j=2;j<img.cols-2;j++)
		{
			if(img.at<uchar>(i,j) > BWThreshold && temp.at<uchar>(i,j) < BWThreshold )
			{
				n=Tool::getInstance()->getTotalNeighbours(i,j,img,BWThreshold);
				if(n==2)
				{
					printf("LOL");
				}
			}
		}
	}



	imwrite("E:/personal/acads/BTP/images/Set1/scaled2/set18_smallEdgesRemoved_temp1.png",temp);
	imwrite("E:/personal/acads/BTP/images/Set1/scaled2/set18_smallEdgesRemoved_img.png",img);
	temp1 = temp1 - img;
	printf("total %d edges detected and %d are taken into consideration ::\n\n",totalEdges,edges.size());
	imwrite("E:/personal/acads/BTP/images/Set1/scaled2/set18_smallEdgesRemoved.png",temp1);

	//imwrite(outputPath1,img);
	return edges;
}
//****************************************************************************************************
Mat GetEdgeScheleton(char* imgPath){
	Mat src = imread(imgPath,0);
	if( !src.data ){ 
		printf("error-noimage found");
	}

	Mat temp;
	temp.create(src.size(), src.type());

	temp=Tool::getInstance()->CannyThreshold(src,lowThreshold,ratio,kernel_size,BWThreshold);
	imwrite(outputPath1_canny,temp);
	copyMakeBorder( temp, temp, 3, 3, 3, 3, BORDER_CONSTANT, 0);
	//copyMakeBorder( src_col, src_col, 3, 3, 3, 3, BORDER_CONSTANT, 0);

	temp = Tool::getInstance()->thinEdges_std(temp);
	//imwrite(outputPath1_thin,temp);
	//temp = Tool::getInstance()->joinEdges(temp,BWThreshold);
	imwrite(outputPath1_thin,temp);
	return temp;
}

vector<edge> GetCurvature(Mat temp,int k){
	vector<edge> edges; 
	vector<point> junction_pts;

	junction_pts = Tool::getInstance()->getJunctionPoints(temp,BWThreshold); //TODO : return the junction point obtained here
	temp = Tool::getInstance()->removeJunctionPoints(temp,junction_pts,0);
	
	//edges = extractCurves(temp);

	edges = removeSmallCurves(curveLenThreshold,temp,edges);
	//imwrite(outputPath1_short,temp);

	edges = Tool::getInstance()->calculateCurvature(k,edges);

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
vector<staple> getStaples(vector<point> points1, vector<point> points2)
{
	int num=0;
	double dist1,dist2;

	double dist_tollerence = 0.03,slope_tollerence = 0.08,position_tollerence = 0.8 , curvature_tollerence = 0.5; //assuming shots are horizontal

	vector<staple> staples;

	double slope1, slope2;

	//Mat src_l = imread(outputPath1);
	//Mat src_r = imread(outputPath2);

	for(int j=1; j < points1.size(); j++)
	{
		for(int i=0; i < j ; i++)  // curvature of i is greater than j
		{
			dist1 = StructureDistance(points1[i],points1[j]);
			slope1 = StructureSlope(points1[i],points1[j]);

			//if(Tool::getInstance()->approxComp(points1[i].y ,points1[j].y, (double)30 )!=0){ //considering only vertical staples
			//	continue;
			//}

			for(int n=1; n < points2.size(); n++)
			{
				for(int m=0; m < n ; m++)	// curvature of m is greater than n
				{
					//if(Tool::getInstance()->approxComp(points1[j].curvature, points2[n].curvature, points1[j].curvature*curvature_tollerence) != -1)
					//{
					dist2 = StructureDistance(points2[m],points2[n]); 
					slope2 = StructureSlope(points2[m],points2[n]);

					//p1's x is greater than p2's x in each staple
					staple tempStaple(points1[i],points1[j],points2[m],points2[n]);

					if(Tool::getInstance()->approxComp(dist1,dist2,dist1*dist_tollerence) == 0 && Tool::getInstance()->approxComp(slope1,slope2,slope1*slope_tollerence)==0 )// && Tool::getInstance()->approxComp(points1[i].curvature + points2[j].curvature, points2[m].curvature + points2[n].curvature, (points2[i].curvature + points2[j].curvature)*curvature_tollerence) == 0)
					{


						for(int p=0; p < points1.size(); p++)
						{
							dist1 = StructureDistance(tempStaple.p1_img1 , points1[p]); // distance from staples high x edge
							slope1 = StructureSlope(tempStaple.p1_img1 , points1[p]) - slope1;		//slope with repect to the line

							for(int q=0; q < points2.size(); q++)
							{
								//if(approxComp(points1[p].curvature, points2[q].curvature, points1[p].curvature*curvature_tollerence) != -1)
								//{
								dist2 = StructureDistance(tempStaple.p1_img2 , points2[q]);	// distance from staples high x edge
								slope2 = StructureSlope(tempStaple.p1_img2 , points2[q]) - slope2;	//slope with repect to the line

								if(Tool::getInstance()->approxComp(dist1,dist2,dist1*dist_tollerence) == 0 && Tool::getInstance()->approxComp(slope1,slope2,slope1*slope_tollerence)==0)
								{
									tempStaple.NumOfMatch++ ;
									break;
								}
								//}
								//else
								//{
								//	break;
								//}
							}
						}
						staples.push_back(tempStaple);
						//printf("%d ,", tempStaple.NumOfMatch);
						//cv::line(src_l, Point(points1[i].y,points1[i].x), Point(points1[j].y,points1[j].x), Scalar( 0, 0, 255 ), 1, 8,0);
						//cv::line(src_r, Point(points2[m].y,points2[m].x), Point(points2[n].y,points2[n].x), Scalar( 0, 0, 255 ), 1, 8,0);
						if(staples.size() >= 5000)
						{
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

	sort(staples.begin(),staples.end(),compStaples);
	return staples;
}
//***************************************************************************************************
/** @function main */
int main()
{
	int lol;
	vector<edge> edges1,edges2;
	vector<point> points1,points2;
	vector<staple> staples;

	float overlap = 0.40;

	Mat src_col1 = imread(imgPath1);
	Mat src_col2 = imread(imgPath2);

	int pivot_left=(src_col1.cols+6)*(1-overlap);
	int pivot_right=(src_col2.cols+6)*overlap;

	//left image 
	Mat tempImage;
	tempImage = GetEdgeScheleton(imgPath1); 
	edges1 = GetCurvature(tempImage,k);	// k =30
	points1 = Tool::getInstance()->GetLocalMaxima(edges1,5,k); 
	printf("pts: %d\n",points1.size());
	points1 =  Tool::getInstance()->removeLeft(points1,pivot_left);
	points1 = StructureSort(points1);
	Tool::getInstance()->PlotImage(imgPath1, outputPath1, points1,5);

	// right image
	tempImage = GetEdgeScheleton(imgPath1); 
	edges2 = GetCurvature(tempImage,k);	// k =30
	points2 = Tool::getInstance()->GetLocalMaxima(edges2,5,k);
	printf("pts: %d\n",points2.size());
	points2 =  Tool::getInstance()->removeRight(points2,pivot_right);
	points2 = StructureSort(points2);
	Tool::getInstance()->PlotImage(imgPath2, outputPath2, points2,5);

	printf("size of 1st pointset :%d\nsize of second point set :%d\n ",points1.size(),points2.size());
	
	staples = getStaples(points1,points2);
	printf("done with %d\n",staples[0].NumOfMatch);
	
	Mat src_l = imread(outputPath1);
	Mat src_r = imread(outputPath2);
	//******************************************************************************


	cv::line(src_l, Point(staples[0].p1_img1.y,staples[0].p1_img1.x), Point(staples[0].p2_img1.y,staples[0].p2_img1.x), Scalar( 255, 0, 0 ), 3, 8,0);
	cv::line(src_r, Point(staples[0].p1_img2.y,staples[0].p1_img2.x), Point(staples[0].p2_img2.y,staples[0].p2_img2.x), Scalar( 255, 0, 0 ), 3, 8,0);

	//cv::line(src_l, Point(staples[1].p1_img1.y,staples[1].p1_img1.x), Point(staples[1].p2_img1.y,staples[0].p2_img1.x), Scalar( 0, 250, 0 ), 3, 8,0);
	//cv::line(src_r, Point(staples[1].p1_img2.y,staples[1].p1_img2.x), Point(staples[1].p2_img2.y,staples[0].p2_img2.x), Scalar( 0, 250, 0 ), 3, 8,0);

	imwrite(outputPath1,src_l);
	imwrite(outputPath2,src_r);

	/// Wait until user exit program by pressing a key
	scanf("%d",&lol);
	waitKey(0);
	return 0;
}