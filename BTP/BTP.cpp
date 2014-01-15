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
	double x= p1.x-p2.x;
	double y= p1.y-p2.y;
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

	imwrite("E:/personal/acads/BTP/images/Set1/scaled2/set18_smallEdgesRemoved_temp1.png",temp);
	imwrite("E:/personal/acads/BTP/images/Set1/scaled2/set18_smallEdgesRemoved_img.png",img);
	temp1 = temp1 - img;
	printf("total %d edges detected and %d are taken into consideration ::\n",totalEdges,edges.size());
	imwrite("E:/personal/acads/BTP/images/Set1/scaled2/set18_smallEdgesRemoved.png",temp1);

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
	imwrite(outputPath1_canny,temp);
	copyMakeBorder( temp, temp, 3, 3, 3, 3, BORDER_CONSTANT, 0);
	//copyMakeBorder( src_col, src_col, 3, 3, 3, 3, BORDER_CONSTANT, 0);

	temp = Tool::getInstance()->thinEdges_std(temp);
	//imwrite(outputPath1_thin,temp);
	//temp = Tool::getInstance()->joinEdges(temp,BWThreshold);
	imwrite(outputPath1_thin,temp);
	return temp;
}
//****************************************************************************************************
vector<edge> GetCurvature(Mat temp,int k){
	vector<edge> edges;  
	vector<point> junction_pts;

	junction_pts = Tool::getInstance()->getJunctionPoints(temp,BWThreshold); //TODO : return the junction point obtained here
	temp = Tool::getInstance()->removePoints(temp,junction_pts,0);

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
	double dist1,dist2;
	double dist_tollerence = 0.02,slope_tollerence = 0.02,position_tollerence = 0.8 , curvature_tollerence = 0.5; //assuming shots are horizontal
	vector<staple> staples;
	double slope1, slope2;

	//Mat src_l = imread(outputPath1);
	//Mat src_r = imread(outputPath2);
	int size1 = points1.size()>10?10:points1.size();
	int size2 = points2.size()>50?50:points2.size();
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
//***************************************************************************************************
/** @function main */
int main()
{
	int lol;
	vector<edge> edges1,edges2;
	vector<point> points1,points2;
	vector<staple> staples;

	double overlap = 0.55;

	Mat src_col1 = imread(imgPath1);
	Mat src_col2 = imread(imgPath2);

	int pivot_left=(src_col1.cols+6)*(1-overlap);
	int pivot_right=(src_col2.cols+6)*overlap;

	//left image 
	Mat tempImage;
	tempImage = GetEdgeScheleton(imgPath1);

	vector<point> junction_pts1;
	junction_pts1 = Tool::getInstance()->getJunctionPoints(tempImage,BWThreshold); // set 1  of new feature points

	edges1 = GetCurvature(tempImage,k);	// k =30
	points1 = Tool::getInstance()->GetLocalMaxima(edges1,5,k); 
	printf("pts: %d\n",points1.size());
	printf("removing left part of left image\n");
	points1 =  Tool::getInstance()->removeLeft(points1,pivot_left);
	junction_pts1 =  Tool::getInstance()->removeLeft(junction_pts1,pivot_left);
	points1 = StructureSort(points1);
	Tool::getInstance()->PlotImage(imgPath1, outputPath1, points1,5,3,0);
	Tool::getInstance()->PlotImage(outputPath1, outputPath1, junction_pts1,3,0,100);

	printf("Left image feature detection complete\n\nstarting with right image\n");
	// right image
	tempImage = GetEdgeScheleton(imgPath2); 
	vector<point> junction_pts2;
	junction_pts2 = Tool::getInstance()->getJunctionPoints(tempImage,BWThreshold); // set 2  of new feature points
	edges2 = GetCurvature(tempImage,k);	// k =30
	points2 = Tool::getInstance()->GetLocalMaxima(edges2,5,k);
	printf("pts: %d\n",points2.size());
	printf("removing right part of right image\n");
	points2 =  Tool::getInstance()->removeRight(points2,pivot_right);
	junction_pts2 =  Tool::getInstance()->removeRight(junction_pts2,pivot_right);
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
		/*printf("Staple info::\n-----------------\nimage-1 : x1=%d, y1=%d, X2= %d , Y2= %d :: Distance = %f\n",tempStaple.p1_img1.x,tempStaple.p1_img1.y,tempStaple.p2_img1.x,tempStaple.p2_img1.y,StructureDistance(tempStaple.p1_img1,tempStaple.p2_img1));
		printf("image-2 : x1=%d, y1=%d, X2= %d , Y2= %d :: Distance = %f\n",tempStaple.p1_img2.x,tempStaple.p1_img2.y,tempStaple.p2_img2.x,tempStaple.p2_img2.y,StructureDistance(tempStaple.p1_img2,tempStaple.p2_img2));				
		*/
	}
	int t= maxIndex;
	printf("done with %d number of match in staple number:: %d\n",staples[t].NumOfMatch,t);
	/*printf("Staple info::\nimage-1 : \n----------\nx1=%d, y1=%d, X2= %d , Y2= %d :: Distance = %f\n",staples[t].p1_img1.x,staples[t].p1_img1.y,staples[t].p2_img1.x,staples[t].p2_img1.y,StructureDistance(staples[t].p1_img1,staples[t].p2_img1));
	printf("Staple info::\nimage-2 : \n----------\nx1=%d, y1=%d, X2= %d , Y2= %d :: Distance = %f\n",staples[t].p1_img2.x,staples[t].p1_img2.y,staples[t].p2_img2.x,staples[t].p2_img2.y,StructureDistance(staples[t].p1_img2,staples[t].p2_img2));
	*/
	Mat src_l = imread(outputPath1);
	Mat src_r = imread(outputPath2);
	cv::line(src_l, Point(staples[t].p1_img1.y,staples[t].p1_img1.x), Point(staples[t].p2_img1.y,staples[t].p2_img1.x), Scalar( 255, 0, 0 ), 3, 8,0);
	cv::line(src_r, Point(staples[t].p1_img2.y,staples[t].p1_img2.x), Point(staples[t].p2_img2.y,staples[t].p2_img2.x), Scalar( 255, 0, 0 ), 3, 8,0);

	imwrite(outputPath1,src_l);
	imwrite(outputPath2,src_r);

	/// Wait until user exit program by pressing a key
	scanf("%d",&lol);
	waitKey(0);
	return 0;
}