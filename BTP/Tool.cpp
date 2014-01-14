#include "StdAfx.h"
#include"Tool.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;

Tool* Tool::tool = NULL ;
Tool::Tool(){
}
Tool* Tool::getInstance(){
	tool = new  Tool();
	return tool;
}

template <class T, class U>
/**
tollerence can be +ve or negative both allowed and value is allowed
returns 
-1 => a << b
1  => a >> b
0  => a ~ b
*/
int Tool::approxComp(T a, T b, U tollerence) {
	T diff;
	diff = a-b;
	tollerence = tollerence >= 0 ? tollerence:tollerence*(-1);
	if(diff > tollerence && diff > 0){
		return 1;
	}
	else if( (diff*(-1)) > tollerence && diff < 0){
		return -1;
	}
	else 
		return 0;
}
// Explicit template instantiation
template int Tool::approxComp<double,double>(double,double,double);
template int Tool::approxComp<float,double>(float,float,double);
template int Tool::approxComp<int,double>(int,int,double);
//********************************************************************************************************************
int Tool::getTotalNeighbours(int x, int y, Mat img, int BWThreshold){
	int sum=0;
	if(img.at<uchar>(x-1,y-1)>BWThreshold){
		sum++;
	}
	if(img.at<uchar>(x-1,y+1)>BWThreshold){
		sum++;
	}
	if(img.at<uchar>(x-1,y)>BWThreshold){
		sum++;
	}
	if(img.at<uchar>(x,y-1)>BWThreshold){
		sum++;
	}
	if(img.at<uchar>(x,y+1)>BWThreshold){
		sum++;
	}
	if(img.at<uchar>(x+1,y-1)>BWThreshold){
		sum++;
	}
	if(img.at<uchar>(x+1,y+1)>BWThreshold){
		sum++;
	}
	if(img.at<uchar>(x+1,y)>BWThreshold){
		sum++;
	}
	return sum;
}
//***************************************************************************
int Tool::getChainCode(int k,int l){

	k=(k+1)*10+l+1;
	switch(k){
	case 12:
		return 0;
	case 2:
		return 1;
	case 1:
		return 2;
	case 0:
		return 3;
	case 10:
		return 4;
	case 20:
		return 5;
	case 21:
		return 6;
	case 22:
		return 7;
	}
	return -1;
}
//****************************************************************************************************
bool Tool::checkValidJoin(int i,int j,Mat img,int BWThreshold){	// works only if the total neighbours of point is =" 2 "
	bool isvalid=false;
	int k,l,i1,j1,count=0;
	int x1=0,y1=0,x2=0,y2=0;
	for(k=-1;k<=1;k++){
		for(l=-1;l<=1 ;l++){
			if(k!=0 || l!=0){
				i1=i+k;
				j1=j+l;
				if(img.at<uchar>(i1,j1) > BWThreshold ){
					count+=1;
					if(count==1){
						x1=k;
						y1=l;
					}
					if(count==2){
						x2=k;
						y2=l;
						goto a1;
					}
				}
			}
		}
	}
a1:
	if( ((x1+x2)==0 && (x1!=x2) ) || ((y1+y2)==0 && (y1!=y2)) ){
		isvalid=true;
	}
	return isvalid;
}
//***********************************************************************************************************
Mat Tool::thinEdges_std(Mat temp){
	Mat temp1;
	temp1.create(temp.size(),temp.type());
	Mat element;

	/*
	element = (Mat_<uchar>(3,3) <<  0,0,0,
	0,1,0,
	1,1,1 );
	erode(temp,temp1,element,Point(-1, -1),1,BORDER_CONSTANT,255);
	temp=temp-temp1;

	element = (Mat_<uchar>(3,3) <<  1,0,0,
	1,1,0,
	1,0,0 );
	erode(temp,temp1,element,Point(-1, -1),1,BORDER_CONSTANT,255);
	temp=temp-temp1;

	element = (Mat_<uchar>(3,3) <<  0,0,1,
	0,1,1,
	0,0,1 );
	erode(temp,temp1,element,Point(-1, -1),1,BORDER_CONSTANT,255);
	temp=temp-temp1;

	element = (Mat_<uchar>(3,3) <<  1,1,1,
	0,1,0,
	0,0,0 );
	erode(temp,temp1,element,Point(-1, -1),1,BORDER_CONSTANT,255);
	temp=temp-temp1;
	*/

	element = (Mat_<uchar>(3,3) <<  0,0,0,
		1,1,0,
		0,1,1 );
	erode(temp,temp1,element,Point(-1, -1),1,BORDER_CONSTANT,255);
	temp=temp-temp1;

	element = (Mat_<uchar>(3,3) <<  0,1,0,
		1,1,0,
		1,0,0 );
	erode(temp,temp1,element,Point(-1, -1),1,BORDER_CONSTANT,255);
	temp=temp-temp1;

	element = (Mat_<uchar>(3,3) <<  1,1,0,
		0,1,1,
		0,0,0 );
	erode(temp,temp1,element,Point(-1, -1),1,BORDER_CONSTANT,255);
	temp=temp-temp1;
	element = (Mat_<uchar>(3,3) <<  0,0,1,
		0,1,1,
		0,1,0 );
	erode(temp,temp1,element,Point(-1, -1),1,BORDER_CONSTANT,255);
	temp=temp-temp1;

	element = (Mat_<uchar>(3,3) <<  0,0,0,
		1,1,0,
		0,1,0 );
	erode(temp,temp1,element,Point(-1, -1),1,BORDER_CONSTANT,255);
	temp=temp-temp1;
	element = (Mat_<uchar>(3,3) <<  0,1,0,
		1,1,0,
		0,0,0 );
	erode(temp,temp1,element,Point(-1, -1),1,BORDER_CONSTANT,255);
	temp=temp-temp1;
	element = (Mat_<uchar>(3,3) <<  0,1,0,
		0,1,1,
		0,0,0 );
	erode(temp,temp1,element,Point(-1, -1),1,BORDER_CONSTANT,255);
	temp=temp-temp1;
	element = (Mat_<uchar>(3,3) <<  0,0,0,
		0,1,1,
		0,1,0 );
	erode(temp,temp1,element,Point(-1, -1),1,BORDER_CONSTANT,255);
	temp=temp-temp1;
	return temp-temp1;
}
//**********************************************************************************************************************
vector<point> Tool::getJunctionPoints(Mat img,int BWThreshold){
	printf("detecting junction points::\n");
	int i ,j,n;
	vector<point> points;
	point tempPoint;

	for(i=2;i<img.rows-2;i++){
		for(j=2;j<img.cols-2;j++){
			if(img.at<uchar>(i,j) > BWThreshold){
				n=Tool::getInstance()->getTotalNeighbours(i,j,img,BWThreshold);
				if(n > 2) {
					tempPoint.x = i;
					tempPoint.y = j;
					points.push_back(tempPoint);
				}
			}
		}
	}
	return points;
}

Mat Tool::removePoints(Mat img,vector<point> points,int WHITE_VAL){
	printf("removing junction points::\n");
	for each(point tempPoint in points){
		img.at<uchar>(tempPoint.x,tempPoint.y) = WHITE_VAL;
	}
	return img;
}
//********************************************************************************************************************
Mat Tool::CannyThreshold(Mat tsrc,int lowThreshold,int ratio,int kernel_size,int BWThreshold)
{
	printf("starting canny edge detection ::\n");
	Mat tdst;

	tdst.create( tsrc.size(), tsrc.type() );
	/// Reduce noise with a kernel 3x3
	blur( tsrc, tsrc, Size(3,3) );
	/// Canny detector
	Canny( tsrc, tsrc, lowThreshold, lowThreshold*ratio, kernel_size ,true);
	/// Using Canny's output as a mask, we display our result
	tdst = Scalar::all(0);
	tsrc.copyTo( tdst, tsrc);
	//imshow( window_name, dst );
	threshold( tdst, tdst, BWThreshold, 255,0 );
	return tdst;
} 
//********************************************************************************************************************
Mat Tool::joinEdges(Mat img,int BWThreshold){
	printf("starting joining edges\n");
	int i,j,n,l,k,i1,j1,n1;
	for(i=2;i<img.rows-2;i++){
		for(j=2;j<img.cols-2;j++){
			if(img.at<uchar>(i,j) > BWThreshold ){
				n=Tool::getInstance()->getTotalNeighbours(i,j,img,BWThreshold);
				if(n < 2){
					for(k=-1;k<=1;k++){
						for(l=-1;l<=1 ;l++){
							if (k!=0 || l!=0){
								i1=i+k;
								j1=j+l;
								// printf("<%d:%d> -> <%d:%d>\n",i,j,i1,j1);
								if(img.at<uchar>(i1,j1) < BWThreshold){
									n1=Tool::getInstance()->getTotalNeighbours(i1,j1,img,BWThreshold);
									if(n1==2){
										if(Tool::getInstance()->checkValidJoin(i1,j1,img,BWThreshold)){
											img.at<uchar>(i1,j1)=255;

											goto l1;
										}
									}
								}
							}
						}
					}
l1:;
				}
			}
		}
	}
	return img;
}
//********************************************************************************************************************
vector<point> Tool::GetLocalMaxima(vector<edge> edges,float tollerence,int k)
{
	vector<point> points;
	point tempPoint;
	tempPoint.chainCode=-1;
	int i,j,ptr=-1,flag,prev_flag;
	int totalPoints=0;
	for(i=0;i<edges.size();i++)
	{
		tempPoint.curvature=0;
		points.push_back(tempPoint);
		ptr++;
		prev_flag=-1;
		totalPoints+=edges[i].points.size();
		for(j=k;j<edges[i].points.size();j++)
		{
			flag = approxComp( points[ptr].curvature , edges[i].points[j].curvature , tollerence);
			if(flag == -1)
			{
				points.pop_back();
				points.push_back(edges[i].points[j]);
			}
			else if (flag == 1)
			{
				if(prev_flag == -1)	// if increase then decrease then add a dummy node for the next local maxima
				{
					tempPoint.curvature= edges[i].points[j].curvature;
					points.push_back(tempPoint);
					ptr++;
				}
				if(prev_flag == 1)	// if already on the decreasing slope then decrease more
				{
					points[ptr].curvature=edges[i].points[j].curvature;
				}
			}
			else			// increasing should increase and decreasing should decrease
			{
				if(prev_flag == -1 )
				{
					flag = -1;
					if(points[ptr].curvature < edges[i].points[j].curvature)	// if the local maxima's value is low by small tollerence then also replace
					{ 
						points.pop_back();
						points.push_back(edges[i].points[j]);
					}
				}
				else if(prev_flag == 1){
					flag = 1;
					if(points[ptr].curvature > edges[i].points[j].curvature)	// decreasing then keep on decreasing
					{ 
						points[ptr].curvature=edges[i].points[j].curvature;
					}
				}
			}
			prev_flag=flag;
		}
		if(points[ptr].chainCode==-1) // if the last one is a dummy then remove it 
		{
			points.pop_back();
			ptr--;
		}
	}
	printf("curve len= %d\n",totalPoints);
	return points;
}
//********************************************************************************************************************
vector<edge> Tool::calculateCurvature(int k,vector<edge> edges){
	int i,j,len,l,val=0,f1,f2,f3;
	for(i=0;i<edges.size();i++)
	{
		len=edges[i].points.size();
		for(j=k;j<len-k-1;j++)
		{
			val=0;
			for(l=1;l<=k;l++)
			{
				f1=edges[i].points[j+l].chainCode-edges[i].points[j-l+1].chainCode;
				f2=edges[i].points[j+l+1].chainCode-edges[i].points[j-l+1].chainCode;
				f3=edges[i].points[j+l].chainCode-edges[i].points[j-l].chainCode;
				if(f1<0){ f1 = f1*-1;}
				if(f2<0){ f2 = f2*-1;}
				if(f3<0){ f3 = f3*-1;}
				if(f1>4){ f1 = 8-f1;}
				if(f2>4){ f2 = 8-f2;}
				if(f3>4){ f3 = 8-f3;}
				if(f1>=f2 && f3>=f2 )
				{
					val+=f2;
				}
				else if(f2>=f1 && f3>=f1)
				{
					val+=f1;
				}
				else
				{
					val+=f3;
				}

				//calculate the 6 things and minimum of that
			}
			//printf("%d",val);
			edges[i].points[j].curvature=val; //devide by k if you need exact curvature value
			/*
			if(val==0)
			{
			temp.at<uchar>(edges[i].points[j].x,edges[i].points[j].y) = 1;
			}
			else
			{
			temp.at<uchar>(edges[i].points[j].x,edges[i].points[j].y) = (float)val*255.0/k;
			}
			*/
		}
		//printf("\n");
	}
	return edges;
}
//********************************************************************************************************************
vector<point> Tool::removeLeft( vector<point> points,int p)
{
	int y;
	vector<point> pts;
	for(int j=0 ; j < points.size() ; j++ )
	{
		y=points[j].y;
		if(y >= p)
			pts.push_back(points[j]);
	}
	return pts;
}
//********************************************************************************************************************
vector<point> Tool::removeRight(vector<point> points,int p)
{
	int y;
	vector<point> pts;
	for(int j=0 ; j < points.size() ; j++ )
	{
		y=points[j].y;
		if(y <= p)
			pts.push_back(points[j]);
	}
	return pts;
}
//********************************************************************************************************************
Mat Tool::PlotImage(char* imgPath, char* outputPath, vector<point> points,int thickness)
{
	Mat src_col = imread(imgPath);
	int x,y;
	copyMakeBorder( src_col, src_col, 3, 3, 3, 3, BORDER_CONSTANT, 0);
	for(int j=0 ; j < points.size() ; j++ )
	{
		x=points[j].x;
		y=points[j].y;
		cv::line(src_col, Point(y,x), Point(y,x), Scalar( 0, 0, 0 ), thickness, 8,0);
	}

	imwrite(outputPath,src_col);
	return src_col;
}
//********************************************************************************************************************
Mat Tool::removeCurve(edge tempEdge, Mat temp){
	//printf("the curve to b rremoved ");
	int i1,i2,j1,j2,l,k,l1;
	i1=tempEdge.x1;
	j1=tempEdge.y1;
	i2=tempEdge.x2;
	j2=tempEdge.y2;
	temp.at<uchar>(i1,j1)=0;
	for(l1=0;l1<tempEdge.points.size();l1++)
	{
		i1=tempEdge.points[l1].x;
		j1=tempEdge.points[l1].y;
		temp.at<uchar>(i1,j1)=0;
	}
	temp.at<uchar>(i2,j2)=0;
	return temp;
}

edge Tool::getCircularCurve(Mat img, int threshold, int i, int j){
	edge tempEdge;
	tempEdge.Type = 0;
	int arr[8][2] = { {1,0},{1,-1},{0,-1},{-1,-1},{-1,0},{-1,1},{0,1},{1,1}};
	point b0,b1,c0,c1,temp,b,c;
	b0.x = i;
	b0.y = j;
	c0.x = i-1;
	c0.y = j;
	b0.chainCode = 2; // to be updated  at end
	int t,t1;
	for( t = 0 ; t < 8 ; t++ ){
		t1 = t + b0.chainCode ;
		t1= t1 % 8;
		temp.x = arr[t1][0] + i;
		temp.y = arr[t1][1] + j;
		temp.chainCode = t1;
		if(img.at<uchar>(temp.x,temp.y) > threshold){
			b1 = temp;
			c1.x = arr[(t1+7)%8][0] + i;
			c1.y = arr[(t1+7)%8][1] + j;
			c1.chainCode = (b1.chainCode + 5) % 8; // with respect to b1
			break;
		}
	}

	b=b1;
	c=c1;
	tempEdge.points.push_back(b);
	tempEdge.x1 = b.x;
	tempEdge.y1 = b.y;
	
	while(b0.x != b.x || b0.y !=b.y){
		for( t = 0 ; t < 8 ; t++ ){
			t1 = t + c.chainCode + 1 ;
			t1= t1 % 8;
			temp.x = arr[t1][0] + b.x;
			temp.y = arr[t1][1] + b.y;
			if(img.at<uchar>(temp.x,temp.y) > threshold){
				temp.chainCode = t1;
				c.x = arr[(t1+7)%8][0] + b.x;
				c.y = arr[(t1+7)%8][1] + b.y;
				
				c.chainCode = (temp.chainCode + 5) % 8; // with respect to b1
				b = temp;
				break;
			}
		}
		tempEdge.points.push_back(b);
	}
	tempEdge.x2 = b.x;
	tempEdge.y2 = b.y;

	return tempEdge;
}