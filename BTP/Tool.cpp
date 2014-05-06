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
	//printf("detecting junction points::\n");
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
//********************************************************************************************************************
Mat Tool::removePoints(Mat img,vector<point> points,int WHITE_VAL){
	//printf("removing junction points::\n");
	for each(point tempPoint in points){
		img.at<uchar>(tempPoint.x,tempPoint.y) = WHITE_VAL;
	}
	return img;
}
//********************************************************************************************************************
Mat Tool::CannyThreshold(Mat tsrc,int lowThreshold,int ratio,int kernel_size,int BWThreshold)
{
	//printf("starting canny edge detection ::\n");
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
	//printf("starting joining edges\n");
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
	int i,j,ptr=-1,flag,prev_flag,k_start;
	int totalPoints=0;
	for(i=0;i<edges.size();i++)
	{
		
		tempPoint.curvature=0;
		points.push_back(tempPoint);
		ptr++;
		prev_flag=-1;
		totalPoints+=edges[i].points.size();
		if(edges[i].Type == 1){
			k_start = 0;
		}
		else
		{
			k_start = k;
		}
		for(j=k_start;j<edges[i].points.size();j++)
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
	//printf("curve len= %d\n",totalPoints);
	return points;
}
//********************************************************************************************************************
vector<edge> Tool::calculateCurvature(int k,vector<edge> edges){
	int i,j,len,l,val=0,f1,f2,f3;
	for(i=0;i<edges.size();i++)
	{
		if(edges[i].Type == 1) //
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
			}
		}
		else
		{// circular path
			len=edges[i].points.size();
			for( j = 0 ; j < len ; j++ )
			{
				val=0;
				for(l=1;l<=k;l++)
				{
					f1=edges[i].points[(j+l)%len].chainCode-edges[i].points[(j-l+1+len)%len].chainCode;
					f2=edges[i].points[(j+l+1)%len].chainCode-edges[i].points[(j-l+1+len)%len].chainCode;
					f3=edges[i].points[(j+l)%len].chainCode-edges[i].points[(j-l+len)%len].chainCode;
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
				}
				edges[i].points[j].curvature=val; //devide by k if you need exact curvature value
			}
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
Mat Tool::PlotImage(char* imgPath, char* outputPath, vector<point> points,int thickness,int padding, int PixVal)
{
	Mat src_col = imread(imgPath);
	int x,y;
	copyMakeBorder( src_col, src_col, padding, padding, padding, padding, BORDER_CONSTANT, 0);
	for(int j=0 ; j < points.size() ; j++ )
	{
		x=points[j].x;
		y=points[j].y;
		cv::line(src_col, Point(y,x), Point(y,x), Scalar( PixVal, 0, 0), thickness, 8,0);
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
//********************************************************************************************************************
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
//****************************************************************************************************
Mat Tool::GetEdgeScheleton(char* imgPath,char* outputPath1_canny,char* outputPath1_thin, int lowThreshold,int ratio,int kernel_size,int BWThreshold){
	Mat src = imread(imgPath,0);
	if( !src.data ){ 
		printf("error-noimage found\n");
	}

	Mat temp;
	temp.create(src.size(), src.type());

	temp=CannyThreshold(src,lowThreshold,ratio,kernel_size,BWThreshold);
	imwrite(outputPath1_canny,temp);
	copyMakeBorder( temp, temp, 3, 3, 3, 3, BORDER_CONSTANT, 0);
	//copyMakeBorder( src_col, src_col, 3, 3, 3, 3, BORDER_CONSTANT, 0);

	temp = Tool::getInstance()->thinEdges_std(temp);
	imwrite(outputPath1_thin,temp);
	//temp = Tool::getInstance()->joinEdges(temp,BWThreshold);
	//imwrite(outputPath1_thin,temp);
	return temp;
}
//****************************************************************************************************
Mat Tool::blendImages1(Mat left, Mat right, double alpha, staple st){	// type 1
	for(int i =0 ; i < left.cols ; i++){
		for(int j =0 ; j < left.rows ; j++){
			if((left.at<Vec3b>(j,i)[0] <=2 && left.at<Vec3b>(j,i)[1] <=2 && left.at<Vec3b>(j,i)[2] <=2 )|| (right.at<Vec3b>(j,i)[0] <=2 && right.at<Vec3b>(j,i)[1] <=2 && right.at<Vec3b>(j,i)[2] <=2 )){
				left.at<uchar>(j,3*i)  = left.at<uchar>(j,3*i)  + right.at<uchar>(j,3*i) ;
				left.at<uchar>(j,3*i+1)  = left.at<uchar>(j,3*i+1)  + right.at<uchar>(j,3*i+1) ;
				left.at<uchar>(j,3*i+2)  = left.at<uchar>(j,3*i+2)  + right.at<uchar>(j,3*i+2) ;
			}
			else{
				left.at<uchar>(j,3*i) = (alpha*left.at<uchar>(j,3*i)  + (1-alpha)*right.at<uchar>(j,3*i)) ; 
				left.at<uchar>(j,3*i+1) = (alpha*left.at<uchar>(j,3*i+1) + (1-alpha)*right.at<uchar>(j,3*i+1) ) ; 
				left.at<uchar>(j,3*i+2) = (alpha*left.at<uchar>(j,3*i+2)  + (1-alpha)*right.at<uchar>(j,3*i+2)  ) ; 
			}
		}
	}
	return left;
}
//***************************************************************************************************
Mat Tool::blendImages2(Mat left, Mat right, double alpha, staple st){	// type 2
	for(int i =0 ; i < left.cols ; i++){
		for(int j =0 ; j < left.rows ; j++){
			if((i >= min(st.p1_img1.y, st.p2_img1.y)) && (i <= max(st.p1_img1.y, st.p2_img1.y))) {
				if((left.at<Vec3b>(j,i)[0] <=2 && left.at<Vec3b>(j,i)[1] <=2 && left.at<Vec3b>(j,i)[2] <=2 )|| (right.at<Vec3b>(j,i)[0] <=2 && right.at<Vec3b>(j,i)[1] <=2 && right.at<Vec3b>(j,i)[2] <=2 )){
					left.at<uchar>(j,3*i)  = left.at<uchar>(j,3*i)  + right.at<uchar>(j,3*i) ;
					left.at<uchar>(j,3*i+1)  = left.at<uchar>(j,3*i+1)  + right.at<uchar>(j,3*i+1) ;
					left.at<uchar>(j,3*i+2)  = left.at<uchar>(j,3*i+2)  + right.at<uchar>(j,3*i+2) ;
				}
				else{
					left.at<uchar>(j,3*i) = (alpha*left.at<uchar>(j,3*i)  + (1-alpha)*right.at<uchar>(j,3*i)) ; 
					left.at<uchar>(j,3*i+1) = (alpha*left.at<uchar>(j,3*i+1) + (1-alpha)*right.at<uchar>(j,3*i+1) ) ; 
					left.at<uchar>(j,3*i+2) = (alpha*left.at<uchar>(j,3*i+2)  + (1-alpha)*right.at<uchar>(j,3*i+2)  ) ; 
				}
			}
			else if(i > st.p2_img1.y){
					left.at<uchar>(j,3*i)  = right.at<uchar>(j,3*i) ;
					left.at<uchar>(j,3*i+1)  = right.at<uchar>(j,3*i+1) ;
					left.at<uchar>(j,3*i+2)  = right.at<uchar>(j,3*i+2) ;
			}
			else{
				// nothing
			}
		}
	}
	return left;
}
//***************************************************************************************************
Mat Tool::blendImages3(Mat left, Mat right, staple st){	// type 3 dynamic alpha
	double alpha;
	int deltaX = st.p1_img1.y - st.p2_img1.y;
	deltaX = deltaX > 0 ? deltaX: -deltaX;
	for(int i =0 ; i < left.cols ; i++){
		for(int j =0 ; j < left.rows ; j++){
			if((i >= min(st.p1_img1.y, st.p2_img1.y)) && (i <= max(st.p1_img1.y, st.p2_img1.y))) {
				if((left.at<Vec3b>(j,i)[0] <=2 && left.at<Vec3b>(j,i)[1] <=2 && left.at<Vec3b>(j,i)[2] <=2 )|| (right.at<Vec3b>(j,i)[0] <=2 && right.at<Vec3b>(j,i)[1] <=2 && right.at<Vec3b>(j,i)[2] <=2 )){
					left.at<uchar>(j,3*i)  = left.at<uchar>(j,3*i)  + right.at<uchar>(j,3*i) ;
					left.at<uchar>(j,3*i+1)  = left.at<uchar>(j,3*i+1)  + right.at<uchar>(j,3*i+1) ;
					left.at<uchar>(j,3*i+2)  = left.at<uchar>(j,3*i+2)  + right.at<uchar>(j,3*i+2) ;
				}
				else{
					alpha = (double)(i - min(st.p1_img1.y, st.p2_img1.y)) / (double)deltaX;
					alpha = 1 - alpha;
					left.at<uchar>(j,3*i) = (alpha*left.at<uchar>(j,3*i)  + (1-alpha)*right.at<uchar>(j,3*i)) ; 
					left.at<uchar>(j,3*i+1) = (alpha*left.at<uchar>(j,3*i+1) + (1-alpha)*right.at<uchar>(j,3*i+1) ) ; 
					left.at<uchar>(j,3*i+2) = (alpha*left.at<uchar>(j,3*i+2)  + (1-alpha)*right.at<uchar>(j,3*i+2)  ) ; 
				}
			}
			else if(i > st.p2_img1.y){
					left.at<uchar>(j,3*i)  = right.at<uchar>(j,3*i) ;
					left.at<uchar>(j,3*i+1)  = right.at<uchar>(j,3*i+1) ;
					left.at<uchar>(j,3*i+2)  = right.at<uchar>(j,3*i+2) ;
			}
			else{
				// nothing
			}
		}
	}
	return left;
}
//***************************************************************************************************
Mat Tool::blendImages4(Mat left, Mat right, staple st){	// type 3 dynamic alpha (gaussian function variation)
	double alpha;
	int deltaX = st.p1_img1.y - st.p2_img1.y;
	deltaX = deltaX > 0 ? deltaX: -deltaX;
	for(int i =0 ; i < left.cols ; i++){
		for(int j =0 ; j < left.rows ; j++){
			if((i >= min(st.p1_img1.y, st.p2_img1.y)) && (i <= max(st.p1_img1.y, st.p2_img1.y))) {
				if((left.at<Vec3b>(j,i)[0] <=2 && left.at<Vec3b>(j,i)[1] <=2 && left.at<Vec3b>(j,i)[2] <=2 )|| (right.at<Vec3b>(j,i)[0] <=2 && right.at<Vec3b>(j,i)[1] <=2 && right.at<Vec3b>(j,i)[2] <=2 )){
					left.at<uchar>(j,3*i)  = left.at<uchar>(j,3*i)  + right.at<uchar>(j,3*i) ;
					left.at<uchar>(j,3*i+1)  = left.at<uchar>(j,3*i+1)  + right.at<uchar>(j,3*i+1) ;
					left.at<uchar>(j,3*i+2)  = left.at<uchar>(j,3*i+2)  + right.at<uchar>(j,3*i+2) ;
				}
				else{
					alpha = (double)(i - ((st.p1_img1.y + st.p2_img1.y)/2));
					alpha = alpha > 0? alpha:-alpha;
					alpha = (-1*pow(alpha,2))/((st.p1_img1.y + st.p2_img1.y)/2);
					alpha = 0.5 * pow(2.71,alpha);
					if(i < ((st.p1_img1.y + st.p2_img1.y) /2 ) ){
						alpha = 1 - alpha;
					}
					left.at<uchar>(j,3*i) = (alpha*left.at<uchar>(j,3*i)  + (1-alpha)*right.at<uchar>(j,3*i)) ; 
					left.at<uchar>(j,3*i+1) = (alpha*left.at<uchar>(j,3*i+1) + (1-alpha)*right.at<uchar>(j,3*i+1) ) ; 
					left.at<uchar>(j,3*i+2) = (alpha*left.at<uchar>(j,3*i+2)  + (1-alpha)*right.at<uchar>(j,3*i+2)  ) ; 
				}
			}
			else if(i > st.p2_img1.y){
					left.at<uchar>(j,3*i)  = right.at<uchar>(j,3*i) ;
					left.at<uchar>(j,3*i+1)  = right.at<uchar>(j,3*i+1) ;
					left.at<uchar>(j,3*i+2)  = right.at<uchar>(j,3*i+2) ;
			}
			else{
				// nothing
			}
		}
	}
	return left;
}
//***************************************************************************************************
Mat Tool::blendImages5(Mat left, Mat right, staple st){	// type 3 dynamic alpha ecponential
	double alpha;
	int deltaX = st.p1_img1.y - st.p2_img1.y;
	deltaX = deltaX > 0 ? deltaX: -deltaX;
	double pos;
	double denom = 2*pow((double)(deltaX)/2,2); 
	for(int i =0 ; i < left.cols ; i++){
		for(int j =0 ; j < left.rows ; j++){
			if((i >= min(st.p1_img1.y, st.p2_img1.y)) && (i <= max(st.p1_img1.y, st.p2_img1.y))) {
				if((left.at<Vec3b>(j,i)[0] <=2 && left.at<Vec3b>(j,i)[1] <=2 && left.at<Vec3b>(j,i)[2] <=2 )|| (right.at<Vec3b>(j,i)[0] <=2 && right.at<Vec3b>(j,i)[1] <=2 && right.at<Vec3b>(j,i)[2] <=2 )){
					left.at<uchar>(j,3*i)  = left.at<uchar>(j,3*i)  + right.at<uchar>(j,3*i) ;
					left.at<uchar>(j,3*i+1)  = left.at<uchar>(j,3*i+1)  + right.at<uchar>(j,3*i+1) ;
					left.at<uchar>(j,3*i+2)  = left.at<uchar>(j,3*i+2)  + right.at<uchar>(j,3*i+2) ;
				}
				else{
					pos = i - min(st.p1_img1.y, st.p2_img1.y);
					if(pos <= (double)deltaX/2){
						alpha = pow((2*(double)pos)/deltaX,6) / 2;
						alpha = 1 - alpha; 
					}
					else{
						alpha = pow((2*(deltaX-(double)pos))/deltaX,6) / 2;
					}
						
					
					left.at<uchar>(j,3*i) = (alpha*left.at<uchar>(j,3*i)  + (1-alpha)*right.at<uchar>(j,3*i)) ; 
					left.at<uchar>(j,3*i+1) = (alpha*left.at<uchar>(j,3*i+1) + (1-alpha)*right.at<uchar>(j,3*i+1) ) ; 
					left.at<uchar>(j,3*i+2) = (alpha*left.at<uchar>(j,3*i+2)  + (1-alpha)*right.at<uchar>(j,3*i+2)  ) ; 
				}
			}
			else if(i > st.p2_img1.y){
					left.at<uchar>(j,3*i)  = right.at<uchar>(j,3*i) ;
					left.at<uchar>(j,3*i+1)  = right.at<uchar>(j,3*i+1) ;
					left.at<uchar>(j,3*i+2)  = right.at<uchar>(j,3*i+2) ;
			}
			else{
				// nothing
			}
		}
	}
	return left;
}
//***************************************************************************************************
staple Tool::getFeatureMatch(staple tempStaple,vector<point> points1,vector<point> points2,double slope1,double slope2 ,double dist_tollerence,double slope_tollerence){
	int count=0,p,q;
	double slope11,slope22,dist11,dist22;
	double reqSlope,reqDist;
	point expectedPoint;
	pair<double,double> tempPos;
	point p1,p2;
	bool matchFound;
	double minTot,tempD,tempA,tempC,tempTot;
	pair<point,point> tempPair;
	p1 = tempStaple.p1_img1;
	p2 = tempStaple.p1_img2;
	double scalingF = StructureDistance(tempStaple.p1_img1,tempStaple.p2_img1) / StructureDistance(tempStaple.p1_img2,tempStaple.p2_img2);
	
	for( p=0; p < points1.size(); p++)
	{
		dist11 = StructureDistance(p1 , points1[p]); // distance from staples high x edge
		slope11 = StructureSlope(p1 , points1[p]) - slope1;		//slope with repect to the line
		reqSlope = slope11 + slope2;
		reqDist = dist11/scalingF;
		expectedPoint.x = p2.x + reqDist*cos(reqSlope);
		expectedPoint.y = p2.y + reqDist*sin(reqSlope);
		matchFound = false;minTot = 1000000000;
		//printf("\n");
		for( q=0; q < points2.size(); q++)
		{
			dist22 = StructureDistance(p2 , points2[q])*scalingF;	// distance from staples high x edge with scaling factor
			slope22 = StructureSlope(p2 , points2[q]) - slope2;	//slope with repect to the line
			//absolute value considered---------------------------------------------------------------------------------------------
			if(Tool::getInstance()->approxComp(dist11,dist22,dist11*dist_tollerence) == 0 && Tool::getInstance()->approxComp(slope11,slope22,slope11*slope_tollerence)==0 && abs((int)expectedPoint.x-points2[q].x) <= 10 && abs((int)expectedPoint.y-points2[q].y) <= 10)
			{
				matchFound = true; tempD = abs(dist22-dist11); tempA=abs(-slope11+slope22); tempC = abs(points1[p].curvature - points2[q].curvature);
				tempTot = tempA*tempD*tempC;
				if(tempTot < minTot){
					minTot = tempTot;
					tempPair = make_pair(points1[p],points2[q]);
					tempPos = make_pair(expectedPoint.x-points2[q].x,expectedPoint.y-points2[q].y);
					//printf("(%d,%d)", (int)expectedPoint.x-points2[q].x,(int)expectedPoint.y-points2[q].y);
				}
				
				break;
			}
			
		}
		if(matchFound){
			tempStaple.match.push_back(tempPair);
			count++;
		}
		
	}
	tempStaple.NumOfMatch = count;
	return tempStaple;
}
//*****************************squared distance is returned***********************************************************************
double Tool::StructureDistance(point p1,point p2){
	double dist;
	double x= p1.x-p2.x;
	double y= p1.y-p2.y;
	dist = (x*x) + (y*y);
	dist = sqrt(dist);
	return dist;
}
//********************************* radian slope **************************************************************
double Tool::StructureSlope(point p1,point p2) // angle slope
{
	double slope;
	double x= p1.x-p2.x;
	double y= p1.y-p2.y;
	slope =atan2(y,x) + 3.15;
	return slope;
}
//***************************************************************************************************
vector<staple> Tool::getStaples(vector<point> points1, vector<point> points2,double dist_tollerence,double slope_tollerence)//slope tollerence is higher
{
	int num=0;
	double dist1,dist2; //*************************************************************
	//double dist_tollerence = 0.02,slope_tollerence = 0.02; //assuming shots are horizontal
	vector<staple> staples;
	double slope1, slope2;
	int size1 = points1.size()>40?40:points1.size();  // move to 20
	int size2 = points2.size()>40?40:points2.size();
	for(int j=1; j < size1 ; j++)
	{
		for(int i=0; i < j ; i++)  // curvature of i is greater than j
		{
			dist1 = StructureDistance(points1[i],points1[j]);
			slope1 = StructureSlope(points1[i],points1[j]);
			for(int n=1; n < size2; n++)
			{
				for(int m=0; m < n ; m++)	// curvature of m is greater than n
				{
					dist2 = StructureDistance(points2[m],points2[n]); 
					slope2 = StructureSlope(points2[m],points2[n]);
					//changed the slope tollerance value to absolute value
					if(Tool::getInstance()->approxComp(dist1,dist2,dist1*dist_tollerence) == 0 && Tool::getInstance()->approxComp(slope1,slope2,slope1*slope_tollerence)==0 )// && Tool::getInstance()->approxComp(points1[i].curvature + points2[j].curvature, points2[m].curvature + points2[n].curvature, (points2[i].curvature + points2[j].curvature)*curvature_tollerence) == 0)
					{
						staple tempStaple(points1[i],points1[j],points2[m],points2[n]);//p1's x is greater than p2's x in each staple
						tempStaple = getFeatureMatch(tempStaple,points1,points2,slope1,slope2,dist_tollerence,slope_tollerence);
						staples.push_back(tempStaple);
						/*if(tempStaple.NumOfMatch >= (points1.size()+points2.size())/6){
							printf("matching found");
							goto hehe;
						}*/
						if(staples.size() >= 100 )
						{
							//printf("Staple info::\nimage-1 : \n----------\nx1=%d, y1=%d, X2= %d , Y2= %d :: Distance = %f\n",staples[t].p1_img1.x,staples[t].p1_img1.y,staples[t].p2_img1.x,staples[t].p2_img1.y,StructureDistance(staples[t].p1_img1,staples[t].p2_img1));
							//printf("Staple info::\nimage-2 : \n----------\nx1=%d, y1=%d, X2= %d , Y2= %d :: Distance = %f\n",staples[t].p1_img2.x,staples[t].p1_img2.y,staples[t].p2_img2.x,staples[t].p2_img2.y,StructureDistance(staples[t].p1_img2,staples[t].p2_img2));
							printf("200 staples checked");
							goto hehe;
						}
					}
	
				}
			}
		}
	}

hehe:

	return staples;
}
//****************************************************************************************************
void Tool::rotateImage(const Mat &input, Mat &output, double alpha, double beta, double gamma, double dx, double dy, double dz, double f)
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
void Tool::rotate(cv::Mat& src, double angle, double scaleFactor, cv::Mat& dst)
{
    int len = std::max(src.cols, src.rows);
    cv::Point2f pt(len/2., len/2.);
	cv::Mat r = cv::getRotationMatrix2D(pt, angle, scaleFactor);

    cv::warpAffine(src, dst, r, cv::Size(len, len));
}

//****************************************************************************************************
vector<edge> Tool::removeSmallCurves(int th, Mat img,vector<edge> edges,vector<point> junction_pts,int curveLenThreshold,int BWThreshold,char* outputPath1_short){
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
	imwrite(outputPath1_short,img);
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
vector<edge> Tool::GetCurvature(Mat temp,int k,int curveLenThreshold,int BWThreshold,char* outputPath1_short){
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

	edges = removeSmallCurves(curveLenThreshold,temp,edges,junction_pts,curveLenThreshold,BWThreshold,outputPath1_short);
	//imwrite(outputPath1_short,temp);
	edge junction = edges.back();
	edges.pop_back();

	edges = Tool::getInstance()->calculateCurvature(k,edges);
	edges.push_back(junction);
	//imwrite("E:/personal/acads/BTP/images/Set1/scaled/set18_smallEdgesRemoved.png",temp);
	return edges;
}