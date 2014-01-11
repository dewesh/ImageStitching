#include "StdAfx.h"
#include "staple.h"

/**
p1's x is greater than p2's x in each staple
*/
staple::staple(point p1_1,point p2_1, point p1_2, point p2_2 )
{

		if(p1_1.x >= p2_1.x){
			p1_img1 = p1_1;
			p2_img1 = p2_1;
		}
		else{
			p1_img1 = p2_1;
			p2_img1 = p1_1;
		}

		if(p1_2.x >= p2_2.x){
			p1_img2 = p1_2;
			p2_img2 = p2_2;
		}
		else{
			p1_img2 = p2_2;
			p2_img2 = p1_2;
		}
		NumOfMatch =0;
}

staple::staple(void)
{
}

staple::~staple(void)
{
}
